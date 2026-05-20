# Technical Memo: Three-Backend Benchmark Design

## 1. Problem Definition

This project is a small-scale controlled benchmark for profiling LLM agent tool-use efficiency, reliability, and guardrail failure cases. It is intentionally constrained to `24` fixed tasks so benchmark behavior is easy to inspect and reproduce.

The core evaluation issue is that final-answer correctness alone is insufficient. An agent can still fail by:

- skipping required tools,
- using wrong tools or wrong argument values,
- violating sequence constraints in multi-step tasks,
- leaking policy-forbidden sensitive data.

The benchmark therefore evaluates both outcomes and process signals: oracle checks, structured traces, and compound failure flags.

## 2. Backend Design

The benchmark uses one unified runner and three interchangeable backends.

### 2.1 rule_based

- deterministic sanity backend
- no external API dependency
- validates runner/evaluator/tracing correctness
- provides stable regression control

### 2.2 lmstudio

- local real LLM backend via LM Studio REST endpoint (`/api/v1/chat`)
- exercises real model behavior under local inference/runtime
- preserves compatibility with offline/local workflows

### 2.3 deepseek

- cloud backend via DeepSeek OpenAI-compatible Chat Completions endpoint (`/chat/completions`)
- uses `DEEPSEEK_API_KEY` from environment variables
- reflects cloud API deployment conditions (network + provider-side effects)

## 3. Why Three Backends

Using one backend only makes diagnosis ambiguous.

- `rule_based` isolates harness logic and catches evaluator/tracing regressions quickly.
- `lmstudio` tests realistic model-driven tool use while reducing external API variance.
- `deepseek` tests a cloud API path closer to production-style deployment constraints.

This backend split improves interpretability of failures without expanding task count.

## 4. Controlled Evaluation Setup

Task set remains fixed:

- `8` tool_use
- `8` multi_step
- `8` guardrail

Execution is strictly sequential: one task at a time, one trace file per task, one CSV row per task.

Oracle success is decomposed into explicit checks:

- `final_answer_correct`
- `required_tools_called`
- `tool_sequence_match`
- `tool_argument_match`
- `tool_execution_success`
- `planning_success`
- `format_correct`
- `contains_excludes_match`
- `guardrail_success`

Failure reporting has two layers:

- `failure_type`: primary aggregate category
- `failure_flags`: compound machine-readable error flags

Trace events record observable execution only:

- `task_start`
- `agent_decision`
- `tool_call`
- `tool_result`
- `guardrail_check`
- `evaluation`
- `task_end`

## 5. Latency Interpretation

Latency across backends is not a pure model-compute comparison.

- `rule_based` latency is mainly local Python/tool execution overhead.
- `lmstudio` latency includes local model runtime, loading state, and local hardware contention.
- `deepseek` latency includes client-side network transport and provider-side queue/runtime effects.

Therefore, cross-backend latency is a system-level measurement, not an apples-to-apples model-speed metric.

## 6. Experimental Results

### 6.1 Rule-based snapshot (latest generated)

From `results_rule_based.csv`:

- total tasks: `24`
- overall success: `100%`
- failure types: `none=24`
- guardrail false positives: `0`
- guardrail false negatives: `0`
- average wall-clock latency by task type (ms):
  - `tool_use`: `0.046825`
  - `multi_step`: `0.0480375`
  - `guardrail`: `0.034`
- average tool latency by task type (ms):
  - `tool_use`: `0.0162`
  - `multi_step`: `0.0154`
  - `guardrail`: `0.013`

### 6.2 LM Studio snapshot (latest generated)

From `results_lmstudio.csv`:

- total tasks: `24`
- overall success: `0%`
- failure type distribution:
  - `hallucinated_result`: `19`
  - `tool_misuse`: `5`
- average wall-clock latency by task type (ms):
  - `tool_use`: `14020.7443875`
  - `multi_step`: `8025.963875`
  - `guardrail`: `8028.3472`

This snapshot shows backend availability and instrumentation are working, while action JSON reliability is currently the main bottleneck for this model/runtime setup.

### 6.3 DeepSeek snapshot (latest generated)

From `results_deepseek.csv`:

- total tasks: `24`
- overall success: `20.83%` (`5/24`)
- failure type distribution:
  - `hallucinated_result`: `19`
  - `none`: `5`
- average wall-clock latency by task type (ms):
  - `tool_use`: `4378.8642`
  - `multi_step`: `5615.483925`
  - `guardrail`: `5131.514625`

## 7. Failure Taxonomy Across Backends

The taxonomy is designed to preserve both top-level and compound signals.

Primary categories (`failure_type`) include:

- `planning_error`
- `tool_misuse`
- `wrong_calculation`
- `answer_mismatch`
- `hallucinated_result`
- `policy_miss`
- `format_error`

Compound flags (`failure_flags`) can preserve secondary causes in a single task, e.g. required tool missing + answer mismatch + disallowed tool request.

This is important for backend comparisons because different backends may fail for different combinations even when final-answer accuracy looks similar.

## 8. Limitations

- small fixed task set (`24`) is useful for control but limited for broad capability claims
- tools are deterministic mocks, not external production systems
- guardrails are deterministic pattern/policy checks, not full privacy classifiers
- traces are observable event traces only and do not expose hidden model reasoning
- token/cost values are approximate estimates for relative profiling, not provider billing truth

## 9. Extensions

- run repeated LM Studio and DeepSeek trials for variance envelopes
- add stricter JSON action schema validation diagnostics
- add backend-specific regression thresholds in CI
- expand policy stress tests while keeping deterministic oracles
