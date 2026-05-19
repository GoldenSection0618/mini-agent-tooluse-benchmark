# Technical Memo: Tracing and Compound Failure Upgrade

## 1. Problem Definition

This project is a small-scale controlled benchmark for profiling LLM agent tool-use efficiency, reliability, and guardrail failure cases. The benchmark is intentionally local and deterministic so evaluator behavior, logging behavior, and analysis behavior can be audited without external API variance.

Core problem addressed in this upgrade:

- final-answer accuracy alone is insufficient for agent evaluation
- single-label failure summaries can hide compound errors
- lack of execution trace makes debugging difficult

Scope constraints remain unchanged:

- exactly `24` tasks
- `8` `tool_use`, `8` `multi_step`, `8` `guardrail`
- deterministic mock tools
- deterministic baseline rule-based agent
- no real LLM API requirement

## 2. Benchmark Setup

Main components:

- `tasks.json`: fixed task definitions with oracle metadata
- `agent.py`: deterministic baseline tool-using agent
- `tools.py`: deterministic mock tools
- `guardrails.py`: deterministic pattern-based sensitive-data checker
- `evaluator.py`: oracle checks + primary failure type + compound failure flags
- `tracing.py`: JSONL trace event helpers
- `benchmark.py`: task execution, trace writing, result logging
- `analysis.py`: aggregate metrics and figures
- `inspect_trace.py`: single-trace inspection helper

Execution flow:

1. validate task schema
2. execute agent task
3. run guardrail checks
4. run evaluator checks
5. write per-task trace file (`traces/{task_id}.jsonl`)
6. append row to `results.csv`
7. aggregate with `analysis.py`

## 3. Task Oracle Design

Success is decomposed into explicit checks rather than final answer only:

- `final_answer_correct`
- `required_tools_called`
- `tool_sequence_match`
- `tool_argument_match`
- `planning_success`
- `format_correct`
- `contains_excludes_match`
- `guardrail_success` (where required)

This prevents “correct answer by incorrect process” from being counted as full success.

## 4. Trace Design

Each task writes a JSONL trace under `traces/`.

Required event types:

- `task_start`
- `agent_decision`
- `tool_call`
- `tool_result`
- `guardrail_check`
- `evaluation`
- `task_end`

Each event contains base fields (`task_id`, `step`, `event_type`, `timestamp`) and optional payload fields depending on event type.

Trace metrics logged in `results.csv`:

- `trace_file`
- `agent_step_count`
- `tool_error_count`

From regenerated results:

- average `agent_step_count` by task type:
  - `tool_use`: `5.0`
  - `multi_step`: `8.25`
  - `guardrail`: `6.0`
- `tool_error_count` summary:
  - total tool errors: `8`
  - tasks with tool errors: `8`

## 5. Failure Taxonomy

Two layers are logged:

- `failure_type`: one primary category for high-level aggregation
- `failure_flags`: list of all detected machine-readable issues

Primary categories currently used:

- `none`
- `planning_error`
- `tool_misuse`
- `wrong_calculation`
- `answer_mismatch`
- `hallucinated_result`
- `policy_miss`
- `format_error`

## 6. Failure Flags and Compound Errors

`failure_flags` preserves secondary failure modes that a single primary label cannot represent.

Examples of supported flags:

- `required_tool_missing`
- `unexpected_tool_used`
- `tool_sequence_mismatch`
- `tool_argument_mismatch`
- `wrong_numeric_answer`
- `answer_mismatch`
- `format_mismatch`
- `contains_required_text_missing`
- `excluded_text_leaked`
- `guardrail_false_positive`
- `guardrail_false_negative`
- `pii_leak_email`
- `pii_leak_phone`
- `pii_leak_address`
- `pii_leak_id`
- `hallucinated_without_tool`

Current run outcome (from regenerated `results.csv`):

- `24/24` tasks successful
- failure type distribution: `none=24`
- failure flags distribution: empty (`{}`)
- guardrail false positives: `0`
- guardrail false negatives: `0`

Even with no current failures, flags remain useful for future model-backed regressions where one task may exhibit multiple simultaneous faults.

## 7. Metrics, Results, and Limitations

Latest regenerated run (`results.csv`):

- overall success: `100.0%` (`24/24`)
- by-task-type success: all `100.0%`
- average wall-clock latency (ms):
  - `tool_use=0.0579125`
  - `multi_step=0.05165`
  - `guardrail=0.0297`
- average tool latency (ms):
  - `tool_use=0.00975`
  - `multi_step=0.013775`
  - `guardrail=0.00045`
- estimated total cost: `$0.000873`

Limitations:

- deterministic baseline agent is not equivalent to open-ended LLM behavior
- traces capture observable execution events, not hidden reasoning
- mock tools and regex guardrails are intentionally simple
- token/cost values are heuristic estimates for relative profiling, not provider billing
- benchmark size is intentionally small (`24` tasks)

This benchmark should be treated as a controlled profiling harness, not as a full-scale real-world capability benchmark.
