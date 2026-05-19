# Technical Memo: Local LM Studio Backend Integration

## 1. Problem Definition

This repository is a small-scale controlled benchmark for profiling LLM agent tool-use efficiency, reliability, and guardrail failure cases. The benchmark is intentionally local-first, deterministic where possible, and limited to `24` fixed tasks.

Why this matters:

- final-answer accuracy alone is not enough
- tool-use process errors can be hidden by seemingly correct outputs
- single-label failures can hide compound issues

The benchmark therefore logs oracle checks, per-task traces, and compound failure flags.

## 2. Benchmark Setup

Task split remains fixed:

- `8` tool_use
- `8` multi_step
- `8` guardrail

Core components:

- `benchmark.py`: runner, backend selection, CSV logging
- `agent.py`: `RuleBasedAgent` and `LocalLLMAgent`
- `llm_clients.py`: LM Studio REST v1 client (`/api/v1/chat`)
- `tools.py`: deterministic mock tools
- `guardrails.py`: deterministic sensitive-data checks
- `evaluator.py`: oracle checks + primary failure + compound flags
- `tracing.py`: JSONL trace writer
- `analysis.py`: aggregation and plotting

## 3. Local LLM Backend

### 3.1 Why LM Studio was added

A local LLM backend was added to evaluate real model-driven tool planning without requiring external cloud APIs. This preserves offline operation and reduces network/provider variability.

### 3.2 Why rule_based baseline is retained

The rule-based backend remains necessary as a deterministic sanity baseline:

- validates benchmark mechanics
- provides stable regression control
- isolates evaluator or schema regressions from model variability

### 3.3 What local backend measures

The local backend can measure:

- model-driven tool selection quality
- tool argument quality
- structured-output robustness (JSON action protocol)
- guardrail outcomes under model-generated answers
- local end-to-end latency profiles

### 3.4 What local backend does not measure

It does not measure hidden model reasoning quality directly. The benchmark reports observable traces and oracle outcomes only.

## 4. Oracle Evaluation and Failure Taxonomy

Success is decomposed into explicit checks:

- final-answer correctness
- required tool usage
- tool sequence correctness
- tool argument correctness
- tool execution success
- planning success
- format correctness
- contains/excludes constraints
- guardrail success

Failure reporting has two layers:

- `failure_type`: one primary category for aggregate plots
- `failure_flags`: all detected failure conditions for compound error analysis

This design avoids under-reporting multi-cause failures.

## 5. Trace Methodology

Each task writes a JSONL trace with observable events:

- task_start
- agent_decision
- tool_call
- tool_result
- guardrail_check
- evaluation
- task_end

For LM Studio mode, agent decision events include backend/model/phase metadata and compact model output previews.

Methodology note:

The benchmark reports observable tool-use traces and oracle-level evaluation results. It does not expose hidden model reasoning. Local LLM traces record prompts, compact action decisions, tool calls, tool results, guardrail checks, and evaluator outputs.

## 6. Experimental Results

### 6.1 Rule-based results (generated)

From `results_rule_based.csv` (latest run):

- total tasks: `24`
- overall success: `100%`
- failure type distribution: `none=24`
- guardrail false positives: `0`
- guardrail false negatives: `0`
- average wall-clock latency by task type (ms):
  - tool_use: `0.0639875`
  - multi_step: `0.0656`
  - guardrail: `0.0509`
- average tool latency by task type (ms):
  - tool_use: `0.018175`
  - multi_step: `0.0265125`
  - guardrail: `0.01805`

### 6.2 LM Studio results status

LM Studio backend support is implemented in code, including endpoint configuration, preflight check, structured two-phase prompting, and trace integration. LM Studio result generation depends on local server availability.

If `results_lmstudio.csv` is not present for a run session, report status as implemented but not generated.

## 7. Variance and Reproducibility

Using local LM Studio reduces:

- network-induced latency variance
- provider queue/rate-limit effects
- provider-side silent model updates

Remaining variance sources include:

- local hardware load
- model load state and caching
- quantization/runtime backend differences
- context length and decoding behavior
- thermal throttling
- LM Studio server overhead

Token and cost values are approximate heuristic estimates for relative profiling, not billing-accurate metering.

## 8. Limitations and Extensions

Current limitations:

- small benchmark size (`24` tasks)
- deterministic mock tools
- regex-style guardrails
- no hidden-reasoning access

Next extensions:

- generate and compare `results_lmstudio.csv` under fixed local setup
- compare multiple local models/settings
- add CI checks for backend-specific regressions
- add stricter per-field output validators for LM-generated JSON actions
