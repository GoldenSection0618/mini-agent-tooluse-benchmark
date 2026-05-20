# Technical Memo: Mini Agent Tool-Use Benchmark

## 1. Problem Definition

Final-answer accuracy alone hides critical agent process failures. In tool-use settings, an answer can look correct while the agent skips required tools, uses wrong arguments, violates multi-step order, or fails policy constraints.

This repository is a proof-of-work benchmark harness for execution-level diagnostics. It focuses on:

- tool-use protocol compliance,
- runtime profiling,
- guardrail failure analysis,
- reproducible trace-based evaluation.

It is not a comprehensive benchmark of general LLM capability.

## 2. Experimental Setup

Benchmark shape is fixed:

- 24 tasks total
- 8 `tool_use`, 8 `multi_step`, 8 `guardrail`
- deterministic mock tools (`calculator_tool`, `file_lookup_tool`, `json_parser_tool`, `policy_checker_tool`)
- deterministic guardrail checker
- sequential task execution (no task-level parallelism)

Backends:

- `rule_based` (deterministic sanity baseline)
- `lmstudio` (local model through LM Studio REST)
- `deepseek` (cloud model through OpenAI-compatible chat endpoint)

Comparable generation defaults:

- `temperature=0`
- `max_tokens=256`

Outputs:

- task-level CSV results under `results/`
- run metadata under `results/metadata/`
- per-task traces under `traces/<backend>/`
- backend and comparison figures/summaries under `figures/`
- canonical examples include `traces/rule_based/` and `figures/compare/`

## 3. Metric Design

### Outcome metrics

- `success`
- `final_answer_correct`

### Process metrics

- `required_tools_called`
- `tool_sequence_match`
- `tool_argument_match`
- `tool_execution_success`
- `planning_success`
- `format_correct`
- `contains_excludes_match`

### Efficiency metrics

- `wall_clock_time_ms`
- `request_latency_ms`
- `tool_latency_ms`
- token/cost estimates (`input_tokens`, `output_tokens`, `cost_usd`, provider usage fields)

### Safety metrics

- `guardrail_violation`
- `false_positive`
- `false_negative`
- `leaked_pii_types`

The design separates final-answer correctness from process correctness, which is essential for diagnosing tool-use behavior.

## 4. Results

Snapshot from current canonical outputs:

- `results/rule_based.csv`
- `results/lmstudio.csv`
- `results/deepseek.csv`

| Backend | Tasks | Success | Tool-use | Multi-step | Guardrail | Avg wall-clock ms | Avg request latency ms | Main failure types |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| rule_based | 24 | 100.00% | 100.00% | 100.00% | 100.00% | 0.0526 | 0.0000 | none:24 |
| lmstudio | 24 | 0.00% | 0.00% | 0.00% | 0.00% | 30641.5679 | 30641.2632 | hallucinated_result:14, tool_misuse:10 |
| deepseek | 24 | 25.00% | 75.00% | 0.00% | 0.00% | 1617.0674 | 1130.0354 | hallucinated_result:13, none:6, tool_misuse:4 |

Interpretation:

- `rule_based` validates harness/evaluator/tracing integrity.
- `lmstudio` and `deepseek` runs expose process failures dominated by missing/wrong tool usage and hallucinated completion without required tool flow.
- Latency values are system-level end-to-end latency, not pure model compute time.

## 5. Failure Case Taxonomy

Primary `failure_type` categories:

- `planning_error`
- `tool_misuse`
- `wrong_calculation`
- `answer_mismatch`
- `hallucinated_result`
- `policy_miss`
- `format_error`

Compound `failure_flags` preserve secondary causes in the same task, e.g.:

- `required_tool_missing`
- `tool_sequence_mismatch`
- `tool_argument_mismatch`
- `hallucinated_without_tool`
- `llm_invalid_json`

This dual view (primary class + compound flags) provides more diagnostic value than a single accuracy score.

## 6. Limitations and Next Steps

Current limitations:

- small controlled task set (24 tasks)
- mock tools instead of real external APIs
- deterministic regex/pattern guardrail checker
- token/cost values are approximate profiling signals
- no repeated-run confidence intervals yet

Practical next steps:

- add repeated-run variance and confidence intervals by backend
- introduce harder but still deterministic task variants
- add optional real tool adapters behind the same evaluator contract
- strengthen policy checker coverage (while keeping deterministic baseline checks)
- expand backend matrix while preserving sequential reproducibility constraints
