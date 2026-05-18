# Mini Agent Tool-Use Benchmark

This repository implements a small-scale benchmark for evaluating LLM agent tool-use efficiency and failure cases. It focuses on task success rate, wall-clock latency, tool-call latency, token/cost usage, invalid tool calls, retries, and simple guardrail violations.

## Motivation

Simple final-answer matching can hide important agent failures. An answer may look correct while the agent skipped required tools, called tools in the wrong order, used wrong arguments, or leaked sensitive fields in a guardrail task. This benchmark keeps the setup small and deterministic, then adds explicit oracle checks so tool-use correctness and safety correctness are measured directly.

## Benchmark Design

- Exactly `24` tasks in `tasks.json`
- Deterministic local mock tools (`tools.py`)
- Deterministic rule-based baseline agent (`agent.py`)
- Deterministic guardrail checker (`guardrails.py`)
- Oracle evaluator (`evaluator.py`) with explicit sub-checks
- End-to-end runner (`benchmark.py`) that writes `results.csv`
- Offline analysis (`analysis.py`) and figures (`figures/`)

## Task Types

- `tool_use` (`8`): single-tool operations and strict tool-use constraints
- `multi_step` (`8`): chained tool calls with expected sequence and argument checks
- `guardrail` (`8`): redaction, refusal, and false-positive control cases

## Why Final Answer Is Insufficient

`success` is not just `expected_answer == final_answer`. The evaluator decomposes success into:

- final-answer correctness
- required tool calls
- tool sequence match
- tool argument match
- planning success
- format correctness
- contains/excludes constraint match
- guardrail success (for guardrail-required tasks)

## Oracle-Level Checks

For each task, `results.csv` logs:

- `final_answer_correct`
- `required_tools_called`
- `tool_sequence_match`
- `tool_argument_match`
- `planning_success`
- `format_correct`
- `contains_excludes_match`
- `guardrail_success`
- `false_positive`
- `false_negative`
- `leaked_pii_types`

## Metrics

`results.csv` includes operational metrics and oracle metrics:

- identity: `task_id`, `task_type`, `task_subtype`
- outcome: `success`, `failure_type`
- oracle checks: the fields listed above
- efficiency: `wall_clock_time_ms`, `tool_latency_ms`, `tool_call_count`, `invalid_tool_call_count`, `retry_count`
- usage proxy: `input_tokens`, `output_tokens`, `cost_usd`
- safety flags: `guardrail_checked`, `guardrail_violation`
- diagnostics: `notes`, `eval_notes`

## Failure Types

- `none`
- `planning_error`
- `tool_misuse`
- `wrong_calculation`
- `answer_mismatch`
- `hallucinated_result`
- `policy_miss`
- `format_error`

## Repository Structure

```text
mini-agent-tooluse-benchmark/
├── README.md
├── benchmark.py
├── evaluator.py
├── agent.py
├── tools.py
├── guardrails.py
├── tasks.json
├── results.csv
├── analysis.py
├── figures/
│   ├── latency_by_task_type.png
│   ├── success_rate_by_task_type.png
│   ├── failure_type_distribution.png
│   └── oracle_metric_breakdown.png
├── requirements.txt
└── memo.md
```

## Setup

```bash
conda create -n agent python=3.11 -y
mamba install -n agent -y pandas matplotlib
```

## Run Benchmark

```bash
conda run -n agent python benchmark.py
```

Output:

- `results.csv` with exactly `24` data rows

## Run Analysis

```bash
conda run -n agent python analysis.py
```

Outputs:

- `figures/latency_by_task_type.png`
- `figures/success_rate_by_task_type.png`
- `figures/failure_type_distribution.png`
- `figures/oracle_metric_breakdown.png` (optional extension)

## Current Baseline Snapshot

From the latest deterministic baseline run:

- overall success: `100%` (`24/24`)
- guardrail false positives: `0`
- guardrail false negatives: `0`

## Limitations

- This is a small-scale controlled benchmark, not a full-scale agent benchmark.
- The baseline agent is rule-based; no real LLM API is used.
- Tools are deterministic mocks with limited domain coverage.
- Guardrail detection is regex-based and intentionally simple.
- The benchmark uses deterministic mock tools and a rule-based baseline agent. Token and cost values are approximate estimates based on a simple token-counting heuristic and fixed mock pricing. They are intended for relative profiling, not provider-level billing accuracy.

## Next Steps

- Add pluggable model-backed agents under the same task/oracle interface.
- Add regression thresholds in CI for oracle sub-metrics.
- Add adversarial tool-use and guardrail stress cases while keeping deterministic controls.
