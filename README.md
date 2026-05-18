# Mini Agent Tool-Use Benchmark

This repository implements a small-scale benchmark for evaluating LLM agent tool-use efficiency and failure cases. It focuses on task success rate, wall-clock latency, tool-call latency, token/cost usage, invalid tool calls, retries, and simple guardrail violations.

## Motivation

Many agent evaluations are difficult to reproduce because they depend on external APIs, changing models, and non-deterministic environments. This project provides a deterministic local baseline with mock tools so benchmark mechanics can be tested quickly before introducing real LLMs.

## Benchmark Design

- `24` total tasks in `tasks.json`
- Deterministic rule-based baseline agent (`agent.py`)
- Local mock tools only (`tools.py`)
- Simple guardrail checker for PII (`guardrails.py`)
- End-to-end runner that writes `results.csv` (`benchmark.py`)
- Offline analysis and figures (`analysis.py`)

## Task Types

- `tool_use` (`8`): single-tool operations (calculator, lookup, JSON path extraction)
- `multi_step` (`8`): chained tool calls and intermediate reasoning
- `guardrail` (`8`): policy checks and PII-safe output formatting

## Metrics

`results.csv` includes:

- `task_id`, `task_type`, `success`
- `wall_clock_time_ms`, `tool_latency_ms`
- `tool_call_count`, `invalid_tool_call_count`, `retry_count`
- `input_tokens`, `output_tokens`, `cost_usd`
- `guardrail_checked`, `guardrail_violation`
- `failure_type`, `notes`

## Failure Types

- `none`
- `planning_error`
- `tool_misuse`
- `wrong_calculation`
- `hallucinated_result`
- `policy_miss`
- `format_error`

## Repository Structure

```text
mini-agent-tooluse-benchmark/
├── README.md
├── benchmark.py
├── agent.py
├── tools.py
├── guardrails.py
├── tasks.json
├── results.csv
├── analysis.py
├── figures/
│   ├── latency_by_task_type.png
│   ├── success_rate_by_task_type.png
│   └── failure_type_distribution.png
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

- `results.csv` with one row per task (`24` rows)

## Run Analysis

```bash
conda run -n agent python analysis.py
```

Outputs:

- `figures/latency_by_task_type.png`
- `figures/success_rate_by_task_type.png`
- `figures/failure_type_distribution.png`

## Current Baseline Outputs

Using the included deterministic baseline:

- Overall success rate: `95.83%` (`23/24`)
- Success by type: `tool_use=100%`, `multi_step=100%`, `guardrail=87.5%`
- Observed failure types: `none`, `policy_miss`

## Limitations

- This baseline is rule-based, not a real LLM agent.
- Mock tools are simplified and in-memory.
- Guardrails cover only simple email/phone detection.
- Token counts and cost are approximate.

## Next Steps

- Add pluggable model backends for real LLM inference.
- Expand task diversity and adversarial guardrail cases.
- Add repeated runs, confidence intervals, and regression checks.
- Add stricter output-format validators per task.
