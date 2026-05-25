# Mini Agent Tool-Use Benchmark

![Python](https://img.shields.io/badge/Python-3.11-blue)
![Tasks](https://img.shields.io/badge/Tasks-24-informational)
![Backends](https://img.shields.io/badge/Backends-3-informational)
![Tracing](https://img.shields.io/badge/Tracing-JSONL-success)
![Status](https://img.shields.io/badge/Status-Proof--of--Work-orange)

This repository implements a small-scale benchmark for evaluating **LLM agent tool-use efficiency and failure cases** under controlled, reproducible conditions.

It focuses on task success rate, wall-clock latency, tool-call latency, token/cost usage, invalid tool calls, retries, guardrail violations, and trace-based diagnostics. The goal is to evaluate not only whether the final answer is correct, but whether the agent follows the required tool-use process.

## Motivation

Final-answer matching alone can hide process failures. An answer may look correct while the agent skips required tools, calls tools in the wrong order, uses wrong arguments, or leaks sensitive fields. This benchmark keeps setup local and deterministic, then adds oracle checks and lightweight traces to measure both outcome and process quality.

Low LLM success rates are expected in this benchmark: they expose process-level tool-use failures such as invalid JSON actions, missing required tools, wrong tool order, wrong arguments, and safe-but-incomplete guardrail responses.

## Contents

- [Current Result Snapshot](#current-result-snapshot)
- [Results Overview](#results-overview)
- [Backends](#backends)
- [Benchmark Design](#benchmark-design)
- [Task Types](#task-types)
- [Oracle Checks](#oracle-checks)
- [Tracing](#tracing)
- [Failure Taxonomy](#failure-taxonomy)
- [Setup](#setup)
- [Run Commands](#run-commands)
- [Output Artifacts](#output-artifacts)
- [Canonical Artifacts](#canonical-artifacts)
- [Reproducibility Note](#reproducibility-note)
- [Limitations](#limitations)

## Current Result Snapshot

> **TL;DR.** Final-answer correctness can overestimate agent reliability.  
> In the current run, DeepSeek solves simple single-tool tasks but fails strict multi-step and guardrail tasks under the process oracle. LM Studio / Gemma shows broader instability at the structured-output and tool-protocol layer. The rule-based backend is only an oracle sanity check confirming that the benchmark pipeline is executable.

| Backend | Model | Tasks | Success | Tool-use | Multi-step | Guardrail | Avg wall-clock ms | Main failure modes |
|---|---|---:|---:|---:|---:|---:|---:|---|
| rule_based | rule_based | 24 | 100.00% | 100.00% | 100.00% | 100.00% | 0.04 | no failures |
| lmstudio | google/gemma-4-e4b | 24 | 8.33% | 25.00% | 0.00% | 0.00% | 25105.48 | tool_misuse:13, hallucinated_result:7, wrong_calculation:2 |
| deepseek | deepseek-v4-flash | 24 | 33.33% | 100.00% | 0.00% | 0.00% | 1922.79 | tool_misuse:10, answer_mismatch:5, planning_error:1 |

Snapshot rates and latency values are sourced from [summary_by_backend.csv](figures/compare/summary_by_backend.csv) and [summary_by_backend_and_task_type.csv](figures/compare/summary_by_backend_and_task_type.csv). Failure counts are sourced from [failure_type_by_backend.csv](figures/compare/failure_type_by_backend.csv).

## Results Overview

The figures below should be read as process-diagnostic evidence. They show where agents fail in the execution protocol: missing tools, wrong tool order, wrong arguments, invalid structured actions, incomplete guardrail responses, or final-answer mismatches.

![Success rate by backend](figures/compare/success_rate_by_backend.png)

*Caption: The oracle sanity backend reaches 100%, confirming the task/evaluator pipeline is executable. LLM backend scores are lower because the oracle checks process compliance, not only final answers.*

![Success rate by backend and task type](figures/compare/success_rate_by_backend_and_task_type.png)

*Caption: DeepSeek succeeds on simple `tool_use` tasks but fails all `multi_step` and `guardrail` tasks under the current strict process oracle. LM Studio / Gemma has failures across all task types, with partial success only on single-tool tasks.*

![Failure type by backend](figures/compare/failure_type_by_backend.png)

*Caption: Primary failure types show the high-level category for each failed task. `none` means a successful task, not a failure mode.*

![Failure flags by backend](figures/compare/failure_flags_by_backend.png)

*Caption: Compound failure flags preserve secondary causes such as invalid JSON actions, missing required tools, wrong tool order, wrong arguments, and missing required text.*

![Latency by backend](figures/compare/latency_by_backend.png)

*Caption: Latency is end-to-end system latency. Local LM Studio, DeepSeek, and rule-based runs have different runtime sources and should not be read as pure model compute speed.*

## Backends

Three backends are supported:

- `rule_based`: oracle sanity backend for validating evaluator/tracing/failure taxonomy (not a competitive baseline)
- `lmstudio`: local LLM backend through LM Studio REST API v1 (reduced external variance, still local runtime variance)
- `deepseek`: cloud LLM backend through DeepSeek OpenAI-compatible Chat Completions API (includes network/provider-side effects)

## LM Studio Setup

Start LM Studio local server and ensure this endpoint is available:

`POST http://localhost:1234/api/v1/chat`

Example test:

```bash
curl http://localhost:1234/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{
    "model": "google/gemma-4-e4b",
    "system_prompt": "Reply with OK only.",
    "input": "ping"
  }'
```

## Benchmark Design

- Exactly `24` tasks in `tasks.json`
- Deterministic local mock tools (`tools.py`)
- Deterministic oracle sanity backend (`RuleBasedAgent`)
- Local/cloud LLM backends through `ToolCallingLLMAgent` + `llm_clients.py`
- Deterministic guardrail checker (`guardrails.py`)
- Oracle evaluator with explicit sub-checks (`evaluator.py`)
- Per-task JSONL traces (`tracing.py` + `traces*/`)
- Analysis and figures (`analysis.py`)

The current results measure LLM backends under an explicit tool-schema and task-context contract.

## Task Types

- `tool_use` (`8`)
- `multi_step` (`8`)
- `guardrail` (`8`)

## Task Design Rationale

The tasks are intentionally small but process-sensitive. This benchmark is not designed to measure broad reasoning ability; it is designed to isolate tool-use protocol compliance failures under controlled conditions. A final answer can be numerically correct while still failing required-tool, sequence, or argument checks. Core failure modes include skipped tools, wrong tool order, wrong arguments, over-calling/disallowed calls, JSON parsing failures, hallucinated results without tools, and policy leakage.

## Oracle Checks

Success is decomposed into:

- `final_answer_correct`
- `required_tools_called`
- `tool_sequence_match`
- `tool_argument_match`
- `tool_execution_success`
- `planning_success`
- `format_correct`
- `contains_excludes_match`
- `guardrail_success` (for guardrail-required tasks)

Guardrail task success and output policy cleanliness are separated:

- `success`: full task-level pass/fail under the oracle checks
- `output_policy_clean`: whether the final output avoids policy-forbidden leakage
- `guardrail_success`: backward-compatible alias for `output_policy_clean`; it is not full task success

A model can produce a policy-clean output while still failing the requested guardrail task.

## Tracing

Each task run produces a JSONL trace at `traces/<backend>/<task_id>.jsonl` by default (or custom `--trace-dir`), for example `traces/rule_based/ms_01.jsonl`.

Trace events include:

- `task_start`
- `agent_decision`
- `tool_call`
- `tool_result`
- `guardrail_check`
- `evaluation`
- `task_end`

This is a lightweight execution trace, not a full reasoning trace.

This benchmark provides end-to-end execution-level explainability through structured traces, oracle checks, tool-call records, guardrail checks, and failure flags. It does not expose or depend on hidden model reasoning.

In `guardrail_check` events:

- `source_contains_sensitive_data`: source/input contains sensitive data
- `output_contains_forbidden_data`: agent output contains policy-forbidden data

## Failure Taxonomy

- `failure_type`: one primary failure category for aggregation
- `failure_flags`: all detected failure conditions for compound analysis

Interpretation distinguishes tool schema violations, missing required tools, wrong arguments, wrong final answers, policy leakage, and safe-but-incomplete or context-misunderstood responses, instead of collapsing all failures into a single hallucination bucket.

## Setup

```bash
conda create -n agent python=3.11 -y
mamba install -n agent -y pandas matplotlib
```

## Run Commands

<details>
<summary>Rule-based oracle sanity backend</summary>

```bash
python benchmark.py --agent rule_based
python analysis.py --input results/rule_based.csv --figures-dir figures/rule_based
```

</details>

<details>
<summary>LM Studio local LLM backend</summary>

```bash
python benchmark.py \
  --agent lmstudio \
  --base-url http://localhost:1234 \
  --chat-endpoint /api/v1/chat \
  --model google/gemma-4-e4b \
  --output results/lmstudio.csv \
  --trace-dir traces/lmstudio

python analysis.py --input results/lmstudio.csv --figures-dir figures/lmstudio
```

</details>

<details>
<summary>DeepSeek cloud LLM backend</summary>

```bash
export DEEPSEEK_API_KEY="your_api_key_here"

python benchmark.py \
  --config config.example.json \
  --agent deepseek \
  --base-url https://api.deepseek.com \
  --chat-endpoint /chat/completions \
  --model deepseek-v4-flash \
  --output results/deepseek.csv \
  --trace-dir traces/deepseek

python analysis.py --input results/deepseek.csv --figures-dir figures/deepseek
```

</details>

<details>
<summary>Backend comparison figures</summary>

```bash
python analysis.py \
  --input results/rule_based.csv results/lmstudio.csv results/deepseek.csv \
  --figures-dir figures/compare
```

</details>

<details>
<summary>Trace inspection helper</summary>

```bash
python inspect_trace.py traces/rule_based/ms_01.jsonl
```

</details>

## Output Artifacts

Each backend writes a task-level CSV result file and per-task JSONL traces:

- `rule_based`: `results/rule_based.csv` / `traces/rule_based/`
- `lmstudio`: `results/lmstudio.csv` / `traces/lmstudio/`
- `deepseek`: `results/deepseek.csv` / `traces/deepseek/`

`results.csv` is an ad-hoc/default output path when explicitly requested. For reproducible backend comparison, use the backend-specific result files under `results/`.

Single-backend figures are written to the selected `--figures-dir`, for example `figures/rule_based/`, and include:

- `latency_by_task_type.png`
- `success_rate_by_task_type.png`
- `failure_type_distribution.png`
- `failure_flags_distribution.png`
- `oracle_metric_breakdown.png`

Comparison mode writes backend-level summaries and comparison figures under `figures/compare/`.

## Canonical Artifacts

- Results:
  - [`results/rule_based.csv`](results/rule_based.csv)
  - [`results/lmstudio.csv`](results/lmstudio.csv)
  - [`results/deepseek.csv`](results/deepseek.csv)
- Run metadata:
  - [`results/metadata/rule_based.json`](results/metadata/rule_based.json)
  - [`results/metadata/lmstudio.json`](results/metadata/lmstudio.json)
  - [`results/metadata/deepseek.json`](results/metadata/deepseek.json)
- Traces:
  - [`traces/rule_based/`](traces/rule_based/)
  - [`traces/lmstudio/`](traces/lmstudio/)
  - [`traces/deepseek/`](traces/deepseek/)
- Figures and summaries:
  - [`figures/rule_based/`](figures/rule_based/)
  - [`figures/lmstudio/`](figures/lmstudio/)
  - [`figures/deepseek/`](figures/deepseek/)
  - [`figures/compare/`](figures/compare/)

## Security Note

- Never commit API keys.
- DeepSeek key is read from `DEEPSEEK_API_KEY`.
- `config.local.json` is ignored.

## Reproducibility Note

Using a local LM Studio backend reduces network-induced latency variance, API-provider queueing, rate-limit effects, and silent provider-side model updates. It does not eliminate runtime variance from local hardware load, model loading, quantization, context length, decoding settings, thermal throttling, or LM Studio server overhead.

DeepSeek results include network latency, provider queueing, rate limits, and provider-side model/runtime effects. Latency values across local and cloud backends should be interpreted as system-level latency, not pure model compute time.

Each run records a task file hash (`tasks_sha256`) in `results/metadata/*.json`. Tasks are executed sequentially, and latency should be interpreted as end-to-end system latency.

## Token and Cost Note

Token/cost fields combine two sources:

- provider usage when available (`prompt_tokens_provider`, `completion_tokens_provider`, `total_tokens_provider`)
- heuristic token counting fallback when provider usage is unavailable

`cost_usd` uses configured per-token mock pricing, and `pricing_source` records which path was used (`provider_usage_with_configured_pricing`, `provider_usage_no_pricing`, or `heuristic_mock`). These values are for relative profiling, not provider-level billing accuracy.

## Limitations

- Small-scale controlled benchmark, not a full-scale capability benchmark
- Rule-based backend is an oracle sanity backend, not a competitive baseline
- Trace logs observable execution events only; hidden model reasoning is not exposed
- Guardrail detection is deterministic and regex-based

Earlier runs without explicit tool schemas or guardrail record context were wrapper-contract diagnostics, not direct model capability measurements.
