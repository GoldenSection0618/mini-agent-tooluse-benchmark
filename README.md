# Mini Agent Tool-Use Benchmark

This repository implements a small-scale benchmark for evaluating LLM agent tool-use efficiency and failure cases. It focuses on task success rate, wall-clock latency, tool-call latency, token/cost usage, invalid tool calls, retries, guardrail violations, and trace-based diagnostics.

## Motivation

Final-answer matching alone can hide process failures. An answer may look correct while the agent skips required tools, calls tools in the wrong order, uses wrong arguments, or leaks sensitive fields. This benchmark keeps setup local and deterministic, then adds oracle checks and lightweight traces to measure both outcome and process quality.

## Backends

Three backends are supported:

- `rule_based`: deterministic sanity backend for validating evaluator/tracing/failure taxonomy
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
- Deterministic rule-based baseline (`RuleBasedAgent`)
- Local LM Studio backend (`LocalLLMAgent` + `llm_clients.py`)
- Deterministic guardrail checker (`guardrails.py`)
- Oracle evaluator with explicit sub-checks (`evaluator.py`)
- Per-task JSONL traces (`tracing.py` + `traces*/`)
- Analysis and figures (`analysis.py`)

## Task Types

- `tool_use` (`8`)
- `multi_step` (`8`)
- `guardrail` (`8`)

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

## Tracing

Each task run produces a JSONL trace at `traces_{backend}/{task_id}.jsonl` by default (or custom `--trace-dir`).

Trace events include:

- `task_start`
- `agent_decision`
- `tool_call`
- `tool_result`
- `guardrail_check`
- `evaluation`
- `task_end`

This is a lightweight execution trace, not a full reasoning trace.

In `guardrail_check` events:

- `source_contains_sensitive_data`: source/input contains sensitive data
- `output_contains_forbidden_data`: agent output contains policy-forbidden data

## Failure Taxonomy

- `failure_type`: one primary failure category for aggregation
- `failure_flags`: all detected failure conditions for compound analysis

## Setup

```bash
conda create -n agent python=3.11 -y
mamba install -n agent -y pandas matplotlib
```

## Run Commands

Rule-based baseline:

```bash
python benchmark.py --agent rule_based
python analysis.py --input results_rule_based.csv --figures-dir figures_rule_based
```

LM Studio run:

```bash
python benchmark.py \
  --agent lmstudio \
  --base-url http://localhost:1234 \
  --chat-endpoint /api/v1/chat \
  --model google/gemma-4-e4b \
  --output results_lmstudio.csv \
  --trace-dir traces_lmstudio

python analysis.py --input results_lmstudio.csv --figures-dir figures_lmstudio
```

DeepSeek run:

```bash
export DEEPSEEK_API_KEY="your_api_key_here"

python benchmark.py \
  --agent deepseek \
  --base-url https://api.deepseek.com \
  --chat-endpoint /chat/completions \
  --model deepseek-v4-flash \
  --output results_deepseek.csv \
  --trace-dir traces_deepseek

python analysis.py --input results_deepseek.csv --figures-dir figures_deepseek
```

Comparison:

```bash
python analysis.py \
  --input results_rule_based.csv results_lmstudio.csv results_deepseek.csv \
  --figures-dir figures_compare
```

Trace inspection helper:

```bash
python inspect_trace.py traces_rule_based/ms_01.jsonl
```

## Output Artifacts

Typical outputs by backend:

- `results_rule_based.csv` / `traces_rule_based/`
- `results_lmstudio.csv` / `traces_lmstudio/`
- `results_deepseek.csv` / `traces_deepseek/`

Core figures:

- `latency_by_task_type.png`
- `success_rate_by_task_type.png`
- `failure_type_distribution.png`
- `failure_flags_distribution.png`

Comparison mode may also generate:

- `success_rate_by_backend.png`
- `latency_by_backend.png`
- `success_rate_by_backend_and_task_type.png`
- `failure_type_by_backend.png`
- `failure_flags_by_backend.png`
- `tool_sequence_match_by_backend.png`
- `tool_argument_match_by_backend.png`

## Security Note

- Never commit API keys.
- DeepSeek key is read from `DEEPSEEK_API_KEY`.
- `config.local.json` is ignored.

## Reproducibility Note

Using a local LM Studio backend reduces network-induced latency variance, API-provider queueing, rate-limit effects, and silent provider-side model updates. It does not eliminate runtime variance from local hardware load, model loading, quantization, context length, decoding settings, thermal throttling, or LM Studio server overhead.

DeepSeek results include network latency, provider queueing, rate limits, and provider-side model/runtime effects. Latency values across local and cloud backends should be interpreted as system-level latency, not pure model compute time.

## Token and Cost Note

Token and cost values are approximate estimates based on a simple heuristic and fixed mock pricing. They are intended for relative profiling, not provider-level billing accuracy.

## Limitations

- Small-scale controlled benchmark, not a full-scale capability benchmark
- Rule-based baseline does not represent open-ended model behavior
- Trace logs observable execution events only; hidden model reasoning is not exposed
- Guardrail detection is deterministic and regex-based
