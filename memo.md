# Technical Memo: Oracle-Level Upgrade for Mini Agent Tool-Use Benchmark

## 1. Problem Definition

This project is a small-scale, controlled benchmark for profiling agent tool-use behavior and failure modes. The primary issue addressed in this upgrade is that final-answer matching alone is not a sufficient oracle for agent quality. An agent can return a correct final answer while still violating core process requirements such as tool selection, tool order, argument correctness, or guardrail policy.

The upgrade goal is to keep the benchmark deterministic and local while strengthening evaluation quality through explicit oracle checks.

Scope constraints:

- exactly `24` tasks
- exactly `8` `tool_use`, `8` `multi_step`, `8` `guardrail`
- deterministic local mock tools
- deterministic rule-based baseline agent
- no dependency on external LLM APIs

## 2. Benchmark Setup

Repository components:

- `tasks.json`: task definitions with explicit oracle metadata
- `tools.py`: deterministic mock tools (calculator, lookup, JSON parsing, policy check)
- `guardrails.py`: deterministic regex-based sensitive-data checker
- `agent.py`: baseline rule-based agent
- `evaluator.py`: oracle-level deterministic evaluator
- `benchmark.py`: runner and result logger
- `analysis.py`: summary statistics and figure generation

The run path remains local and reproducible:

1. load and validate task schema
2. execute each task via baseline agent
3. evaluate with oracle checks
4. write `results.csv`
5. aggregate and visualize with `analysis.py`

## 3. Task Oracle Design

Each task now carries explicit oracle fields beyond instruction and expected answer:

- `required_tools`
- `expected_tool_sequence`
- `answer_type`
- `tolerance`
- `expected_answer_contains`
- `expected_answer_excludes`
- `guardrail_required`

Multi-step tasks also include explicit step intent (`expected_steps`).

Guardrail tasks include:

- source policy (`policy`)
- source expectation (`expected_violation`)
- content constraints (`expected_answer_contains` / `expected_answer_excludes`)

Guardrail mix includes:

- email/phone redaction cases
- address/ID-like redaction cases
- direct sensitive-field request refusal cases
- false-positive control cases with non-sensitive operational text

## 4. Metrics and Logging

`results.csv` now logs both operational and oracle metrics.

Core oracle metrics:

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

Operational metrics remain:

- latency (`wall_clock_time_ms`, `tool_latency_ms`)
- call behavior (`tool_call_count`, `invalid_tool_call_count`, `retry_count`)
- usage proxy (`input_tokens`, `output_tokens`, `cost_usd`)

Success is now decomposed and strictly conjunctive:

- final-answer correctness
- tool-use correctness
- planning correctness
- guardrail correctness (when required)

This removes ambiguity from “correct by coincidence” outcomes.

## 5. Experimental Results

Values below come directly from regenerated `results.csv` after oracle upgrade.

Dataset-level results:

- tasks: `24`
- overall success: `100.0%` (`24/24`)
- failure distribution: `none=24`

By task type (all at `100.0%`):

- success rate
- final-answer correctness
- tool-sequence match rate
- tool-argument match rate
- planning success rate

Guardrail-specific:

- guardrail false positives: `0`
- guardrail false negatives: `0`

Latency (average, ms):

- wall-clock: `tool_use=0.046675`, `multi_step=0.0423125`, `guardrail=0.0164875`
- tool latency: `tool_use=0.01505`, `multi_step=0.016925`, `guardrail=0.000575`

Usage proxy:

- total estimated cost: `$0.000873`
- average estimated cost per task: `$0.000036375`
- average tool calls: `tool_use=1.0`, `multi_step=2.625`, `guardrail=1.0`

Figures generated:

- `figures/latency_by_task_type.png`
- `figures/success_rate_by_task_type.png`
- `figures/failure_type_distribution.png`
- `figures/oracle_metric_breakdown.png`

## 6. Failure Case Analysis

In the latest run, no failures are observed (`24/24`). This should be interpreted carefully:

- the baseline agent is deterministic and aligned to the deterministic task schema
- this run confirms internal consistency of task schema, execution, oracle checks, and logging
- it does not imply generalization to unseen instructions or real model behavior

The improved oracle remains valuable even in all-pass runs because it creates explicit diagnostics if regressions appear later. For example, a future model-backed agent can fail sequence/argument/guardrail checks even when final-answer accuracy appears high.

## 7. Limitations and Extensions

Current limitations:

- small task count (`24`) by design
- rule-based baseline agent, not a stochastic LLM agent
- regex-based guardrail checks (not a comprehensive privacy classifier)
- deterministic mock tools with narrow domain coverage
- token and cost are approximate heuristics, not provider billing measurements

Planned extensions:

- add pluggable model-backed agents while keeping the same oracle interface
- evaluate multiple runs for non-deterministic agents and report confidence intervals
- expand adversarial cases for tool misuse and guardrail bypass
- add CI regression thresholds on oracle sub-metrics, not only final success

This benchmark should be treated as a controlled harness for profiling tool-use efficiency and failure cases, not as a full-scale benchmark of real-world agent capability.
