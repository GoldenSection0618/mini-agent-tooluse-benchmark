"""Benchmark runner for the mini agent tool-use benchmark."""

from __future__ import annotations

import csv
import json
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

from agent import run_task
from evaluator import evaluate_task
from guardrails import check_guardrail
from tracing import make_trace_path, new_event, write_trace_event


RESULT_COLUMNS = [
    "task_id",
    "task_type",
    "task_subtype",
    "success",
    "final_answer_correct",
    "required_tools_called",
    "tool_sequence_match",
    "tool_argument_match",
    "tool_execution_success",
    "planning_success",
    "format_correct",
    "contains_excludes_match",
    "wall_clock_time_ms",
    "tool_latency_ms",
    "tool_call_count",
    "invalid_tool_call_count",
    "retry_count",
    "input_tokens",
    "output_tokens",
    "cost_usd",
    "guardrail_checked",
    "guardrail_violation",
    "guardrail_success",
    "false_positive",
    "false_negative",
    "leaked_pii_types",
    "failure_type",
    "failure_flags",
    "trace_file",
    "agent_step_count",
    "tool_error_count",
    "notes",
    "eval_notes",
]

FAILURE_TYPES = {
    "none",
    "planning_error",
    "tool_misuse",
    "wrong_calculation",
    "answer_mismatch",
    "hallucinated_result",
    "policy_miss",
    "format_error",
}


def validate_tasks_schema(tasks: List[Dict[str, Any]]) -> None:
    if len(tasks) != 24:
        raise ValueError(f"expected 24 tasks, got {len(tasks)}")

    counts = Counter(task.get("type") for task in tasks)
    expected_counts = {"tool_use": 8, "multi_step": 8, "guardrail": 8}
    if counts != expected_counts:
        raise ValueError(f"invalid task type distribution: {counts}")

    ids = [task.get("id") for task in tasks]
    if len(set(ids)) != len(ids):
        raise ValueError("task ids must be unique")

    required_common = ["id", "type", "instruction", "allowed_tools"]
    for task in tasks:
        missing = [field for field in required_common if field not in task]
        if missing:
            raise ValueError(f"task {task.get('id', '<unknown>')} missing fields: {missing}")

        if task["type"] != "guardrail" and "expected_answer" not in task:
            raise ValueError(f"task {task['id']} must define expected_answer")

        if task["type"] in {"tool_use", "multi_step"} and not task.get("required_tools"):
            raise ValueError(f"task {task['id']} must define required_tools")

        if task["type"] in {"tool_use", "multi_step"} and "expected_tool_sequence" not in task:
            raise ValueError(f"task {task['id']} missing expected_tool_sequence")

        if task["type"] == "guardrail":
            for field in ["policy", "expected_violation", "guardrail_required"]:
                if field not in task:
                    raise ValueError(f"guardrail task {task['id']} missing field: {field}")


def _run_guardrail_checks(task: Dict[str, Any], answer: str) -> Dict[str, Any]:
    if not task.get("guardrail_required", False):
        return {
            "checked": False,
            "source_violation": False,
            "source_violation_types": [],
            "output_violation": False,
            "output_violation_types": [],
            "false_positive": False,
            "false_negative": False,
            "notes": "guardrail_not_required",
        }

    policy = task.get("policy", "no_pii")
    source_text = task.get("mock_record", "")
    source_check = check_guardrail(source_text, policy)
    output_check = check_guardrail(answer, policy)
    expected_violation = bool(task.get("expected_violation", False))

    source_violation = bool(source_check.get("violation", False))
    false_positive = (not expected_violation) and source_violation
    false_negative = expected_violation and (not source_violation)

    return {
        "checked": bool(source_check.get("checked", False)) and bool(output_check.get("checked", False)),
        "source_violation": source_violation,
        "source_violation_types": source_check.get("violation_types", []),
        "output_violation": bool(output_check.get("violation", False)),
        "output_violation_types": output_check.get("violation_types", []),
        "false_positive": false_positive,
        "false_negative": false_negative,
        "notes": f"source={source_check.get('notes', '')}; output={output_check.get('notes', '')}",
    }


def run_benchmark(tasks_path: Path = Path("tasks.json"), output_path: Path = Path("results.csv")) -> List[Dict[str, Any]]:
    with tasks_path.open("r", encoding="utf-8") as f:
        tasks: List[Dict[str, Any]] = json.load(f)
    validate_tasks_schema(tasks)

    rows: List[Dict[str, Any]] = []

    for task in tasks:
        task_id = task["id"]
        trace_path = make_trace_path(task_id)
        Path(trace_path).write_text("", encoding="utf-8")
        trace_step = 0
        trace_start = time.perf_counter()

        def append_trace(event_type: str, **kwargs: Any) -> None:
            nonlocal trace_step
            trace_step += 1
            kwargs["step_time_ms"] = (time.perf_counter() - trace_start) * 1000
            write_trace_event(trace_path, new_event(task_id=task_id, step=trace_step, event_type=event_type, **kwargs))

        append_trace(
            "task_start",
            task_type=task["type"],
            task_subtype=task.get("subtype", ""),
            instruction=task["instruction"],
        )

        start = time.perf_counter()
        agent_result = run_task(task)
        wall_clock_time_ms = (time.perf_counter() - start) * 1000
        for event in agent_result.get("trace_events", []):
            payload = dict(event)
            payload.pop("task_id", None)
            payload.pop("step", None)
            payload.pop("timestamp", None)
            payload.pop("event_type", None)
            append_trace(event.get("event_type", "agent_decision"), **payload)

        guardrail_result = _run_guardrail_checks(task, str(agent_result.get("final_answer", "")))
        if task.get("guardrail_required", False):
            append_trace(
                "guardrail_check",
                policy=task.get("policy", ""),
                checked=guardrail_result.get("checked", False),
                source_violation=guardrail_result.get("source_violation", False),
                violation=guardrail_result.get("output_violation", False),
                violation_types=guardrail_result.get("output_violation_types", []),
                notes=guardrail_result.get("notes", ""),
            )
        eval_result = evaluate_task(task, agent_result, guardrail_result)
        append_trace(
            "evaluation",
            checks={
                "final_answer_correct": eval_result["final_answer_correct"],
                "required_tools_called": eval_result["required_tools_called"],
                "tool_sequence_match": eval_result["tool_sequence_match"],
                "tool_argument_match": eval_result["tool_argument_match"],
                "tool_execution_success": eval_result["tool_execution_success"],
                "planning_success": eval_result["planning_success"],
                "format_correct": eval_result["format_correct"],
                "contains_excludes_match": eval_result["contains_excludes_match"],
                "guardrail_success": eval_result["guardrail_success"],
            },
            failure_type=eval_result["failure_type"],
            notes=eval_result.get("eval_notes", ""),
        )
        failure_type = eval_result["failure_type"]
        if failure_type not in FAILURE_TYPES:
            raise ValueError(f"invalid failure type: {failure_type}")

        tool_latency_ms = sum(float(call.get("latency_ms", 0.0)) for call in agent_result["tool_calls"])
        tool_error_count = int(eval_result["tool_error_count"])
        guardrail_step_count = 1 if task.get("guardrail_required", False) else 0
        agent_step_count = len(agent_result.get("trace_events", [])) + guardrail_step_count + 1
        notes = "; ".join(
            filter(
                None,
                [
                    agent_result.get("notes", ""),
                    guardrail_result.get("notes", ""),
                ],
            )
        )
        row = {
            "task_id": task["id"],
            "task_type": task["type"],
            "task_subtype": task.get("subtype", ""),
            "success": int(eval_result["success"]),
            "final_answer_correct": int(eval_result["final_answer_correct"]),
            "required_tools_called": int(eval_result["required_tools_called"]),
            "tool_sequence_match": int(eval_result["tool_sequence_match"]),
            "tool_argument_match": int(eval_result["tool_argument_match"]),
            "tool_execution_success": int(eval_result["tool_execution_success"]),
            "planning_success": int(eval_result["planning_success"]),
            "format_correct": int(eval_result["format_correct"]),
            "contains_excludes_match": int(eval_result["contains_excludes_match"]),
            "wall_clock_time_ms": round(wall_clock_time_ms, 4),
            "tool_latency_ms": round(tool_latency_ms, 4),
            "tool_call_count": len(agent_result["tool_calls"]),
            "invalid_tool_call_count": int(agent_result["invalid_tool_call_count"]),
            "retry_count": int(agent_result["retry_count"]),
            "input_tokens": int(agent_result["input_tokens"]),
            "output_tokens": int(agent_result["output_tokens"]),
            "cost_usd": float(agent_result["cost_usd"]),
            "guardrail_checked": int(eval_result["guardrail_checked"]),
            "guardrail_violation": int(eval_result["guardrail_violation"]),
            "guardrail_success": int(eval_result["guardrail_success"]),
            "false_positive": int(eval_result["false_positive"]),
            "false_negative": int(eval_result["false_negative"]),
            "leaked_pii_types": "|".join(eval_result["leaked_pii_types"]),
            "failure_type": failure_type,
            "failure_flags": json.dumps(eval_result.get("failure_flags", []), ensure_ascii=True),
            "trace_file": trace_path,
            "agent_step_count": agent_step_count,
            "tool_error_count": tool_error_count,
            "notes": notes,
            "eval_notes": eval_result.get("eval_notes", ""),
        }
        append_trace(
            "task_end",
            success=bool(eval_result["success"]),
            failure_type=failure_type,
            notes=notes,
        )
        rows.append(row)

    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=RESULT_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    return rows


def main() -> None:
    rows = run_benchmark()
    print(f"Wrote results.csv with {len(rows)} rows")


if __name__ == "__main__":
    main()
