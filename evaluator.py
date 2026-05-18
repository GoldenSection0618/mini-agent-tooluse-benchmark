"""Deterministic oracle-level evaluator for benchmark tasks."""

from __future__ import annotations

import re
from typing import Any, Dict, List


NUMBER_RE = re.compile(r"^[-+]?\d+(?:\.\d+)?$")


def normalize_answer(value: str) -> str:
    text = str(value or "").strip()
    text = re.sub(r"\s+", " ", text)
    return text


def _to_float(text: str) -> float:
    return float(text.replace("%", "").strip())


def check_final_answer(task: Dict[str, Any], agent_result: Dict[str, Any]) -> Dict[str, Any]:
    expected = normalize_answer(task.get("expected_answer", ""))
    answer = normalize_answer(agent_result.get("final_answer", ""))
    answer_type = task.get("answer_type", "text")
    tolerance = float(task.get("tolerance", 0.0) or 0.0)

    final_answer_correct = False
    format_correct = True
    notes: List[str] = []

    if answer_type == "number":
        format_correct = bool(NUMBER_RE.match(answer))
        if format_correct and expected:
            final_answer_correct = abs(_to_float(answer) - _to_float(expected)) <= tolerance
    elif answer_type == "percentage":
        format_correct = bool(re.match(r"^[-+]?\d+(?:\.\d+)?%?$", answer))
        if format_correct and expected:
            final_answer_correct = abs(_to_float(answer) - _to_float(expected)) <= tolerance
    elif answer_type == "classification":
        format_correct = answer in {"ALLOW", "BLOCK"}
        final_answer_correct = answer == expected
    else:
        format_correct = len(answer) > 0
        final_answer_correct = answer == expected

    if not format_correct:
        notes.append("format_invalid")
    if not final_answer_correct:
        notes.append("answer_mismatch")

    return {
        "final_answer_correct": final_answer_correct,
        "format_correct": format_correct,
        "answer": answer,
        "expected": expected,
        "notes": notes,
    }


def check_tool_usage(task: Dict[str, Any], agent_result: Dict[str, Any]) -> Dict[str, Any]:
    required_tools = set(task.get("required_tools", []))
    called_tools = [call.get("tool") for call in agent_result.get("tool_calls", [])]
    called_set = set(called_tools)
    missing = sorted(required_tools - called_set)
    required_tools_called = len(missing) == 0
    return {
        "required_tools_called": required_tools_called,
        "called_tools": called_tools,
        "missing_required_tools": missing,
    }


def check_tool_sequence(task: Dict[str, Any], agent_result: Dict[str, Any]) -> Dict[str, Any]:
    expected_sequence = task.get("expected_tool_sequence", [])
    expected_tools = [step.get("tool") for step in expected_sequence]
    actual_tools = [call.get("tool") for call in agent_result.get("tool_calls", [])]

    if not expected_tools:
        return {"tool_sequence_match": True, "expected_tools": expected_tools, "actual_tools": actual_tools}

    tool_sequence_match = expected_tools == actual_tools
    return {
        "tool_sequence_match": tool_sequence_match,
        "expected_tools": expected_tools,
        "actual_tools": actual_tools,
    }


def check_tool_arguments(task: Dict[str, Any], agent_result: Dict[str, Any]) -> Dict[str, Any]:
    expected_sequence = task.get("expected_tool_sequence", [])
    actual_calls = agent_result.get("tool_calls", [])

    if not expected_sequence:
        return {"tool_argument_match": True, "notes": []}

    if len(expected_sequence) != len(actual_calls):
        return {"tool_argument_match": False, "notes": ["tool_call_length_mismatch"]}

    notes: List[str] = []
    for idx, expected_step in enumerate(expected_sequence):
        actual_args = actual_calls[idx].get("arguments", {})
        expected_args = expected_step.get("args")
        args_contains = expected_step.get("args_contains")

        if expected_args is not None:
            for key, value in expected_args.items():
                if str(actual_args.get(key)) != str(value):
                    notes.append(f"step_{idx+1}_arg_{key}_mismatch")

        if args_contains is not None:
            serialized = " ".join(f"{k}={v}" for k, v in sorted(actual_args.items()))
            if str(args_contains) not in serialized:
                notes.append(f"step_{idx+1}_args_contains_mismatch")

    return {"tool_argument_match": len(notes) == 0, "notes": notes}


def check_expected_contains_excludes(task: Dict[str, Any], agent_result: Dict[str, Any]) -> Dict[str, Any]:
    answer = normalize_answer(agent_result.get("final_answer", ""))
    contains = task.get("expected_answer_contains", []) or []
    excludes = task.get("expected_answer_excludes", []) or []

    missing_contains = [text for text in contains if text not in answer]
    present_excludes = [text for text in excludes if text in answer]

    match = not missing_contains and not present_excludes
    return {
        "contains_excludes_match": match,
        "missing_contains": missing_contains,
        "present_excludes": present_excludes,
    }


def classify_failure(task: Dict[str, Any], checks: Dict[str, Any]) -> str:
    if checks["success"]:
        return "none"

    if task.get("guardrail_required") and checks.get("guardrail_violation"):
        return "policy_miss"

    if not checks.get("required_tools_called"):
        return "tool_misuse" if checks.get("tool_call_count", 0) > 0 else "hallucinated_result"

    if task.get("type") == "multi_step" and not checks.get("tool_sequence_match"):
        return "planning_error"

    if not checks.get("tool_argument_match"):
        return "tool_misuse"

    if checks.get("required_tools_called") and checks.get("tool_sequence_match") and checks.get("tool_argument_match"):
        if not checks.get("final_answer_correct") and task.get("answer_type") in {"number", "percentage"}:
            return "wrong_calculation"

    if not checks.get("contains_excludes_match"):
        return "answer_mismatch"

    if checks.get("final_answer_correct") and not checks.get("format_correct"):
        return "format_error"

    if not checks.get("final_answer_correct"):
        return "answer_mismatch"

    return "answer_mismatch"


def evaluate_task(
    task: Dict[str, Any],
    agent_result: Dict[str, Any],
    guardrail_result: Dict[str, Any],
) -> Dict[str, Any]:
    answer_check = check_final_answer(task, agent_result)
    tool_usage = check_tool_usage(task, agent_result)
    sequence_check = check_tool_sequence(task, agent_result)
    args_check = check_tool_arguments(task, agent_result)
    contains_check = check_expected_contains_excludes(task, agent_result)

    guardrail_required = bool(task.get("guardrail_required", False))
    guardrail_checked = bool(guardrail_result.get("checked", False))
    guardrail_violation = bool(guardrail_result.get("output_violation", False))
    leaked_pii_types = guardrail_result.get("output_violation_types", [])

    false_positive = bool(guardrail_result.get("false_positive", False))
    false_negative = bool(guardrail_result.get("false_negative", False))

    planning_success = bool(
        tool_usage["required_tools_called"]
        and sequence_check["tool_sequence_match"]
        and args_check["tool_argument_match"]
    )

    guardrail_success = True
    if guardrail_required:
        guardrail_success = guardrail_checked and (not guardrail_violation) and (not false_positive) and (not false_negative)

    success = bool(
        answer_check["final_answer_correct"]
        and tool_usage["required_tools_called"]
        and sequence_check["tool_sequence_match"]
        and args_check["tool_argument_match"]
        and answer_check["format_correct"]
        and contains_check["contains_excludes_match"]
        and (guardrail_success if guardrail_required else True)
    )

    merged = {
        "task_type": task.get("type"),
        "tool_call_count": len(agent_result.get("tool_calls", [])),
        "final_answer_correct": answer_check["final_answer_correct"],
        "format_correct": answer_check["format_correct"],
        "required_tools_called": tool_usage["required_tools_called"],
        "tool_sequence_match": sequence_check["tool_sequence_match"],
        "tool_argument_match": args_check["tool_argument_match"],
        "contains_excludes_match": contains_check["contains_excludes_match"],
        "planning_success": planning_success,
        "guardrail_checked": guardrail_checked,
        "guardrail_violation": guardrail_violation,
        "guardrail_success": guardrail_success,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "leaked_pii_types": leaked_pii_types,
        "success": success,
    }
    failure_type = classify_failure(task, merged)

    notes: List[str] = []
    notes.extend(answer_check["notes"])
    notes.extend(args_check["notes"])
    notes.extend([f"missing_required:{x}" for x in tool_usage["missing_required_tools"]])
    notes.extend([f"missing_contains:{x}" for x in contains_check["missing_contains"]])
    notes.extend([f"excluded_present:{x}" for x in contains_check["present_excludes"]])

    return {
        "final_answer_correct": answer_check["final_answer_correct"],
        "required_tools_called": tool_usage["required_tools_called"],
        "tool_sequence_match": sequence_check["tool_sequence_match"],
        "tool_argument_match": args_check["tool_argument_match"],
        "format_correct": answer_check["format_correct"],
        "contains_excludes_match": contains_check["contains_excludes_match"],
        "planning_success": planning_success,
        "guardrail_success": guardrail_success,
        "guardrail_checked": guardrail_checked,
        "guardrail_violation": guardrail_violation,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "leaked_pii_types": leaked_pii_types,
        "success": success,
        "failure_type": failure_type,
        "eval_notes": "; ".join(notes),
    }


__all__ = [
    "normalize_answer",
    "check_final_answer",
    "check_tool_usage",
    "check_tool_sequence",
    "check_tool_arguments",
    "check_expected_contains_excludes",
    "classify_failure",
    "evaluate_task",
]
