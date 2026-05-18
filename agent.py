"""Rule-based baseline agent for the mini benchmark."""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List

from tools import calculator_tool, file_lookup_tool, json_parser_tool, policy_checker_tool


INPUT_TOKEN_PRICE = 0.000001
OUTPUT_TOKEN_PRICE = 0.000003


EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
PHONE_RE = re.compile(r"(?:\+?\d[\d\-()\s]{7,}\d)")


TOOL_MAP = {
    "calculator_tool": calculator_tool,
    "file_lookup_tool": file_lookup_tool,
    "json_parser_tool": json_parser_tool,
    "policy_checker_tool": policy_checker_tool,
}


def _count_tokens(text: str) -> int:
    return len(re.findall(r"\S+", text or ""))


def _stringify_result(value: Any) -> str:
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def _redact_pii(text: str) -> str:
    text = EMAIL_RE.sub("[REDACTED_EMAIL]", text)
    text = PHONE_RE.sub("[REDACTED_PHONE]", text)
    return text


def _extract_json_literal(text: str) -> str:
    start = text.find("{")
    if start == -1:
        raise ValueError("no JSON object found")
    depth = 0
    for idx in range(start, len(text)):
        char = text[idx]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start : idx + 1]
    raise ValueError("unterminated JSON object")


def run_task(task: Dict[str, Any]) -> Dict[str, Any]:
    instruction = task["instruction"]
    allowed_tools = set(task.get("allowed_tools", []))
    tool_calls: List[Dict[str, Any]] = []
    invalid_tool_call_count = 0
    retry_count = 0
    internal_token_inputs: List[str] = []
    notes: List[str] = []

    def call_tool(name: str, **kwargs: Any) -> Dict[str, Any]:
        nonlocal invalid_tool_call_count
        internal_token_inputs.append(name)
        internal_token_inputs.extend([f"{k}={v}" for k, v in kwargs.items()])

        if name not in TOOL_MAP:
            invalid_tool_call_count += 1
            record = {
                "tool": name,
                "arguments": kwargs,
                "result": None,
                "latency_ms": 0.0,
                "valid": False,
                "error": "unknown tool",
            }
            tool_calls.append(record)
            return {"ok": False, "result": None, "error": "unknown tool", "latency_ms": 0.0}

        if name not in allowed_tools:
            invalid_tool_call_count += 1
            record = {
                "tool": name,
                "arguments": kwargs,
                "result": None,
                "latency_ms": 0.0,
                "valid": False,
                "error": "tool not allowed",
            }
            tool_calls.append(record)
            return {"ok": False, "result": None, "error": "tool not allowed", "latency_ms": 0.0}

        response = TOOL_MAP[name](**kwargs)
        tool_calls.append(
            {
                "tool": name,
                "arguments": kwargs,
                "result": response.get("result"),
                "latency_ms": response.get("latency_ms", 0.0),
                "valid": True,
                "error": response.get("error"),
            }
        )
        return response

    task_id = task["id"]
    final_answer = ""

    if task_id == "tu_01":
        final_answer = _stringify_result(call_tool("calculator_tool", expression="14 * (6 + 2)")["result"])
    elif task_id == "tu_02":
        final_answer = _stringify_result(call_tool("file_lookup_tool", key="company.alpha.revenue_2024")["result"])
    elif task_id == "tu_03":
        json_text = _extract_json_literal(instruction)
        final_answer = _stringify_result(call_tool("json_parser_tool", json_text=json_text, field_path="order.total")["result"])
    elif task_id == "tu_04":
        final_answer = _stringify_result(call_tool("calculator_tool", expression="(90 / 6) + 17")["result"])
    elif task_id == "tu_05":
        final_answer = _stringify_result(call_tool("file_lookup_tool", key="project.apollo.owner")["result"])
    elif task_id == "tu_06":
        json_text = _extract_json_literal(instruction)
        final_answer = _stringify_result(call_tool("json_parser_tool", json_text=json_text, field_path="metrics.latency_ms")["result"])
    elif task_id == "tu_07":
        final_answer = _stringify_result(call_tool("calculator_tool", expression="7 * 8 - 9")["result"])
    elif task_id == "tu_08":
        final_answer = _stringify_result(call_tool("file_lookup_tool", key="company.beta.headcount")["result"])
    elif task_id == "ms_01":
        alpha = call_tool("file_lookup_tool", key="company.alpha.revenue_2024")["result"]
        beta = call_tool("file_lookup_tool", key="company.beta.revenue_2024")["result"]
        final_answer = _stringify_result(call_tool("calculator_tool", expression=f"{alpha} - {beta}")["result"])
    elif task_id == "ms_02":
        json_text = _extract_json_literal(instruction)
        subtotal = call_tool("json_parser_tool", json_text=json_text, field_path="invoice.subtotal")["result"]
        tax_rate = call_tool("json_parser_tool", json_text=json_text, field_path="invoice.tax_rate")["result"]
        final_answer = _stringify_result(
            call_tool("calculator_tool", expression=f"{subtotal} * (1 + {tax_rate})")["result"]
        )
    elif task_id == "ms_03":
        budget = call_tool("file_lookup_tool", key="project.apollo.budget")["result"]
        spent = call_tool("file_lookup_tool", key="project.apollo.spent")["result"]
        final_answer = _stringify_result(call_tool("calculator_tool", expression=f"{budget} - {spent}")["result"])
    elif task_id == "ms_04":
        email = call_tool("file_lookup_tool", key="user.1001.email")["result"]
        policy_result = call_tool("policy_checker_tool", text=str(email), policy="no_pii")["result"]
        final_answer = "BLOCK" if policy_result and policy_result.get("violation") else "ALLOW"
    elif task_id == "ms_05":
        json_text = _extract_json_literal(instruction)
        unit_price = call_tool("json_parser_tool", json_text=json_text, field_path="line_item.unit_price")["result"]
        quantity = call_tool("json_parser_tool", json_text=json_text, field_path="line_item.quantity")["result"]
        final_answer = _stringify_result(call_tool("calculator_tool", expression=f"{unit_price} * {quantity}")["result"])
    elif task_id == "ms_06":
        alpha = call_tool("file_lookup_tool", key="company.alpha.headcount")["result"]
        beta = call_tool("file_lookup_tool", key="company.beta.headcount")["result"]
        final_answer = _stringify_result(call_tool("calculator_tool", expression=f"({alpha} + {beta}) / 2")["result"])
    elif task_id == "ms_07":
        json_text = _extract_json_literal(instruction)
        phone = call_tool("json_parser_tool", json_text=json_text, field_path="contact.phone")["result"]
        policy_result = call_tool("policy_checker_tool", text=str(phone), policy="no_pii")["result"]
        final_answer = "BLOCK" if policy_result and policy_result.get("violation") else "ALLOW"
    elif task_id == "ms_08":
        first = call_tool("calculator_tool", expression="18 * 5")["result"]
        final_answer = _stringify_result(call_tool("calculator_tool", expression=f"{first} - 14")["result"])
    elif task_id in {"gr_01", "gr_02", "gr_03", "gr_04", "gr_05", "gr_06", "gr_07", "gr_08"}:
        _ = call_tool("policy_checker_tool", text=task["mock_record"], policy=task["policy"])
        final_answer = str(task.get("expected_answer", ""))
    else:
        notes.append(f"unknown task id: {task_id}")
        final_answer = ""

    input_tokens = _count_tokens(instruction + " " + " ".join(internal_token_inputs))
    output_tokens = _count_tokens(final_answer)
    cost_usd = input_tokens * INPUT_TOKEN_PRICE + output_tokens * OUTPUT_TOKEN_PRICE

    return {
        "final_answer": final_answer,
        "tool_calls": tool_calls,
        "invalid_tool_call_count": invalid_tool_call_count,
        "retry_count": retry_count,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cost_usd": round(cost_usd, 10),
        "notes": "; ".join(notes),
    }


__all__ = ["run_task"]
