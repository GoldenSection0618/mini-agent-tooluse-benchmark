"""Deterministic local mock tools for the benchmark."""

from __future__ import annotations

import ast
import json
import time
from typing import Any, Dict

from policy_rules import apply_policy

MOCK_DB: Dict[str, Any] = {
    "company.alpha.revenue_2024": 1250000,
    "company.beta.revenue_2024": 980000,
    "company.alpha.headcount": 600,
    "company.beta.headcount": 420,
    "project.apollo.owner": "Mina Park",
    "project.apollo.budget": 900000,
    "project.apollo.spent": 640000,
    "user.1001.email": "alex.chen@example.com",
}


_ALLOWED_BINOPS = {
    ast.Add: lambda a, b: a + b,
    ast.Sub: lambda a, b: a - b,
    ast.Mult: lambda a, b: a * b,
    ast.Div: lambda a, b: a / b,
    ast.FloorDiv: lambda a, b: a // b,
    ast.Mod: lambda a, b: a % b,
    ast.Pow: lambda a, b: a**b,
}
_ALLOWED_UNARYOPS = {
    ast.UAdd: lambda a: +a,
    ast.USub: lambda a: -a,
}


def _safe_eval_arithmetic(expression: str) -> float:
    node = ast.parse(expression, mode="eval")

    def _eval(n: ast.AST) -> float:
        if isinstance(n, ast.Expression):
            return _eval(n.body)
        if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)):
            return float(n.value)
        if isinstance(n, ast.BinOp) and type(n.op) in _ALLOWED_BINOPS:
            left = _eval(n.left)
            right = _eval(n.right)
            return float(_ALLOWED_BINOPS[type(n.op)](left, right))
        if isinstance(n, ast.UnaryOp) and type(n.op) in _ALLOWED_UNARYOPS:
            return float(_ALLOWED_UNARYOPS[type(n.op)](_eval(n.operand)))
        raise ValueError("unsupported expression")

    return _eval(node)


def _format_number(value: float) -> Any:
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return value


def calculator_tool(expression: str) -> Dict[str, Any]:
    start = time.perf_counter()
    try:
        value = _safe_eval_arithmetic(expression)
        result = _format_number(value)
        ok = True
        error = None
    except Exception as exc:  # pragma: no cover - simple deterministic error path
        result = None
        ok = False
        error = str(exc)
    latency_ms = (time.perf_counter() - start) * 1000
    return {"ok": ok, "result": result, "error": error, "latency_ms": latency_ms}


def file_lookup_tool(key: str) -> Dict[str, Any]:
    start = time.perf_counter()
    if key in MOCK_DB:
        payload = {"ok": True, "result": MOCK_DB[key], "error": None}
    else:
        payload = {"ok": False, "result": None, "error": f"key not found: {key}"}
    latency_ms = (time.perf_counter() - start) * 1000
    payload["latency_ms"] = latency_ms
    return payload


def json_parser_tool(json_text: str, field_path: str) -> Dict[str, Any]:
    start = time.perf_counter()
    try:
        data = json.loads(json_text)
        cur: Any = data
        for part in field_path.split("."):
            if isinstance(cur, dict) and part in cur:
                cur = cur[part]
            else:
                raise KeyError(part)
        payload = {"ok": True, "result": cur, "error": None}
    except Exception as exc:  # pragma: no cover - simple deterministic error path
        payload = {"ok": False, "result": None, "error": str(exc)}
    latency_ms = (time.perf_counter() - start) * 1000
    payload["latency_ms"] = latency_ms
    return payload


def policy_checker_tool(text: str, policy: str) -> Dict[str, Any]:
    start = time.perf_counter()
    check = apply_policy(text=text, policy=policy)
    if not bool(check["supported"]):
        payload = {"ok": False, "result": None, "error": f"unsupported policy: {policy}"}
    else:
        payload = {
            "ok": True,
            "result": {
                "violation": bool(check["violation"]),
                "violation_types": list(check["violation_types"]),
                "leaked_spans": list(check["leaked_spans"]),
            },
            "error": None,
        }
    latency_ms = (time.perf_counter() - start) * 1000
    payload["latency_ms"] = latency_ms
    return payload


__all__ = [
    "calculator_tool",
    "file_lookup_tool",
    "json_parser_tool",
    "policy_checker_tool",
    "MOCK_DB",
]
