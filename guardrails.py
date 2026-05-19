"""Simple deterministic guardrail checks."""

from __future__ import annotations

from typing import Any, Dict

from policy_rules import apply_policy, supported_policies


def check_guardrail(text: str, policy: str) -> Dict[str, Any]:
    checked = policy in supported_policies()
    if not checked:
        return {
            "checked": False,
            "violation": False,
            "violation_types": [],
            "leaked_spans": [],
            "notes": f"unsupported policy: {policy}",
        }

    check = apply_policy(text=text, policy=policy)

    return {
        "checked": True,
        "violation": bool(check["violation"]),
        "violation_types": list(check["violation_types"]),
        "leaked_spans": list(check["leaked_spans"]),
        "notes": str(check["notes"]),
    }


__all__ = ["check_guardrail"]
