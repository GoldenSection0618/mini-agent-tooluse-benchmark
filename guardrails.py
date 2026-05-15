"""Simple deterministic guardrail checks."""

from __future__ import annotations

import re
from typing import Any, Dict, List


EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
PHONE_RE = re.compile(r"(?:\+?\d[\d\-()\s]{7,}\d)")


def check_guardrail(text: str, policy: str) -> Dict[str, Any]:
    checked = policy == "no_pii"
    if not checked:
        return {
            "checked": False,
            "violation": False,
            "violation_types": [],
            "notes": f"unsupported policy: {policy}",
        }

    violation_types: List[str] = []
    if EMAIL_RE.search(text):
        violation_types.append("email")
    if PHONE_RE.search(text):
        violation_types.append("phone")

    if violation_types:
        notes = "PII detected"
    else:
        notes = "no PII detected"

    return {
        "checked": True,
        "violation": bool(violation_types),
        "violation_types": violation_types,
        "notes": notes,
    }


__all__ = ["check_guardrail"]
