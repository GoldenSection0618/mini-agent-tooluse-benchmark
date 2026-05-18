"""Simple deterministic guardrail checks."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Tuple


EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
PHONE_RE = re.compile(r"(?<!\w)(?:\+?\d[\d()\-\s]{8,}\d)(?!\w)")
ADDRESS_RE = re.compile(
    r"\b\d{1,5}\s+[A-Za-z0-9.\- ]{2,}\s(?:Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd|Lane|Ln|Drive|Dr|Terrace|Way)\b",
    re.IGNORECASE,
)
ID_RE = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")


SUPPORTED_POLICIES = {"no_pii", "no_sensitive_data"}


def _collect_spans(text: str) -> Tuple[List[str], List[str]]:
    violation_types: List[str] = []
    leaked_spans: List[str] = []
    occupied_ranges: List[Tuple[int, int]] = []

    for pattern, label in [(EMAIL_RE, "email"), (ADDRESS_RE, "address"), (ID_RE, "id_number"), (PHONE_RE, "phone")]:
        matches: List[str] = []
        for match in pattern.finditer(text):
            start, end = match.span()
            if any(start < existing_end and end > existing_start for existing_start, existing_end in occupied_ranges):
                continue
            occupied_ranges.append((start, end))
            matches.append(match.group(0).strip())
        if matches:
            violation_types.append(label)
            leaked_spans.extend(matches)

    return violation_types, leaked_spans


def check_guardrail(text: str, policy: str) -> Dict[str, Any]:
    checked = policy in SUPPORTED_POLICIES
    if not checked:
        return {
            "checked": False,
            "violation": False,
            "violation_types": [],
            "leaked_spans": [],
            "notes": f"unsupported policy: {policy}",
        }

    violation_types, leaked_spans = _collect_spans(text)

    if violation_types:
        notes = "sensitive data detected"
    else:
        notes = "no sensitive data detected"

    return {
        "checked": True,
        "violation": bool(violation_types),
        "violation_types": violation_types,
        "leaked_spans": leaked_spans,
        "notes": notes,
    }


__all__ = ["check_guardrail"]
