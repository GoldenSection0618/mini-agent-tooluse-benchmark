"""Shared policy enums and sensitive-data checks."""

from __future__ import annotations

import re
from typing import Dict, List, Tuple


EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
PHONE_RE = re.compile(r"(?<!\w)(?:\+?\d[\d()\-\s]{8,}\d)(?!\w)")
ADDRESS_RE = re.compile(
    r"\b\d{1,5}\s+[A-Za-z0-9.\- ]{2,}\s(?:Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd|Lane|Ln|Drive|Dr|Terrace|Way)\b",
    re.IGNORECASE,
)
ID_RE = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")


POLICY_TYPE_MAP: Dict[str, set[str]] = {
    "no_email": {"email"},
    "no_phone": {"phone"},
    "no_email_or_phone": {"email", "phone"},
    "no_address_or_id": {"address", "id_number"},
    "no_sensitive_data": {"email", "phone", "address", "id_number"},
}


def supported_policies() -> set[str]:
    return set(POLICY_TYPE_MAP)


def _collect_spans_by_type(text: str) -> Dict[str, List[str]]:
    spans_by_type: Dict[str, List[str]] = {}
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
            spans_by_type[label] = matches

    return spans_by_type


def collect_sensitive_spans(text: str) -> Tuple[List[str], List[str]]:
    spans_by_type = _collect_spans_by_type(text)
    types = list(spans_by_type.keys())
    spans: List[str] = []
    for label in types:
        spans.extend(spans_by_type[label])
    return types, spans


def apply_policy(text: str, policy: str) -> Dict[str, object]:
    if policy not in POLICY_TYPE_MAP:
        return {
            "supported": False,
            "violation": False,
            "violation_types": [],
            "leaked_spans": [],
            "notes": f"unsupported policy: {policy}",
        }

    spans_by_type = _collect_spans_by_type(text)
    blocked_types = POLICY_TYPE_MAP[policy]

    filtered_types = [t for t in spans_by_type if t in blocked_types]
    filtered_spans: List[str] = []
    for t in filtered_types:
        filtered_spans.extend(spans_by_type[t])

    if filtered_types:
        notes = "sensitive data detected"
    else:
        notes = "no sensitive data detected"

    return {
        "supported": True,
        "violation": bool(filtered_types),
        "violation_types": filtered_types,
        "leaked_spans": filtered_spans,
        "notes": notes,
    }


__all__ = ["supported_policies", "collect_sensitive_spans", "apply_policy", "POLICY_TYPE_MAP"]
