"""Helpers for extracting compact JSON objects from model text outputs."""

from __future__ import annotations

import json
from json import JSONDecodeError
from typing import Any, Dict


def _strip_markdown_fence(text: str) -> str:
    stripped = text.strip()
    if not stripped.startswith("```"):
        return stripped

    lines = stripped.splitlines()
    if len(lines) >= 2 and lines[0].startswith("```") and lines[-1].strip() == "```":
        return "\n".join(lines[1:-1]).strip()
    return stripped.strip("`").strip()


def extract_json_object(text: str) -> Dict[str, Any]:
    if not isinstance(text, str) or not text.strip():
        raise ValueError("empty model output")

    candidate = _strip_markdown_fence(text)

    try:
        parsed = json.loads(candidate)
        if isinstance(parsed, dict):
            return parsed
        raise ValueError("parsed JSON is not an object")
    except Exception:
        pass

    start = candidate.find("{")
    if start == -1:
        raise ValueError("no JSON object start found")

    depth = 0
    end = -1
    for idx in range(start, len(candidate)):
        ch = candidate[idx]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                end = idx
                break

    if end == -1:
        raise ValueError("unterminated JSON object")

    snippet = candidate[start : end + 1]
    try:
        parsed = json.loads(snippet)
    except JSONDecodeError as exc:
        raise ValueError(f"invalid JSON object: {exc}") from exc

    if not isinstance(parsed, dict):
        raise ValueError("parsed JSON is not an object")
    return parsed


__all__ = ["extract_json_object"]
