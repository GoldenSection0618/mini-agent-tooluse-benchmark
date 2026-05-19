"""Lightweight JSONL tracing utilities for benchmark runs."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict


TRACE_DIR = Path("traces")
ALLOWED_EVENT_TYPES = {
    "task_start",
    "agent_decision",
    "tool_call",
    "tool_result",
    "guardrail_check",
    "evaluation",
    "task_end",
}


def make_trace_path(task_id: str) -> str:
    TRACE_DIR.mkdir(parents=True, exist_ok=True)
    return str(TRACE_DIR / f"{task_id}.jsonl")


def write_trace_event(trace_path: str, event: Dict[str, Any]) -> None:
    with open(trace_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(event, ensure_ascii=True) + "\n")


def new_event(task_id: str, step: int, event_type: str, **kwargs: Any) -> Dict[str, Any]:
    if event_type not in ALLOWED_EVENT_TYPES:
        raise ValueError(f"unsupported event_type: {event_type}")

    event: Dict[str, Any] = {
        "task_id": task_id,
        "step": int(step),
        "event_type": event_type,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    event.update(kwargs)
    return event


__all__ = ["make_trace_path", "write_trace_event", "new_event", "ALLOWED_EVENT_TYPES"]
