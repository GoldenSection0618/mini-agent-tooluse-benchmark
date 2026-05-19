"""Inspect a benchmark JSONL trace file."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List


def _resolve_trace_path(raw_path: str) -> Path:
    path = Path(raw_path)
    if path.exists():
        return path

    name = path.stem
    if name.startswith("plan_"):
        suffix = name.split("_", 1)[1]
        candidates = [path.with_name(f"ms_{suffix}.jsonl")]
        if suffix.isdigit():
            candidates.append(path.with_name(f"ms_{int(suffix):02d}.jsonl"))
        for candidate in candidates:
            if candidate.exists():
                return candidate

    raise FileNotFoundError(f"trace file not found: {raw_path}")


def _load_events(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def main() -> None:
    if len(sys.argv) != 2:
        print("Usage: python inspect_trace.py <trace_file.jsonl>")
        sys.exit(1)

    path = _resolve_trace_path(sys.argv[1])
    events = _load_events(path)
    if not events:
        print(f"Trace is empty: {path}")
        sys.exit(1)

    task_id = events[0].get("task_id", "unknown")
    event_types = [e.get("event_type", "unknown") for e in events]

    tools_called: List[str] = []
    total_tool_latency = 0.0
    failure_type = "none"
    notes: List[str] = []

    for event in events:
        if event.get("event_type") == "tool_call":
            tool = event.get("tool")
            if tool is not None:
                tools_called.append(str(tool))
        if event.get("event_type") == "tool_result":
            total_tool_latency += float(event.get("latency_ms", 0.0))
        if event.get("event_type") in {"evaluation", "task_end"} and event.get("failure_type"):
            failure_type = str(event["failure_type"])
        if event.get("notes"):
            notes.append(str(event["notes"]))

    print(f"trace_file: {path}")
    print(f"task_id: {task_id}")
    print(f"event_count: {len(events)}")
    print(f"event_types: {' -> '.join(event_types)}")
    print(f"tools_called: {tools_called}")
    print(f"total_tool_latency_ms: {total_tool_latency:.4f}")
    print(f"failure_type: {failure_type}")
    if notes:
        print("notes:")
        for note in notes:
            print(f"- {note}")


if __name__ == "__main__":
    main()
