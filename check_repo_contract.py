"""Lightweight repository contract checks for mini-agent-tooluse-benchmark."""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

from benchmark import RESULT_COLUMNS


FORBIDDEN_PATTERNS = [
    "ThreadPoolExecutor",
    "ProcessPoolExecutor",
    "asyncio.gather",
    "multiprocessing",
    "parallel map",
    "batch request",
]


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def check_tasks() -> None:
    tasks_path = Path("tasks.json")
    assert tasks_path.exists(), "tasks.json missing"
    tasks = load_json(tasks_path)
    assert isinstance(tasks, list), "tasks.json must be a list"
    assert len(tasks) == 24, f"expected 24 tasks, got {len(tasks)}"

    counts = Counter(t.get("type") for t in tasks)
    expected = {"tool_use": 8, "multi_step": 8, "guardrail": 8}
    assert counts == expected, f"unexpected task distribution: {counts}"

    ids = [t.get("id") for t in tasks]
    assert len(ids) == len(set(ids)), "duplicate task ids detected"

    for t in tasks:
        t_id = t.get("id", "<unknown>")
        t_type = t.get("type")
        if t_type != "guardrail":
            for field in ["expected_answer", "required_tools", "expected_tool_sequence"]:
                assert field in t, f"task {t_id} missing {field}"
        else:
            for field in ["policy", "expected_violation", "guardrail_required"]:
                assert field in t, f"guardrail task {t_id} missing {field}"



def check_result_columns() -> None:
    candidate_paths = [
        Path("results/rule_based.csv"),
        Path("results/lmstudio.csv"),
        Path("results/deepseek.csv"),
    ]
    for path in candidate_paths:
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            cols = reader.fieldnames or []
        missing = [c for c in RESULT_COLUMNS if c not in cols]
        assert not missing, f"{path} missing columns: {missing}"



def check_forbidden_parallelism() -> None:
    here = Path(__file__).name
    for path in Path(".").rglob("*.py"):
        if any(part.startswith(".") for part in path.parts):
            continue
        if "__pycache__" in path.parts:
            continue
        if path.name == here:
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        for token in FORBIDDEN_PATTERNS:
            assert token not in text, f"forbidden token {token!r} found in {path}"



def check_config_defaults() -> None:
    cfg_path = Path("config.example.json")
    assert cfg_path.exists(), "config.example.json missing"
    cfg = load_json(cfg_path)

    for backend in ["lmstudio", "deepseek"]:
        assert backend in cfg, f"{backend} config missing"
        bcfg = cfg[backend]
        assert bcfg.get("temperature") == 0, f"{backend}.temperature must be 0"
        assert bcfg.get("max_tokens") == 256, f"{backend}.max_tokens must be 256"



def main() -> None:
    check_tasks()
    check_result_columns()
    check_forbidden_parallelism()
    check_config_defaults()
    print("repo contract check passed")


if __name__ == "__main__":
    main()
