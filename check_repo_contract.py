"""Lightweight repository contract checks for mini-agent-tooluse-benchmark."""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

from agent import _build_llm_task_payload
from benchmark import RESULT_COLUMNS


FORBIDDEN_PATTERNS = [
    "ThreadPoolExecutor",
    "ProcessPoolExecutor",
    "asyncio.gather",
    "multiprocessing",
    "parallel map",
    "batch request",
]

FORBIDDEN_ORACLE_PAYLOAD_FIELDS = {
    "expected_answer",
    "expected_tool_sequence",
    "expected_steps",
    "tolerance",
    "answer_type",
    "expected_answer_contains",
    "expected_answer_excludes",
}

ALLOWED_PAYLOAD_TOP_LEVEL_FIELDS = {
    "task_id",
    "task_type",
    "instruction",
    "allowed_tools",
    "tool_schemas",
    "task_context",
}


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


def check_llm_payload_contract() -> None:
    tasks = load_json(Path("tasks.json"))
    for task in tasks:
        task_id = task.get("id", "<unknown>")
        allowed_tools = set(task.get("allowed_tools", []))
        payload = _build_llm_task_payload(task, allowed_tools)

        payload_keys = set(payload.keys())
        assert payload_keys == ALLOWED_PAYLOAD_TOP_LEVEL_FIELDS, (
            f"task {task_id} payload keys mismatch: {sorted(payload_keys)}"
        )
        assert not (payload_keys & FORBIDDEN_ORACLE_PAYLOAD_FIELDS), (
            f"task {task_id} payload leaks forbidden top-level fields"
        )

        payload_text = json.dumps(payload, ensure_ascii=True, sort_keys=True)
        for forbidden in FORBIDDEN_ORACLE_PAYLOAD_FIELDS:
            assert f'"{forbidden}"' not in payload_text, (
                f"task {task_id} payload leaks forbidden field: {forbidden}"
            )

        schema_keys = set(payload.get("tool_schemas", {}).keys())
        assert schema_keys == allowed_tools, (
            f"task {task_id} tool_schemas mismatch: expected {sorted(allowed_tools)} got {sorted(schema_keys)}"
        )
        for tool_name, schema in payload.get("tool_schemas", {}).items():
            assert isinstance(schema, dict), f"task {task_id} schema for {tool_name} must be dict"
            assert "required_args" in schema, f"task {task_id} schema for {tool_name} missing required_args"
            assert isinstance(schema["required_args"], dict), (
                f"task {task_id} schema for {tool_name} required_args must be dict"
            )

        task_context = payload.get("task_context", {})
        if task.get("type") == "guardrail":
            assert task_context.get("record") == task.get("mock_record"), (
                f"task {task_id} guardrail payload missing/mismatched record"
            )
            assert task_context.get("policy") == task.get("policy"), (
                f"task {task_id} guardrail payload missing/mismatched policy"
            )
        else:
            assert task_context == {}, f"task {task_id} non-guardrail task_context must be empty"


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
            rows = list(reader)
        missing = [c for c in RESULT_COLUMNS if c not in cols]
        assert not missing, f"{path} missing columns: {missing}"
        assert len(rows) == 24, f"{path} must contain 24 data rows, got {len(rows)}"

        ids = [row.get("task_id", "") for row in rows]
        assert len(ids) == len(set(ids)), f"{path} contains duplicate task_id values"

        assert "output_policy_clean" in cols, f"{path} missing output_policy_clean column"
        for row in rows:
            trace_file = row.get("trace_file", "")
            assert trace_file, f"{path} contains empty trace_file"
            assert Path(trace_file).exists(), f"{path} trace_file does not exist: {trace_file}"
            if "guardrail_success" in cols:
                assert str(row.get("guardrail_success", "")) == str(row.get("output_policy_clean", "")), (
                    f"{path} guardrail_success must equal output_policy_clean for task_id={row.get('task_id')}"
                )


def check_forbidden_parallelism() -> None:
    here = Path(__file__).name
    for path in Path(".").rglob("*.py"):
        if any(part.startswith(".") for part in path.parts):
            continue
        if "skills" in path.parts:
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
    check_llm_payload_contract()
    check_result_columns()
    check_forbidden_parallelism()
    check_config_defaults()
    print("repo contract check passed")


if __name__ == "__main__":
    main()
