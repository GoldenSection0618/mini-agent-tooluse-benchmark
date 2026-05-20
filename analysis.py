"""Analysis script for oracle-level benchmark results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt
import pandas as pd


def _backend_role(agent_backend: str) -> str:
    if str(agent_backend) == "rule_based":
        return "oracle_sanity_check"
    return "llm_backend"


def _safe_parse_flags(raw: Any) -> tuple[list[str], bool]:
    if isinstance(raw, list):
        return [str(x) for x in raw], False
    if not isinstance(raw, str):
        return [], True
    text = raw.strip()
    if not text:
        return [], False
    try:
        parsed = json.loads(text)
    except Exception:
        return [], True
    if not isinstance(parsed, list):
        return [], True
    return [str(x) for x in parsed], False


def _rate_pct(series: pd.Series) -> float:
    return float(series.mean() * 100.0)


def _top_failure_types(df: pd.DataFrame, top_n: int = 3) -> str:
    counts = df["failure_type"].value_counts()
    parts = [f"{k}:{int(v)}" for k, v in counts.head(top_n).items()]
    return "|".join(parts)


def _write_csv(path: Path, frame: pd.DataFrame, sort_by: Iterable[str]) -> None:
    if not frame.empty:
        sort_cols = [c for c in sort_by if c in frame.columns]
        if sort_cols:
            frame = frame.sort_values(sort_cols, kind="mergesort")
    frame.to_csv(path, index=False)


def _write_summary_artifacts(df: pd.DataFrame, fig_dir: Path) -> None:
    rows = []
    for (backend, source_file), chunk in df.groupby(["agent_backend", "source_file"], dropna=False):
        malformed_flags = int(chunk["failure_flags_malformed"].sum())
        rows.append(
            {
                "agent_backend": backend,
                "backend_role": _backend_role(str(backend)),
                "source_file": source_file,
                "n_tasks": int(len(chunk)),
                "success_rate_pct": _rate_pct(chunk["success"]),
                "final_answer_correct_rate_pct": _rate_pct(chunk["final_answer_correct"]),
                "required_tools_called_rate_pct": _rate_pct(chunk["required_tools_called"]),
                "tool_sequence_match_rate_pct": _rate_pct(chunk["tool_sequence_match"]),
                "tool_argument_match_rate_pct": _rate_pct(chunk["tool_argument_match"]),
                "planning_success_rate_pct": _rate_pct(chunk["planning_success"]),
                "format_correct_rate_pct": _rate_pct(chunk["format_correct"]),
                "contains_excludes_match_rate_pct": _rate_pct(chunk["contains_excludes_match"]),
                "output_policy_clean_rate_pct": _rate_pct(chunk["output_policy_clean"]),
                "guardrail_success_rate_pct": _rate_pct(chunk["guardrail_success"]),
                "avg_wall_clock_time_ms": float(chunk["wall_clock_time_ms"].mean()),
                "avg_request_latency_ms": float(chunk["request_latency_ms"].mean()),
                "avg_tool_latency_ms": float(chunk["tool_latency_ms"].mean()),
                "avg_tool_call_count": float(chunk["tool_call_count"].mean()),
                "invalid_tool_call_count": int(chunk["invalid_tool_call_count"].sum()),
                "retry_count": int(chunk["retry_count"].sum()),
                "tool_error_count": int(chunk["tool_error_count"].sum()),
                "guardrail_false_positive_count": int(chunk["false_positive"].sum()),
                "guardrail_false_negative_count": int(chunk["false_negative"].sum()),
                "input_tokens": int(chunk["input_tokens"].sum()),
                "output_tokens": int(chunk["output_tokens"].sum()),
                "cost_usd": float(chunk["cost_usd"].sum()),
                "main_failure_types": _top_failure_types(chunk),
                "malformed_failure_flags_count": malformed_flags,
            }
        )
    summary_overall = pd.DataFrame(rows)
    _write_csv(fig_dir / "summary_overall.csv", summary_overall, ["backend_role", "agent_backend", "source_file"])

    by_task_rows = []
    for (backend, task_type), chunk in df.groupby(["agent_backend", "task_type"], dropna=False):
        by_task_rows.append(
            {
                "agent_backend": backend,
                "backend_role": _backend_role(str(backend)),
                "task_type": task_type,
                "n_tasks": int(len(chunk)),
                "success_rate_pct": _rate_pct(chunk["success"]),
                "final_answer_correct_rate_pct": _rate_pct(chunk["final_answer_correct"]),
                "tool_sequence_match_rate_pct": _rate_pct(chunk["tool_sequence_match"]),
                "tool_argument_match_rate_pct": _rate_pct(chunk["tool_argument_match"]),
                "output_policy_clean_rate_pct": _rate_pct(chunk["output_policy_clean"]),
                "avg_wall_clock_time_ms": float(chunk["wall_clock_time_ms"].mean()),
                "avg_request_latency_ms": float(chunk["request_latency_ms"].mean()),
                "avg_tool_latency_ms": float(chunk["tool_latency_ms"].mean()),
            }
        )
    summary_by_task = pd.DataFrame(by_task_rows)
    _write_csv(fig_dir / "summary_by_task_type.csv", summary_by_task, ["backend_role", "agent_backend", "task_type"])

    failure_summary = (
        df.groupby(["agent_backend", "failure_type"], as_index=False)["task_id"]
        .count()
        .rename(columns={"task_id": "count"})
    )
    _write_csv(fig_dir / "failure_summary.csv", failure_summary, ["agent_backend", "failure_type"])

    flags_rows = []
    for _, row in df.iterrows():
        flags = row.get("failure_flags_list", [])
        for flag in flags:
            flags_rows.append({"agent_backend": row["agent_backend"], "failure_flag": flag})
    failure_flags_summary = pd.DataFrame(flags_rows)
    if failure_flags_summary.empty:
        failure_flags_summary = pd.DataFrame(columns=["agent_backend", "failure_flag", "count"])
    else:
        failure_flags_summary = (
            failure_flags_summary.groupby(["agent_backend", "failure_flag"], as_index=False)
            .size()
            .rename(columns={"size": "count"})
        )
    _write_csv(fig_dir / "failure_flags_summary.csv", failure_flags_summary, ["agent_backend", "failure_flag"])

    if df["agent_backend"].nunique() > 1:
        by_backend = (
            df.groupby("agent_backend", as_index=False)
            .agg(
                n_tasks=("task_id", "count"),
                success_rate_pct=("success", lambda s: _rate_pct(s)),
                final_answer_correct_rate_pct=("final_answer_correct", lambda s: _rate_pct(s)),
                avg_wall_clock_time_ms=("wall_clock_time_ms", "mean"),
                avg_request_latency_ms=("request_latency_ms", "mean"),
                avg_tool_latency_ms=("tool_latency_ms", "mean"),
            )
        )
        by_backend["backend_role"] = by_backend["agent_backend"].map(_backend_role)
        by_backend = by_backend[
            [
                "agent_backend",
                "backend_role",
                "n_tasks",
                "success_rate_pct",
                "final_answer_correct_rate_pct",
                "avg_wall_clock_time_ms",
                "avg_request_latency_ms",
                "avg_tool_latency_ms",
            ]
        ]
        _write_csv(fig_dir / "summary_by_backend.csv", by_backend, ["backend_role", "agent_backend"])
        _write_csv(
            fig_dir / "summary_by_backend_and_task_type.csv",
            summary_by_task,
            ["backend_role", "agent_backend", "task_type"],
        )

        backend_roles = by_backend[["agent_backend", "backend_role"]].drop_duplicates()
        _write_csv(fig_dir / "backend_roles.csv", backend_roles, ["backend_role", "agent_backend"])

        failure_type_by_backend = (
            df.groupby(["agent_backend", "failure_type"], as_index=False)["task_id"]
            .count()
            .rename(columns={"task_id": "count"})
        )
        _write_csv(fig_dir / "failure_type_by_backend.csv", failure_type_by_backend, ["agent_backend", "failure_type"])

        guardrail_df = df[df["task_type"] == "guardrail"]
        guardrail_fp_fn = (
            guardrail_df.groupby("agent_backend", as_index=False)[["false_positive", "false_negative"]]
            .sum()
            .rename(columns={"false_positive": "guardrail_false_positive_count", "false_negative": "guardrail_false_negative_count"})
        )
        _write_csv(fig_dir / "guardrail_fp_fn_by_backend.csv", guardrail_fp_fn, ["agent_backend"])


def _plot_latency_by_task_type(df: pd.DataFrame, fig_dir: Path) -> None:
    latency = df.groupby("task_type", as_index=False)[["wall_clock_time_ms", "tool_latency_ms"]].mean()
    x = range(len(latency))
    width = 0.35

    plt.figure(figsize=(8, 5))
    plt.bar([i - width / 2 for i in x], latency["wall_clock_time_ms"], width=width, label="wall_clock_time_ms")
    plt.bar([i + width / 2 for i in x], latency["tool_latency_ms"], width=width, label="tool_latency_ms")
    plt.xticks(list(x), latency["task_type"])
    plt.ylabel("Latency (ms)")
    plt.title("Average Latency by Task Type")
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_dir / "latency_by_task_type.png", dpi=150)
    plt.close()


def _plot_success_rate(df: pd.DataFrame, fig_dir: Path) -> None:
    summary = df.groupby("task_type", as_index=False)[["success", "final_answer_correct"]].mean() * 100
    summary["task_type"] = df.groupby("task_type", as_index=False)["task_type"].first()["task_type"]
    x = range(len(summary))
    width = 0.35

    plt.figure(figsize=(8, 5))
    plt.bar([i - width / 2 for i in x], summary["success"], width=width, label="success")
    plt.bar([i + width / 2 for i in x], summary["final_answer_correct"], width=width, label="final_answer_correct")
    plt.xticks(list(x), summary["task_type"])
    plt.ylim(0, 100)
    plt.ylabel("Rate (%)")
    plt.title("Success and Final-Answer Correctness by Task Type")
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_dir / "success_rate_by_task_type.png", dpi=150)
    plt.close()


def _plot_failure_distribution(df: pd.DataFrame, fig_dir: Path) -> None:
    failure_counts = df["failure_type"].value_counts().sort_index()

    plt.figure(figsize=(9, 5))
    plt.bar(failure_counts.index, failure_counts.values)
    plt.ylabel("Count")
    plt.title("Failure Type Distribution")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(fig_dir / "failure_type_distribution.png", dpi=150)
    plt.close()


def _plot_failure_flags_distribution(df: pd.DataFrame, fig_dir: Path) -> None:
    flags = []
    for raw in df["failure_flags_list"]:
        parsed = raw if isinstance(raw, list) else []
        flags.extend(parsed)

    counts = pd.Series(flags).value_counts().sort_index() if flags else pd.Series(dtype="int64")
    plt.figure(figsize=(10, 5))
    if not counts.empty:
        plt.bar(counts.index, counts.values)
        plt.xticks(rotation=30, ha="right")
    plt.ylabel("Count")
    plt.title("Failure Flags Distribution")
    plt.tight_layout()
    plt.savefig(fig_dir / "failure_flags_distribution.png", dpi=150)
    plt.close()


def _plot_oracle_metric_breakdown(df: pd.DataFrame, fig_dir: Path) -> None:
    metrics = [
        "success",
        "final_answer_correct",
        "required_tools_called",
        "tool_sequence_match",
        "tool_argument_match",
        "planning_success",
        "format_correct",
        "contains_excludes_match",
        "output_policy_clean",
    ]
    rates = [(df[col].mean() * 100) for col in metrics]

    plt.figure(figsize=(10, 5))
    plt.bar(metrics, rates)
    plt.ylim(0, 100)
    plt.ylabel("Rate (%)")
    plt.title("Oracle Metric Breakdown")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig(fig_dir / "oracle_metric_breakdown.png", dpi=150)
    plt.close()


def _plot_success_rate_by_backend(df: pd.DataFrame, fig_dir: Path) -> None:
    summary = df.groupby("agent_backend", as_index=False)["success"].mean()
    summary["success_rate"] = summary["success"] * 100
    plt.figure(figsize=(7, 5))
    plt.bar(summary["agent_backend"], summary["success_rate"])
    plt.ylim(0, 100)
    plt.ylabel("Success Rate (%)")
    plt.title("Success Rate by Backend")
    plt.tight_layout()
    plt.savefig(fig_dir / "success_rate_by_backend.png", dpi=150)
    plt.close()


def _plot_latency_by_backend(df: pd.DataFrame, fig_dir: Path) -> None:
    summary = df.groupby("agent_backend", as_index=False)[["wall_clock_time_ms", "tool_latency_ms"]].mean()
    x = range(len(summary))
    width = 0.35
    plt.figure(figsize=(8, 5))
    plt.bar([i - width / 2 for i in x], summary["wall_clock_time_ms"], width=width, label="wall_clock_time_ms")
    plt.bar([i + width / 2 for i in x], summary["tool_latency_ms"], width=width, label="tool_latency_ms")
    plt.xticks(list(x), summary["agent_backend"])
    plt.ylabel("Latency (ms)")
    plt.title("Latency by Backend")
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_dir / "latency_by_backend.png", dpi=150)
    plt.close()


def _plot_success_rate_by_backend_and_task_type(df: pd.DataFrame, fig_dir: Path) -> None:
    pivot = (
        df.pivot_table(index="agent_backend", columns="task_type", values="success", aggfunc="mean", fill_value=0.0)
        * 100
    )
    ax = pivot.plot(kind="bar", figsize=(9, 5))
    ax.set_ylim(0, 100)
    ax.set_ylabel("Success Rate (%)")
    ax.set_title("Success Rate by Backend and Task Type")
    ax.legend(title="task_type")
    plt.tight_layout()
    plt.savefig(fig_dir / "success_rate_by_backend_and_task_type.png", dpi=150)
    plt.close()


def _plot_failure_type_by_backend(df: pd.DataFrame, fig_dir: Path) -> None:
    pivot = (
        df.pivot_table(index="agent_backend", columns="failure_type", values="task_id", aggfunc="count", fill_value=0)
    )
    ax = pivot.plot(kind="bar", figsize=(10, 5))
    ax.set_ylabel("Count")
    ax.set_title("Failure Type by Backend")
    ax.legend(title="failure_type", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(fig_dir / "failure_type_by_backend.png", dpi=150)
    plt.close()


def _plot_failure_flags_by_backend(df: pd.DataFrame, fig_dir: Path) -> None:
    rows = []
    for _, r in df.iterrows():
        flags = r.get("failure_flags_list", [])
        for flag in flags:
            rows.append({"agent_backend": r["agent_backend"], "failure_flag": flag})
    if not rows:
        return
    flags_df = pd.DataFrame(rows)
    pivot = (
        flags_df.pivot_table(index="agent_backend", columns="failure_flag", values="agent_backend", aggfunc="count", fill_value=0)
    )
    ax = pivot.plot(kind="bar", figsize=(11, 5))
    ax.set_ylabel("Count")
    ax.set_title("Failure Flags by Backend")
    ax.legend(title="failure_flag", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(fig_dir / "failure_flags_by_backend.png", dpi=150)
    plt.close()


def _plot_tool_match_by_backend(df: pd.DataFrame, fig_dir: Path) -> None:
    summary = (
        df.groupby("agent_backend", as_index=False)[["tool_sequence_match", "tool_argument_match"]].mean() * 100
    )
    summary["agent_backend"] = df.groupby("agent_backend", as_index=False)["agent_backend"].first()["agent_backend"]
    x = range(len(summary))
    width = 0.35
    plt.figure(figsize=(9, 5))
    plt.bar([i - width / 2 for i in x], summary["tool_sequence_match"], width=width, label="tool_sequence_match")
    plt.bar([i + width / 2 for i in x], summary["tool_argument_match"], width=width, label="tool_argument_match")
    plt.xticks(list(x), summary["agent_backend"])
    plt.ylim(0, 100)
    plt.ylabel("Rate (%)")
    plt.title("Tool Match Rates by Backend")
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_dir / "tool_sequence_match_by_backend.png", dpi=150)
    plt.close()

    plt.figure(figsize=(7, 5))
    plt.bar(summary["agent_backend"], summary["tool_argument_match"])
    plt.ylim(0, 100)
    plt.ylabel("Rate (%)")
    plt.title("Tool Argument Match by Backend")
    plt.tight_layout()
    plt.savefig(fig_dir / "tool_argument_match_by_backend.png", dpi=150)
    plt.close()


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Analyze benchmark result files.")
    parser.add_argument("--input", nargs="*", default=None, help="Input results CSV path(s).")
    parser.add_argument("--figures-dir", default="figures", help="Directory to save figures.")
    return parser


def main() -> None:
    parser = _build_arg_parser()
    args = parser.parse_args()
    input_paths = args.input
    if not input_paths:
        if Path("results/rule_based.csv").exists():
            input_paths = ["results/rule_based.csv"]
        elif Path("results.csv").exists():
            input_paths = ["results.csv"]
        else:
            input_paths = ["results/rule_based.csv"]

    fig_dir = Path(args.figures_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)

    frames = []
    for p in input_paths:
        frame = pd.read_csv(p)
        if "agent_backend" not in frame.columns:
            frame["agent_backend"] = Path(p).stem
        if "output_policy_clean" not in frame.columns:
            frame["output_policy_clean"] = frame.get("guardrail_success", 1)
        frame["source_file"] = p
        parsed = frame["failure_flags"].apply(_safe_parse_flags)
        frame["failure_flags_list"] = parsed.apply(lambda x: x[0])
        frame["failure_flags_malformed"] = parsed.apply(lambda x: 1 if x[1] else 0)
        frames.append(frame)
    df = pd.concat(frames, ignore_index=True)
    _write_summary_artifacts(df, fig_dir)

    overall_success_rate = df["success"].mean() * 100
    success_by_type = (df.groupby("task_type")["success"].mean() * 100).to_dict()
    final_answer_by_type = (df.groupby("task_type")["final_answer_correct"].mean() * 100).to_dict()
    tool_sequence_by_type = (df.groupby("task_type")["tool_sequence_match"].mean() * 100).to_dict()
    tool_argument_by_type = (df.groupby("task_type")["tool_argument_match"].mean() * 100).to_dict()
    output_policy_clean_by_type = (df.groupby("task_type")["output_policy_clean"].mean() * 100).to_dict()
    guardrail_false_positive_count = int(df[df["task_type"] == "guardrail"]["false_positive"].sum())
    guardrail_false_negative_count = int(df[df["task_type"] == "guardrail"]["false_negative"].sum())
    avg_wall_clock = df.groupby("task_type")["wall_clock_time_ms"].mean().to_dict()
    avg_tool_latency = df.groupby("task_type")["tool_latency_ms"].mean().to_dict()
    failure_dist = df["failure_type"].value_counts().to_dict()
    avg_agent_steps = df.groupby("task_type")["agent_step_count"].mean().to_dict()
    tool_error_summary = {
        "total_tool_errors": int(df["tool_error_count"].sum()),
        "tasks_with_tool_errors": int((df["tool_error_count"] > 0).sum()),
    }

    failure_flags = []
    for raw in df["failure_flags_list"]:
        failure_flags.extend(raw if isinstance(raw, list) else [])
    failure_flags_dist = pd.Series(failure_flags).value_counts().to_dict() if failure_flags else {}

    _plot_latency_by_task_type(df, fig_dir)
    _plot_success_rate(df, fig_dir)
    _plot_failure_distribution(df, fig_dir)
    _plot_failure_flags_distribution(df, fig_dir)
    _plot_oracle_metric_breakdown(df, fig_dir)
    if len(input_paths) > 1:
        _plot_success_rate_by_backend(df, fig_dir)
        _plot_latency_by_backend(df, fig_dir)
        _plot_success_rate_by_backend_and_task_type(df, fig_dir)
        _plot_failure_type_by_backend(df, fig_dir)
        _plot_failure_flags_by_backend(df, fig_dir)
        _plot_tool_match_by_backend(df, fig_dir)

    print(f"Overall success rate: {overall_success_rate:.2f}%")
    print(f"Success rate by task type: {success_by_type}")
    print(f"Final-answer correctness by task type: {final_answer_by_type}")
    print(f"Tool-sequence match rate by task type: {tool_sequence_by_type}")
    print(f"Tool-argument match rate by task type: {tool_argument_by_type}")
    print(f"Output-policy-clean rate by task type: {output_policy_clean_by_type}")
    print(f"Guardrail false positive count: {guardrail_false_positive_count}")
    print(f"Guardrail false negative count: {guardrail_false_negative_count}")
    print(f"Average wall-clock latency by task type (ms): {avg_wall_clock}")
    print(f"Average tool latency by task type (ms): {avg_tool_latency}")
    print(f"Failure type distribution: {failure_dist}")
    print(f"Failure flags distribution: {failure_flags_dist}")
    print(f"Average agent step count by task type: {avg_agent_steps}")
    print(f"Tool error count summary: {tool_error_summary}")
    if len(input_paths) > 1:
        overall_by_backend = (df.groupby("agent_backend")["success"].mean() * 100).to_dict()
        avg_wall_clock_by_backend = df.groupby("agent_backend")["wall_clock_time_ms"].mean().to_dict()
        avg_request_latency_by_backend = df.groupby("agent_backend")["request_latency_ms"].mean().to_dict()
        avg_tool_latency_by_backend = df.groupby("agent_backend")["tool_latency_ms"].mean().to_dict()
        failure_type_by_backend = (
            df.groupby(["agent_backend", "failure_type"])["task_id"].count().to_dict()
        )
        guardrail_df = df[df["task_type"] == "guardrail"]
        fp_fn_by_backend = (
            guardrail_df.groupby("agent_backend")[["false_positive", "false_negative"]].sum().to_dict("index")
            if not guardrail_df.empty
            else {}
        )

        print(f"Overall success by backend (%): {overall_by_backend}")
        grouped = (
            df.groupby(["agent_backend", "task_type"], as_index=False)["success"]
            .mean()
            .rename(columns={"success": "success_rate"})
        )
        print("Success by backend and task type:")
        print(grouped.to_string(index=False))
        print(f"Average wall_clock_time_ms by backend: {avg_wall_clock_by_backend}")
        print(f"Average request_latency_ms by backend: {avg_request_latency_by_backend}")
        print(f"Average tool_latency_ms by backend: {avg_tool_latency_by_backend}")
        print(f"Failure type counts by backend: {failure_type_by_backend}")
        print(f"Guardrail FP/FN by backend: {fp_fn_by_backend}")


if __name__ == "__main__":
    main()
