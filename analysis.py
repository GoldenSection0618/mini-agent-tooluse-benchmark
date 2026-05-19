"""Analysis script for oracle-level benchmark results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


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
    for raw in df["failure_flags"]:
        parsed = json.loads(raw)
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
        "guardrail_success",
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
        if Path("results.csv").exists():
            input_paths = ["results.csv"]
        else:
            input_paths = ["results_rule_based.csv"]

    fig_dir = Path(args.figures_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)

    frames = []
    for p in input_paths:
        frame = pd.read_csv(p)
        if "agent_backend" not in frame.columns:
            frame["agent_backend"] = Path(p).stem
        frame["source_file"] = p
        frames.append(frame)
    df = pd.concat(frames, ignore_index=True)

    overall_success_rate = df["success"].mean() * 100
    success_by_type = (df.groupby("task_type")["success"].mean() * 100).to_dict()
    final_answer_by_type = (df.groupby("task_type")["final_answer_correct"].mean() * 100).to_dict()
    tool_sequence_by_type = (df.groupby("task_type")["tool_sequence_match"].mean() * 100).to_dict()
    tool_argument_by_type = (df.groupby("task_type")["tool_argument_match"].mean() * 100).to_dict()
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
    for raw in df["failure_flags"]:
        failure_flags.extend(json.loads(raw))
    failure_flags_dist = pd.Series(failure_flags).value_counts().to_dict() if failure_flags else {}

    _plot_latency_by_task_type(df, fig_dir)
    _plot_success_rate(df, fig_dir)
    _plot_failure_distribution(df, fig_dir)
    _plot_failure_flags_distribution(df, fig_dir)
    _plot_oracle_metric_breakdown(df, fig_dir)
    if len(input_paths) > 1:
        _plot_success_rate_by_backend(df, fig_dir)
        _plot_latency_by_backend(df, fig_dir)

    print(f"Overall success rate: {overall_success_rate:.2f}%")
    print(f"Success rate by task type: {success_by_type}")
    print(f"Final-answer correctness by task type: {final_answer_by_type}")
    print(f"Tool-sequence match rate by task type: {tool_sequence_by_type}")
    print(f"Tool-argument match rate by task type: {tool_argument_by_type}")
    print(f"Guardrail false positive count: {guardrail_false_positive_count}")
    print(f"Guardrail false negative count: {guardrail_false_negative_count}")
    print(f"Average wall-clock latency by task type (ms): {avg_wall_clock}")
    print(f"Average tool latency by task type (ms): {avg_tool_latency}")
    print(f"Failure type distribution: {failure_dist}")
    print(f"Failure flags distribution: {failure_flags_dist}")
    print(f"Average agent step count by task type: {avg_agent_steps}")
    print(f"Tool error count summary: {tool_error_summary}")
    if len(input_paths) > 1:
        grouped = (
            df.groupby(["agent_backend", "task_type"], as_index=False)["success"]
            .mean()
            .rename(columns={"success": "success_rate"})
        )
        print("Success by backend and task type:")
        print(grouped.to_string(index=False))


if __name__ == "__main__":
    main()
