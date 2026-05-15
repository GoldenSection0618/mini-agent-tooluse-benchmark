"""Analysis script for benchmark results."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


FIG_DIR = Path("figures")


def _plot_latency_by_task_type(df: pd.DataFrame) -> None:
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
    plt.savefig(FIG_DIR / "latency_by_task_type.png", dpi=150)
    plt.close()


def _plot_success_rate(df: pd.DataFrame) -> None:
    success = df.groupby("task_type", as_index=False)["success"].mean()
    success["success_rate"] = success["success"] * 100

    plt.figure(figsize=(7, 5))
    plt.bar(success["task_type"], success["success_rate"])
    plt.ylim(0, 100)
    plt.ylabel("Success Rate (%)")
    plt.title("Success Rate by Task Type")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "success_rate_by_task_type.png", dpi=150)
    plt.close()


def _plot_failure_distribution(df: pd.DataFrame) -> None:
    failure_counts = df["failure_type"].value_counts().sort_index()

    plt.figure(figsize=(9, 5))
    plt.bar(failure_counts.index, failure_counts.values)
    plt.ylabel("Count")
    plt.title("Failure Type Distribution")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "failure_type_distribution.png", dpi=150)
    plt.close()


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv("results.csv")

    overall_success_rate = df["success"].mean() * 100
    by_type_success = (df.groupby("task_type")["success"].mean() * 100).to_dict()
    avg_wall_clock = df.groupby("task_type")["wall_clock_time_ms"].mean().to_dict()
    avg_tool_latency = df.groupby("task_type")["tool_latency_ms"].mean().to_dict()
    failure_dist = df["failure_type"].value_counts().to_dict()

    _plot_latency_by_task_type(df)
    _plot_success_rate(df)
    _plot_failure_distribution(df)

    print(f"Overall success rate: {overall_success_rate:.2f}%")
    print(f"Success rate by task type: {by_type_success}")
    print(f"Average wall-clock latency by task type (ms): {avg_wall_clock}")
    print(f"Average tool latency by task type (ms): {avg_tool_latency}")
    print(f"Failure type distribution: {failure_dist}")


if __name__ == "__main__":
    main()
