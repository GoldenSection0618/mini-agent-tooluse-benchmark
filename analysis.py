"""Analysis script for oracle-level benchmark results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import pandas as pd


BACKEND_ORDER = ["rule_based", "lmstudio", "deepseek"]
TASK_TYPE_ORDER = ["tool_use", "multi_step", "guardrail"]

BACKEND_LABELS = {
    "rule_based": "Rule-based\noracle",
    "lmstudio": "LM Studio\nGemma",
    "deepseek": "DeepSeek",
}

TASK_TYPE_LABELS = {
    "tool_use": "Tool use",
    "multi_step": "Multi-step",
    "guardrail": "Guardrail",
}

BACKEND_COLORS = {
    "rule_based": "#8A8A8A",
    "lmstudio": "#B65A5A",
    "deepseek": "#2F5F8F",
}

TASK_TYPE_COLORS = {
    "tool_use": "#4C78A8",
    "multi_step": "#9FBAD6",
    "guardrail": "#C99A5B",
}

METRIC_COLOR = "#4C78A8"
NEUTRAL_DARK = "#333333"
NEUTRAL_MID = "#737373"
NEUTRAL_LIGHT = "#D9D9D9"


def _set_publication_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 7.5,
            "axes.labelsize": 7.5,
            "axes.titlesize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "legend.title_fontsize": 7,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 0.7,
            "xtick.major.width": 0.7,
            "ytick.major.width": 0.7,
            "xtick.major.size": 2.6,
            "ytick.major.size": 2.6,
            "legend.frameon": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def _ordered_values(values: Iterable[str], preferred: list[str]) -> list[str]:
    present = [str(v) for v in values]
    ordered = [v for v in preferred if v in present]
    ordered.extend(sorted(v for v in present if v not in ordered))
    return ordered


def _backend_label(name: Any) -> str:
    return BACKEND_LABELS.get(str(name), str(name))


def _backend_label_short(name: Any) -> str:
    labels = {
        "rule_based": "Oracle",
        "lmstudio": "Gemma",
        "deepseek": "DeepSeek",
    }
    return labels.get(str(name), str(name))


def _task_type_label(name: Any) -> str:
    return TASK_TYPE_LABELS.get(str(name), str(name))


def _metric_label(name: str) -> str:
    return name.replace("_", " ")


def _flag_label(name: str) -> str:
    return name.replace("_", " ")


def _save_figure(fig: mpl.figure.Figure, path: Path, dpi: int = 600) -> None:
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def _panel_label(ax: mpl.axes.Axes, label: str) -> None:
    ax.text(
        -0.12,
        1.06,
        label,
        transform=ax.transAxes,
        fontsize=9,
        fontweight="bold",
        va="bottom",
        ha="left",
    )


def _soften_axes(ax: mpl.axes.Axes, *, y_grid: bool = False, x_grid: bool = False) -> None:
    ax.spines["left"].set_color(NEUTRAL_DARK)
    ax.spines["bottom"].set_color(NEUTRAL_DARK)
    ax.tick_params(colors=NEUTRAL_DARK)
    if y_grid:
        ax.grid(axis="y", color=NEUTRAL_LIGHT, linewidth=0.45, alpha=0.75)
        ax.set_axisbelow(True)
    if x_grid:
        ax.grid(axis="x", color=NEUTRAL_LIGHT, linewidth=0.45, alpha=0.75)
        ax.set_axisbelow(True)


def _annotate_bars(ax: mpl.axes.Axes, bars: Iterable[Any], *, fmt: str = "{:.0f}") -> None:
    for bar in bars:
        height = float(bar.get_height())
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + 2,
            fmt.format(height),
            ha="center",
            va="bottom",
            fontsize=6.5,
            color=NEUTRAL_DARK,
        )


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
    order = _ordered_values(latency["task_type"], TASK_TYPE_ORDER)
    latency = latency.set_index("task_type").loc[order].reset_index()
    y = list(range(len(latency)))

    fig, ax_wall = plt.subplots(figsize=(4.3, 2.5))
    labels = [_task_type_label(t) for t in latency["task_type"]]
    ax_wall.barh(y, latency["wall_clock_time_ms"], color="#748DB4", edgecolor="white", linewidth=0.4)
    ax_wall.set_yticks(y, labels)
    ax_wall.invert_yaxis()
    ax_wall.set_xlabel("Wall-clock latency (ms)")
    ax_wall.set_xlim(0, float(latency["wall_clock_time_ms"].max()) * 1.12)
    _soften_axes(ax_wall, x_grid=True)

    ax_tool = ax_wall.twiny()
    tool_latency_us = latency["tool_latency_ms"] * 1000.0
    for yi, value in zip(y, tool_latency_us):
        ax_tool.hlines(yi, 0, value, color="#C99A5B", linewidth=1.0, zorder=3)
        ax_tool.vlines(value, yi - 0.22, yi + 0.22, color="#C99A5B", linewidth=1.0, zorder=3)
    ax_tool.set_xlim(0, max(float(tool_latency_us.max()) * 1.15, 1.0))
    ax_tool.set_xlabel("Tool execution latency (us)", color="#8A642C")
    ax_tool.tick_params(axis="x", colors="#8A642C")
    ax_tool.tick_params(axis="y", left=False, labelleft=False)
    ax_tool.spines["top"].set_visible(True)
    ax_tool.spines["top"].set_color("#8A642C")
    ax_tool.spines["bottom"].set_visible(False)
    _save_figure(fig, fig_dir / "latency_by_task_type.png")


def _plot_success_rate(df: pd.DataFrame, fig_dir: Path) -> None:
    summary = df.groupby("task_type", as_index=False)[["success", "final_answer_correct"]].mean()
    summary[["success", "final_answer_correct"]] = summary[["success", "final_answer_correct"]] * 100
    order = _ordered_values(summary["task_type"], TASK_TYPE_ORDER)
    summary = summary.set_index("task_type").loc[order].reset_index()
    x = list(range(len(summary)))
    width = 0.35

    fig, ax = plt.subplots(figsize=(3.6, 2.5))
    ax.bar(
        [i - width / 2 for i in x],
        summary["success"],
        width=width,
        label="Strict success",
        color="#2F5F8F",
        edgecolor="white",
        linewidth=0.4,
    )
    ax.bar(
        [i + width / 2 for i in x],
        summary["final_answer_correct"],
        width=width,
        label="Final answer",
        color="#B7CBE2",
        edgecolor="white",
        linewidth=0.4,
    )
    ax.set_xticks(x, [_task_type_label(t) for t in summary["task_type"]])
    ax.set_ylim(0, 105)
    ax.set_ylabel("Rate (%)")
    ax.legend(loc="upper right")
    _soften_axes(ax, y_grid=True)
    _save_figure(fig, fig_dir / "success_rate_by_task_type.png")


def _plot_failure_distribution(df: pd.DataFrame, fig_dir: Path) -> None:
    failure_counts = df[df["failure_type"] != "none"]["failure_type"].value_counts().sort_values()

    fig, ax = plt.subplots(figsize=(3.8, 2.6))
    if not failure_counts.empty:
        ax.barh(
            [_flag_label(str(i)) for i in failure_counts.index],
            failure_counts.values,
            color="#8E5D5D",
            edgecolor="white",
            linewidth=0.4,
        )
    ax.set_xlabel("Task count")
    _soften_axes(ax, x_grid=True)
    _save_figure(fig, fig_dir / "failure_type_distribution.png")


def _plot_failure_flags_distribution(df: pd.DataFrame, fig_dir: Path) -> None:
    flags = []
    for raw in df["failure_flags_list"]:
        parsed = raw if isinstance(raw, list) else []
        flags.extend(parsed)

    counts = pd.Series(flags).value_counts().sort_values() if flags else pd.Series(dtype="int64")
    fig, ax = plt.subplots(figsize=(4.2, 3.0))
    if not counts.empty:
        ax.barh(
            [_flag_label(str(i)) for i in counts.index],
            counts.values,
            color="#B65A5A",
            edgecolor="white",
            linewidth=0.4,
        )
    ax.set_xlabel("Flag count")
    _soften_axes(ax, x_grid=True)
    _save_figure(fig, fig_dir / "failure_flags_distribution.png")


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
    rates = pd.Series({col: df[col].mean() * 100 for col in metrics}).sort_values()

    fig, ax = plt.subplots(figsize=(4.4, 3.0))
    ax.barh([_metric_label(m) for m in rates.index], rates.values, color=METRIC_COLOR, edgecolor="white", linewidth=0.4)
    ax.set_xlim(0, 105)
    ax.set_xlabel("Pass rate (%)")
    _soften_axes(ax, x_grid=True)
    _save_figure(fig, fig_dir / "oracle_metric_breakdown.png")


def _plot_success_rate_by_backend(df: pd.DataFrame, fig_dir: Path) -> None:
    summary = df.groupby("agent_backend", as_index=False)["success"].mean()
    summary["success_rate"] = summary["success"] * 100
    order = _ordered_values(summary["agent_backend"], BACKEND_ORDER)
    summary = summary.set_index("agent_backend").loc[order].reset_index()
    colors = [BACKEND_COLORS.get(b, NEUTRAL_MID) for b in summary["agent_backend"]]

    fig, ax = plt.subplots(figsize=(3.2, 2.5))
    bars = ax.bar(
        [_backend_label(b) for b in summary["agent_backend"]],
        summary["success_rate"],
        color=colors,
        edgecolor="white",
        linewidth=0.4,
    )
    ax.set_ylim(0, 105)
    ax.set_ylabel("Strict success (%)")
    _annotate_bars(ax, bars)
    _soften_axes(ax, y_grid=True)
    _save_figure(fig, fig_dir / "success_rate_by_backend.png")


def _plot_latency_by_backend(df: pd.DataFrame, fig_dir: Path) -> None:
    summary = df.groupby("agent_backend", as_index=False)[["wall_clock_time_ms", "tool_latency_ms"]].mean()
    order = _ordered_values(summary["agent_backend"], BACKEND_ORDER)
    summary = summary.set_index("agent_backend").loc[order].reset_index()
    y = list(range(len(summary)))

    fig, ax_wall = plt.subplots(figsize=(4.3, 2.5))
    labels = [_backend_label_short(b) for b in summary["agent_backend"]]
    ax_wall.barh(y, summary["wall_clock_time_ms"], color="#748DB4", edgecolor="white", linewidth=0.4)
    ax_wall.set_yticks(y, labels)
    ax_wall.invert_yaxis()
    ax_wall.set_xscale("log")
    ax_wall.set_xlabel("Wall-clock latency (ms)")
    positive_wall = summary.loc[summary["wall_clock_time_ms"] > 0, "wall_clock_time_ms"]
    if not positive_wall.empty:
        ax_wall.set_xlim(float(positive_wall.min()) * 0.6, float(positive_wall.max()) * 1.8)
    _soften_axes(ax_wall, x_grid=True)

    ax_tool = ax_wall.twiny()
    tool_latency_us = summary["tool_latency_ms"] * 1000.0
    for yi, value in zip(y, tool_latency_us):
        ax_tool.hlines(yi, 0, value, color="#C99A5B", linewidth=1.0, zorder=3)
        ax_tool.vlines(value, yi - 0.22, yi + 0.22, color="#C99A5B", linewidth=1.0, zorder=3)
    ax_tool.set_xlim(0, max(float(tool_latency_us.max()) * 1.15, 1.0))
    ax_tool.set_xlabel("Tool execution latency (us)", color="#8A642C")
    ax_tool.tick_params(axis="x", colors="#8A642C")
    ax_tool.tick_params(axis="y", left=False, labelleft=False)
    ax_tool.spines["top"].set_visible(True)
    ax_tool.spines["top"].set_color("#8A642C")
    ax_tool.spines["bottom"].set_visible(False)
    _save_figure(fig, fig_dir / "latency_by_backend.png")


def _plot_success_rate_by_backend_and_task_type(df: pd.DataFrame, fig_dir: Path) -> None:
    pivot = (
        df.pivot_table(index="agent_backend", columns="task_type", values="success", aggfunc="mean", fill_value=0.0)
        * 100
    )
    backend_order = _ordered_values(pivot.index, BACKEND_ORDER)
    task_order = _ordered_values(pivot.columns, TASK_TYPE_ORDER)
    pivot = pivot.loc[backend_order, task_order]
    fig, ax = plt.subplots(figsize=(4.2, 2.8))
    y = list(range(len(pivot.index)))
    height = 0.22
    offsets = [(-height), 0, height]
    for idx, task_type in enumerate(pivot.columns):
        offset = offsets[idx] if idx < len(offsets) else (idx - len(pivot.columns) / 2) * height
        ax.barh(
            [i + offset for i in y],
            pivot[task_type],
            height=height,
            label=_task_type_label(task_type),
            color=TASK_TYPE_COLORS.get(task_type, NEUTRAL_MID),
            edgecolor="white",
            linewidth=0.4,
        )
    ax.set_yticks(y, [_backend_label_short(b) for b in pivot.index])
    ax.invert_yaxis()
    ax.set_xlim(0, 105)
    ax.set_xlabel("Strict success (%)")
    ax.legend(
        loc="lower left",
        bbox_to_anchor=(0, 1.02),
        ncol=3,
        handlelength=1.2,
        columnspacing=0.9,
    )
    _soften_axes(ax, x_grid=True)
    _save_figure(fig, fig_dir / "success_rate_by_backend_and_task_type.png")


def _plot_failure_type_by_backend(df: pd.DataFrame, fig_dir: Path) -> None:
    with mpl.rc_context(mpl.rcParamsDefault):
        pivot = (
            df.pivot_table(
                index="agent_backend",
                columns="failure_type",
                values="task_id",
                aggfunc="count",
                fill_value=0,
            )
        )
        ax = pivot.plot(kind="bar", figsize=(10, 5))
        ax.set_ylabel("Count")
        ax.set_title("Failure Type by Backend")
        ax.tick_params(axis="x", rotation=0)
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
    with mpl.rc_context(mpl.rcParamsDefault):
        pivot = flags_df.groupby(["agent_backend", "failure_flag"]).size().unstack(fill_value=0)
        ax = pivot.plot(kind="bar", figsize=(11, 5))
        ax.set_ylabel("Count")
        ax.set_title("Failure Flags by Backend")
        ax.tick_params(axis="x", rotation=0)
        ax.legend(title="failure_flag", bbox_to_anchor=(1.02, 1), loc="upper left")
        plt.tight_layout()
        plt.savefig(fig_dir / "failure_flags_by_backend.png", dpi=150)
        plt.close()


def _plot_tool_match_by_backend(df: pd.DataFrame, fig_dir: Path) -> None:
    summary = df.groupby("agent_backend", as_index=False)[["tool_sequence_match", "tool_argument_match"]].mean()
    summary[["tool_sequence_match", "tool_argument_match"]] = (
        summary[["tool_sequence_match", "tool_argument_match"]] * 100
    )
    order = _ordered_values(summary["agent_backend"], BACKEND_ORDER)
    summary = summary.set_index("agent_backend").loc[order].reset_index()
    x = list(range(len(summary)))
    width = 0.35

    fig, ax = plt.subplots(figsize=(3.7, 2.5))
    ax.bar(
        [i - width / 2 for i in x],
        summary["tool_sequence_match"],
        width=width,
        label="Sequence",
        color="#2F5F8F",
        edgecolor="white",
        linewidth=0.4,
    )
    ax.bar(
        [i + width / 2 for i in x],
        summary["tool_argument_match"],
        width=width,
        label="Arguments",
        color="#9FBAD6",
        edgecolor="white",
        linewidth=0.4,
    )
    ax.set_xticks(x, [_backend_label(b) for b in summary["agent_backend"]])
    ax.set_ylim(0, 105)
    ax.set_ylabel("Pass rate (%)")
    ax.legend(loc="upper right")
    _soften_axes(ax, y_grid=True)
    _save_figure(fig, fig_dir / "tool_sequence_match_by_backend.png")

    fig, ax = plt.subplots(figsize=(3.2, 2.5))
    bars = ax.bar(
        [_backend_label(b) for b in summary["agent_backend"]],
        summary["tool_argument_match"],
        color=[BACKEND_COLORS.get(b, NEUTRAL_MID) for b in summary["agent_backend"]],
        edgecolor="white",
        linewidth=0.4,
    )
    ax.set_ylim(0, 105)
    ax.set_ylabel("Argument match (%)")
    _annotate_bars(ax, bars)
    _soften_axes(ax, y_grid=True)
    _save_figure(fig, fig_dir / "tool_argument_match_by_backend.png")


def _plot_benchmark_overview(df: pd.DataFrame, fig_dir: Path) -> None:
    backend_order = _ordered_values(df["agent_backend"].unique(), BACKEND_ORDER)
    task_order = _ordered_values(df["task_type"].unique(), TASK_TYPE_ORDER)

    fig = plt.figure(figsize=(7.4, 5.8), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, width_ratios=[1.04, 1.0], height_ratios=[0.92, 1.2])

    ax_a = fig.add_subplot(gs[0, 0])
    success_task = (
        df.pivot_table(index="agent_backend", columns="task_type", values="success", aggfunc="mean", fill_value=0.0)
        * 100
    ).reindex(index=backend_order, columns=task_order)
    im = ax_a.imshow(success_task.values, vmin=0, vmax=100, cmap="Blues", aspect="auto")
    ax_a.set_xticks(range(len(task_order)), [_task_type_label(t) for t in task_order], rotation=25, ha="right")
    ax_a.set_yticks(range(len(backend_order)), [_backend_label_short(b) for b in backend_order])
    for row_idx, backend in enumerate(success_task.index):
        for col_idx, task_type in enumerate(success_task.columns):
            value = success_task.loc[backend, task_type]
            color = "white" if value >= 70 else NEUTRAL_DARK
            ax_a.text(col_idx, row_idx, f"{value:.0f}", ha="center", va="center", fontsize=7, color=color)
    ax_a.tick_params(length=0)
    for spine in ax_a.spines.values():
        spine.set_visible(False)
    cbar = fig.colorbar(im, ax=ax_a, fraction=0.046, pad=0.02)
    cbar.set_label("Strict success (%)", fontsize=7)
    cbar.ax.tick_params(labelsize=6.5, length=2)
    _panel_label(ax_a, "a")

    ax_b = fig.add_subplot(gs[0, 1])
    metrics = [
        "final_answer_correct",
        "required_tools_called",
        "tool_sequence_match",
        "tool_argument_match",
        "format_correct",
        "output_policy_clean",
    ]
    metric_rates = pd.Series({m: df[m].mean() * 100 for m in metrics}).sort_values()
    ax_b.barh(
        [_metric_label(m) for m in metric_rates.index],
        metric_rates.values,
        color=METRIC_COLOR,
        edgecolor="white",
        linewidth=0.4,
    )
    ax_b.set_xlim(0, 105)
    ax_b.set_xlabel("Pass rate (%)")
    _soften_axes(ax_b, x_grid=True)
    _panel_label(ax_b, "b")

    ax_c = fig.add_subplot(gs[1, 0])
    rows = []
    for _, row in df.iterrows():
        for flag in row.get("failure_flags_list", []):
            rows.append({"agent_backend": row["agent_backend"], "failure_flag": flag})
    if rows:
        flags_df = pd.DataFrame(rows)
        pivot = flags_df.groupby(["failure_flag", "agent_backend"]).size().unstack(fill_value=0)
        pivot = pivot.reindex(columns=backend_order, fill_value=0)
        pivot = pivot.loc[pivot.sum(axis=1).sort_values().tail(8).index]
        pivot.plot(
            kind="barh",
            ax=ax_c,
            color=[BACKEND_COLORS.get(b, NEUTRAL_MID) for b in pivot.columns],
            edgecolor="white",
            linewidth=0.4,
        )
        ax_c.set_yticklabels([_flag_label(str(t.get_text())) for t in ax_c.get_yticklabels()])
    ax_c.set_xlabel("Flag count")
    ax_c.set_ylabel("")
    legend_backends = [b for b in backend_order if b in {"lmstudio", "deepseek"}]
    ax_c.legend(
        handles=[Patch(facecolor=BACKEND_COLORS.get(b, NEUTRAL_MID), label=_backend_label(b).replace("\n", " ")) for b in legend_backends],
        loc="lower left",
        bbox_to_anchor=(0, 1.02),
        ncol=2,
        handlelength=1.2,
        columnspacing=0.9,
    )
    _soften_axes(ax_c, x_grid=True)
    _panel_label(ax_c, "c")

    ax_d = fig.add_subplot(gs[1, 1])
    latency = df.groupby("agent_backend")[["wall_clock_time_ms", "tool_latency_ms"]].mean().reindex(backend_order)
    y = list(range(len(latency.index)))
    ax_d.barh(y, latency["wall_clock_time_ms"], color="#748DB4", edgecolor="white", linewidth=0.4)
    ax_d.set_yticks(y, [_backend_label_short(b) for b in latency.index])
    ax_d.invert_yaxis()
    ax_d.set_xscale("log")
    positive_wall = latency.loc[latency["wall_clock_time_ms"] > 0, "wall_clock_time_ms"]
    if not positive_wall.empty:
        ax_d.set_xlim(float(positive_wall.min()) * 0.6, float(positive_wall.max()) * 1.8)
    ax_d.set_xlabel("Wall-clock latency (ms)")
    _soften_axes(ax_d, x_grid=True)

    ax_d_tool = ax_d.twiny()
    tool_latency_us = latency["tool_latency_ms"] * 1000.0
    for yi, value in zip(y, tool_latency_us):
        ax_d_tool.hlines(yi, 0, value, color="#C99A5B", linewidth=1.0, zorder=3)
        ax_d_tool.vlines(value, yi - 0.22, yi + 0.22, color="#C99A5B", linewidth=1.0, zorder=3)
    ax_d_tool.set_xlim(0, max(float(tool_latency_us.max()) * 1.15, 1.0))
    ax_d_tool.set_xlabel("Tool execution latency (us)", color="#8A642C")
    ax_d_tool.tick_params(axis="x", colors="#8A642C", labelsize=6.5)
    ax_d_tool.tick_params(axis="y", left=False, labelleft=False)
    ax_d_tool.spines["top"].set_visible(True)
    ax_d_tool.spines["top"].set_color("#8A642C")
    ax_d_tool.spines["bottom"].set_visible(False)
    _panel_label(ax_d, "d")

    _save_figure(fig, fig_dir / "benchmark_overview.png")


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Analyze benchmark result files.")
    parser.add_argument("--input", nargs="*", default=None, help="Input results CSV path(s).")
    parser.add_argument("--figures-dir", default="figures", help="Directory to save figures.")
    return parser


def main() -> None:
    _set_publication_style()
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
        _plot_benchmark_overview(df, fig_dir)
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
