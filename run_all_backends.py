"""Run available benchmark backends in sequence and generate analysis artifacts."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import List, Tuple

from llm_clients import DeepSeekClient, LMStudioClient, deepseek_preflight


def run_cmd(args: List[str]) -> bool:
    print(f"$ {' '.join(args)}")
    proc = subprocess.run(args)
    return proc.returncode == 0


def check_lmstudio() -> Tuple[bool, str]:
    client = LMStudioClient(
        base_url="http://localhost:1234",
        chat_endpoint="/api/v1/chat",
        model="google/gemma-4-e4b",
        temperature=0,
        max_tokens=256,
        timeout_seconds=20,
    )
    result = client.chat("You are a concise assistant.", "Reply with OK only.")
    if not result.get("ok", False):
        return False, str(result.get("error", "preflight failed"))
    return True, "ok"


def check_deepseek() -> Tuple[bool, str]:
    if not os.environ.get("DEEPSEEK_API_KEY", "").strip():
        return False, "DEEPSEEK_API_KEY is not set"
    client = DeepSeekClient(
        base_url="https://api.deepseek.com",
        chat_endpoint="/chat/completions",
        model="deepseek-v4-flash",
        temperature=0,
        max_tokens=256,
        timeout_seconds=30,
        thinking={"type": "disabled"},
    )
    if not deepseek_preflight(client):
        return False, "preflight failed"
    return True, "ok"


def main() -> None:
    py = sys.executable
    completed_results: List[str] = []

    if run_cmd([py, "benchmark.py", "--agent", "rule_based"]):
        completed_results.append("results_rule_based.csv")
    else:
        print("rule_based run failed; continue to optional checks")

    lm_ok, lm_reason = check_lmstudio()
    if lm_ok:
        if run_cmd(
            [
                py,
                "benchmark.py",
                "--agent",
                "lmstudio",
                "--base-url",
                "http://localhost:1234",
                "--chat-endpoint",
                "/api/v1/chat",
                "--model",
                "google/gemma-4-e4b",
                "--output",
                "results_lmstudio.csv",
                "--trace-dir",
                "traces_lmstudio",
            ]
        ):
            completed_results.append("results_lmstudio.csv")
        else:
            print("lmstudio run failed")
    else:
        print(f"skip lmstudio: {lm_reason}")

    ds_ok, ds_reason = check_deepseek()
    if ds_ok:
        if run_cmd(
            [
                py,
                "benchmark.py",
                "--agent",
                "deepseek",
                "--base-url",
                "https://api.deepseek.com",
                "--chat-endpoint",
                "/chat/completions",
                "--model",
                "deepseek-v4-flash",
                "--output",
                "results_deepseek.csv",
                "--trace-dir",
                "traces_deepseek",
            ]
        ):
            completed_results.append("results_deepseek.csv")
        else:
            print("deepseek run failed")
    else:
        print(f"skip deepseek: {ds_reason}")

    existing_results = [f for f in completed_results if Path(f).exists()]
    if existing_results:
        single_backend = existing_results[0]
        run_cmd([py, "analysis.py", "--input", single_backend, "--figures-dir", f"figures_{Path(single_backend).stem}"])

    if len(existing_results) >= 2:
        run_cmd([py, "analysis.py", "--input", *existing_results, "--figures-dir", "figures_compare"])


if __name__ == "__main__":
    main()
