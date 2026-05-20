"""Local LLM clients for benchmark backends."""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from typing import Any, Dict


class ChatClient:
    def chat(self, system_prompt: str, user_input: str) -> Dict[str, Any]:
        raise NotImplementedError


def extract_lmstudio_content(response_json: Dict[str, Any]) -> str:
    if isinstance(response_json.get("content"), str):
        return str(response_json["content"])

    message = response_json.get("message")
    if isinstance(message, dict) and isinstance(message.get("content"), str):
        return str(message["content"])

    choices = response_json.get("choices")
    if isinstance(choices, list) and choices:
        first = choices[0]
        if isinstance(first, dict):
            msg = first.get("message")
            if isinstance(msg, dict) and isinstance(msg.get("content"), str):
                return str(msg["content"])

    output = response_json.get("output")
    if isinstance(output, str):
        return str(output)
    if isinstance(output, list) and output:
        first = output[0]
        if isinstance(first, dict) and isinstance(first.get("content"), str):
            return str(first["content"])

    if isinstance(response_json.get("text"), str):
        return str(response_json["text"])

    return json.dumps(response_json, ensure_ascii=True)


class LMStudioClient(ChatClient):
    def __init__(
        self,
        base_url: str,
        chat_endpoint: str,
        model: str,
        temperature: float = 0,
        max_tokens: int = 512,
        timeout_seconds: int = 120,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.chat_endpoint = chat_endpoint if chat_endpoint.startswith("/") else f"/{chat_endpoint}"
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout_seconds = timeout_seconds

    @property
    def chat_url(self) -> str:
        return f"{self.base_url}{self.chat_endpoint}"

    def _post_json(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        body = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            self.chat_url,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=self.timeout_seconds) as resp:
            response_bytes = resp.read()
        return json.loads(response_bytes.decode("utf-8"))

    def chat(self, system_prompt: str, user_input: str) -> Dict[str, Any]:
        start = time.perf_counter()

        payload_full = {
            "model": self.model,
            "system_prompt": system_prompt,
            "input": user_input,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        payload_min = {
            "model": self.model,
            "system_prompt": system_prompt,
            "input": user_input,
        }

        try:
            response_json = self._post_json(payload_full)
            latency_ms = (time.perf_counter() - start) * 1000
            return {
                "ok": True,
                "content": extract_lmstudio_content(response_json),
                "latency_ms": latency_ms,
                "request_latency_ms": latency_ms,
                "raw_response": response_json,
                "error": None,
                "usage": {},
            }
        except urllib.error.HTTPError as exc:
            # Retry with minimal payload in case endpoint rejects temperature/max_tokens.
            try:
                if 400 <= exc.code < 500:
                    response_json = self._post_json(payload_min)
                    latency_ms = (time.perf_counter() - start) * 1000
                    return {
                        "ok": True,
                        "content": extract_lmstudio_content(response_json),
                        "latency_ms": latency_ms,
                        "request_latency_ms": latency_ms,
                        "raw_response": response_json,
                        "error": None,
                        "usage": {},
                    }
            except Exception:
                pass

            latency_ms = (time.perf_counter() - start) * 1000
            try:
                error_body = exc.read().decode("utf-8")
            except Exception:
                error_body = str(exc)
            return {
                "ok": False,
                "content": "",
                "latency_ms": latency_ms,
                "request_latency_ms": latency_ms,
                "raw_response": None,
                "error": f"HTTPError {exc.code}: {error_body}",
                "usage": {},
            }
        except Exception as exc:
            latency_ms = (time.perf_counter() - start) * 1000
            return {
                "ok": False,
                "content": "",
                "latency_ms": latency_ms,
                "request_latency_ms": latency_ms,
                "raw_response": None,
                "error": str(exc),
                "usage": {},
            }


__all__ = ["ChatClient", "LMStudioClient", "extract_lmstudio_content"]
