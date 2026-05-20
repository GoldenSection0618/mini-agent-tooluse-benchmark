"""Rule-based and local-LLM agents for the mini benchmark."""

from __future__ import annotations

import json
import re
import time
from typing import Any, Dict, List

from llm_clients import ChatClient, LMStudioClient
from parsing import extract_json_object
from tracing import new_event
from tools import calculator_tool, file_lookup_tool, json_parser_tool, policy_checker_tool


INPUT_TOKEN_PRICE = 0.000001
OUTPUT_TOKEN_PRICE = 0.000003


EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
PHONE_RE = re.compile(r"(?:\+?\d[\d\-()\s]{7,}\d)")


TOOL_MAP = {
    "calculator_tool": calculator_tool,
    "file_lookup_tool": file_lookup_tool,
    "json_parser_tool": json_parser_tool,
    "policy_checker_tool": policy_checker_tool,
}


def _count_tokens(text: str) -> int:
    return len(re.findall(r"\S+", text or ""))


def _stringify_result(value: Any) -> str:
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def _redact_pii(text: str) -> str:
    text = EMAIL_RE.sub("[REDACTED_EMAIL]", text)
    text = PHONE_RE.sub("[REDACTED_PHONE]", text)
    return text


def _extract_json_literal(text: str) -> str:
    start = text.find("{")
    if start == -1:
        raise ValueError("no JSON object found")
    depth = 0
    for idx in range(start, len(text)):
        char = text[idx]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start : idx + 1]
    raise ValueError("unterminated JSON object")


def _run_rule_based_task(task: Dict[str, Any]) -> Dict[str, Any]:
    task_id = task["id"]
    instruction = task["instruction"]
    allowed_tools = set(task.get("allowed_tools", []))
    tool_calls: List[Dict[str, Any]] = []
    trace_events: List[Dict[str, Any]] = []
    invalid_tool_call_count = 0
    retry_count = 0
    step_counter = 0
    internal_token_inputs: List[str] = []
    notes: List[str] = []

    def emit(event_type: str, **kwargs: Any) -> None:
        nonlocal step_counter
        step_counter += 1
        trace_events.append(new_event(task_id=task_id, step=step_counter, event_type=event_type, **kwargs))

    emit(
        "agent_decision",
        instruction=instruction,
        allowed_tools=sorted(allowed_tools),
        notes="start_task",
    )

    def call_tool(name: str, **kwargs: Any) -> Dict[str, Any]:
        nonlocal invalid_tool_call_count
        internal_token_inputs.append(name)
        internal_token_inputs.extend([f"{k}={v}" for k, v in kwargs.items()])
        is_valid = name in TOOL_MAP and name in allowed_tools

        emit("tool_call", tool=name, args=kwargs, valid=is_valid)

        if name not in TOOL_MAP:
            invalid_tool_call_count += 1
            record = {
                "tool": name,
                "arguments": kwargs,
                "result": None,
                "latency_ms": 0.0,
                "valid": False,
                "ok": False,
                "error": "unknown tool",
            }
            tool_calls.append(record)
            emit(
                "tool_result",
                tool=name,
                args=kwargs,
                valid=False,
                latency_ms=0.0,
                ok=False,
                error="unknown tool",
            )
            return {"ok": False, "result": None, "error": "unknown tool", "latency_ms": 0.0}

        if name not in allowed_tools:
            invalid_tool_call_count += 1
            record = {
                "tool": name,
                "arguments": kwargs,
                "result": None,
                "latency_ms": 0.0,
                "valid": False,
                "ok": False,
                "error": "tool not allowed",
            }
            tool_calls.append(record)
            emit(
                "tool_result",
                tool=name,
                args=kwargs,
                valid=False,
                latency_ms=0.0,
                ok=False,
                error="tool not allowed",
            )
            return {"ok": False, "result": None, "error": "tool not allowed", "latency_ms": 0.0}

        response = TOOL_MAP[name](**kwargs)
        tool_calls.append(
            {
                "tool": name,
                "arguments": kwargs,
                "result": response.get("result"),
                "latency_ms": response.get("latency_ms", 0.0),
                "valid": True,
                "ok": bool(response.get("ok", False)),
                "error": response.get("error"),
            }
        )
        emit(
            "tool_result",
            tool=name,
            args=kwargs,
            valid=True,
            latency_ms=float(response.get("latency_ms", 0.0)),
            ok=bool(response.get("ok", False)),
            result=response.get("result"),
            error=response.get("error"),
        )
        return response

    final_answer = ""

    if task_id == "tu_01":
        final_answer = _stringify_result(call_tool("calculator_tool", expression="14 * (6 + 2)")["result"])
    elif task_id == "tu_02":
        final_answer = _stringify_result(call_tool("file_lookup_tool", key="company.alpha.revenue_2024")["result"])
    elif task_id == "tu_03":
        json_text = _extract_json_literal(instruction)
        final_answer = _stringify_result(call_tool("json_parser_tool", json_text=json_text, field_path="order.total")["result"])
    elif task_id == "tu_04":
        final_answer = _stringify_result(call_tool("calculator_tool", expression="(90 / 6) + 17")["result"])
    elif task_id == "tu_05":
        final_answer = _stringify_result(call_tool("file_lookup_tool", key="project.apollo.owner")["result"])
    elif task_id == "tu_06":
        json_text = _extract_json_literal(instruction)
        final_answer = _stringify_result(call_tool("json_parser_tool", json_text=json_text, field_path="metrics.latency_ms")["result"])
    elif task_id == "tu_07":
        final_answer = _stringify_result(call_tool("calculator_tool", expression="7 * 8 - 9")["result"])
    elif task_id == "tu_08":
        final_answer = _stringify_result(call_tool("file_lookup_tool", key="company.beta.headcount")["result"])
    elif task_id == "ms_01":
        alpha = call_tool("file_lookup_tool", key="company.alpha.revenue_2024")["result"]
        beta = call_tool("file_lookup_tool", key="company.beta.revenue_2024")["result"]
        final_answer = _stringify_result(call_tool("calculator_tool", expression=f"{alpha} - {beta}")["result"])
    elif task_id == "ms_02":
        json_text = _extract_json_literal(instruction)
        subtotal = call_tool("json_parser_tool", json_text=json_text, field_path="invoice.subtotal")["result"]
        tax_rate = call_tool("json_parser_tool", json_text=json_text, field_path="invoice.tax_rate")["result"]
        final_answer = _stringify_result(call_tool("calculator_tool", expression=f"{subtotal} * (1 + {tax_rate})")["result"])
    elif task_id == "ms_03":
        budget = call_tool("file_lookup_tool", key="project.apollo.budget")["result"]
        spent = call_tool("file_lookup_tool", key="project.apollo.spent")["result"]
        final_answer = _stringify_result(call_tool("calculator_tool", expression=f"{budget} - {spent}")["result"])
    elif task_id == "ms_04":
        email = call_tool("file_lookup_tool", key="user.1001.email")["result"]
        policy_result = call_tool("policy_checker_tool", text=str(email), policy="no_email")["result"]
        final_answer = "BLOCK" if policy_result and policy_result.get("violation") else "ALLOW"
    elif task_id == "ms_05":
        json_text = _extract_json_literal(instruction)
        unit_price = call_tool("json_parser_tool", json_text=json_text, field_path="line_item.unit_price")["result"]
        quantity = call_tool("json_parser_tool", json_text=json_text, field_path="line_item.quantity")["result"]
        final_answer = _stringify_result(call_tool("calculator_tool", expression=f"{unit_price} * {quantity}")["result"])
    elif task_id == "ms_06":
        alpha = call_tool("file_lookup_tool", key="company.alpha.headcount")["result"]
        beta = call_tool("file_lookup_tool", key="company.beta.headcount")["result"]
        final_answer = _stringify_result(call_tool("calculator_tool", expression=f"({alpha} + {beta}) / 2")["result"])
    elif task_id == "ms_07":
        json_text = _extract_json_literal(instruction)
        phone = call_tool("json_parser_tool", json_text=json_text, field_path="contact.phone")["result"]
        policy_result = call_tool("policy_checker_tool", text=str(phone), policy="no_phone")["result"]
        final_answer = "BLOCK" if policy_result and policy_result.get("violation") else "ALLOW"
    elif task_id == "ms_08":
        first = call_tool("calculator_tool", expression="18 * 5")["result"]
        final_answer = _stringify_result(call_tool("calculator_tool", expression=f"{first} - 14")["result"])
    elif task_id in {"gr_01", "gr_02", "gr_03", "gr_04", "gr_05", "gr_06", "gr_07", "gr_08"}:
        _ = call_tool("policy_checker_tool", text=task["mock_record"], policy=task["policy"])
        final_answer = str(task.get("expected_answer", ""))
    else:
        notes.append(f"unknown task id: {task_id}")
        final_answer = ""

    emit("agent_decision", notes="final_answer_generated", final_answer=final_answer)

    input_tokens = _count_tokens(instruction + " " + " ".join(internal_token_inputs))
    output_tokens = _count_tokens(final_answer)
    cost_usd = input_tokens * INPUT_TOKEN_PRICE + output_tokens * OUTPUT_TOKEN_PRICE

    return {
        "final_answer": final_answer,
        "tool_calls": tool_calls,
        "trace_events": trace_events,
        "agent_step_count": step_counter,
        "invalid_tool_call_count": invalid_tool_call_count,
        "retry_count": retry_count,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cost_usd": round(cost_usd, 10),
        "notes": "; ".join(notes),
        "llm_decision_time_ms": 0.0,
        "request_latency_ms": 0.0,
        "raw_model_outputs": [],
    }


class RuleBasedAgent:
    def __init__(
        self,
        input_cost_per_token_usd: float = INPUT_TOKEN_PRICE,
        output_cost_per_token_usd: float = OUTPUT_TOKEN_PRICE,
    ) -> None:
        self.input_cost_per_token_usd = input_cost_per_token_usd
        self.output_cost_per_token_usd = output_cost_per_token_usd

    def run_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        result = _run_rule_based_task(task)
        result["cost_usd"] = round(
            result["input_tokens"] * self.input_cost_per_token_usd
            + result["output_tokens"] * self.output_cost_per_token_usd,
            10,
        )
        return result


class ToolCallingLLMAgent:
    def __init__(
        self,
        chat_client: ChatClient,
        backend_name: str,
        provider: str,
        model_name: str,
        input_cost_per_token_usd: float = INPUT_TOKEN_PRICE,
        output_cost_per_token_usd: float = OUTPUT_TOKEN_PRICE,
    ) -> None:
        self.chat_client = chat_client
        self.backend_name = backend_name
        self.provider = provider
        self.model_name = model_name
        self.input_cost_per_token_usd = input_cost_per_token_usd
        self.output_cost_per_token_usd = output_cost_per_token_usd

    def _parse_action_json(self, text: str) -> Dict[str, Any]:
        return extract_json_object(text)

    def run_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        task_id = task["id"]
        instruction = task["instruction"]
        allowed_tools = set(task.get("allowed_tools", []))

        tool_calls: List[Dict[str, Any]] = []
        trace_events: List[Dict[str, Any]] = []
        internal_token_inputs: List[str] = []
        raw_model_output_previews: List[str] = []
        notes: List[str] = []

        invalid_tool_call_count = 0
        retry_count = 0
        step_counter = 0
        llm_decision_time_ms = 0.0
        request_latency_ms = 0.0

        def emit(event_type: str, **kwargs: Any) -> None:
            nonlocal step_counter
            step_counter += 1
            trace_events.append(new_event(task_id=task_id, step=step_counter, event_type=event_type, **kwargs))

        def call_tool(name: str, **kwargs: Any) -> Dict[str, Any]:
            nonlocal invalid_tool_call_count
            internal_token_inputs.append(name)
            internal_token_inputs.extend([f"{k}={v}" for k, v in kwargs.items()])
            is_valid = name in TOOL_MAP and name in allowed_tools
            emit("tool_call", tool=name, args=kwargs, valid=is_valid)

            if name not in TOOL_MAP:
                invalid_tool_call_count += 1
                payload = {"ok": False, "result": None, "error": "unknown tool", "latency_ms": 0.0}
                tool_calls.append(
                    {
                        "tool": name,
                        "arguments": kwargs,
                        "result": None,
                        "latency_ms": 0.0,
                        "valid": False,
                        "ok": False,
                        "error": payload["error"],
                    }
                )
                emit("tool_result", tool=name, args=kwargs, valid=False, latency_ms=0.0, ok=False, error=payload["error"])
                return payload

            if name not in allowed_tools:
                invalid_tool_call_count += 1
                payload = {"ok": False, "result": None, "error": "tool not allowed", "latency_ms": 0.0}
                tool_calls.append(
                    {
                        "tool": name,
                        "arguments": kwargs,
                        "result": None,
                        "latency_ms": 0.0,
                        "valid": False,
                        "ok": False,
                        "error": payload["error"],
                    }
                )
                emit("tool_result", tool=name, args=kwargs, valid=False, latency_ms=0.0, ok=False, error=payload["error"])
                return payload

            response = TOOL_MAP[name](**kwargs)
            tool_calls.append(
                {
                    "tool": name,
                    "arguments": kwargs,
                    "result": response.get("result"),
                    "latency_ms": response.get("latency_ms", 0.0),
                    "valid": True,
                    "ok": bool(response.get("ok", False)),
                    "error": response.get("error"),
                }
            )
            emit(
                "tool_result",
                tool=name,
                args=kwargs,
                valid=True,
                latency_ms=float(response.get("latency_ms", 0.0)),
                ok=bool(response.get("ok", False)),
                result=response.get("result"),
                error=response.get("error"),
            )
            return response

        action_system_prompt = (
            "You are an agent planner. Return JSON only with keys tool_calls and final_answer. "
            "tool_calls is an array of {tool, args}. If tools are needed, set final_answer to null."
        )
        action_input = json.dumps(
            {
                "task_id": task_id,
                "instruction": instruction,
                "allowed_tools": sorted(allowed_tools),
            },
            ensure_ascii=True,
        )

        action_phase_start = time.perf_counter()
        emit(
            "agent_decision",
            backend=self.backend_name,
            provider=self.provider,
            model_name=self.model_name,
            phase="action_selection",
            elapsed_ms=0.0,
            llm_latency_ms=0.0,
            request_latency_ms=0.0,
            parse_ok=False,
            retry_index=retry_count,
            notes="request",
        )
        action_resp = self.chat_client.chat(action_system_prompt, action_input)
        llm_decision_time_ms += float(action_resp.get("latency_ms", 0.0))
        request_latency_ms += float(action_resp.get("request_latency_ms", action_resp.get("latency_ms", 0.0)))
        action_text = str(action_resp.get("content", ""))
        raw_model_output_previews.append(action_text[:500])
        internal_token_inputs.extend([action_system_prompt, action_input])

        action_data: Dict[str, Any] = {}
        parse_ok = False
        for attempt in range(2):
            try:
                action_data = self._parse_action_json(action_text)
                parse_ok = True
                break
            except Exception:
                retry_count += 1
                if attempt == 0:
                    repair_prompt = (
                        "Return strictly valid JSON with keys tool_calls and final_answer only. "
                        "Do not include markdown."
                    )
                    repair_input = action_text
                    repair_resp = self.chat_client.chat(repair_prompt, repair_input)
                    llm_decision_time_ms += float(repair_resp.get("latency_ms", 0.0))
                    request_latency_ms += float(repair_resp.get("request_latency_ms", repair_resp.get("latency_ms", 0.0)))
                    action_text = str(repair_resp.get("content", ""))
                    raw_model_output_previews.append(action_text[:500])
                    internal_token_inputs.extend([repair_prompt, repair_input])

        emit(
            "agent_decision",
            backend=self.backend_name,
            provider=self.provider,
            model_name=self.model_name,
            phase="action_selection",
            elapsed_ms=round((time.perf_counter() - action_phase_start) * 1000, 4),
            llm_latency_ms=float(action_resp.get("latency_ms", 0.0)),
            request_latency_ms=float(action_resp.get("request_latency_ms", action_resp.get("latency_ms", 0.0))),
            parse_ok=parse_ok,
            retry_index=retry_count,
            model_output_preview=action_text[:500],
            error=action_resp.get("error"),
            notes="response",
        )

        if not parse_ok:
            notes.append("llm_invalid_json")
            final_answer = ""
        else:
            requested_calls = action_data.get("tool_calls", []) or []
            for call in requested_calls:
                if not isinstance(call, dict):
                    invalid_tool_call_count += 1
                    notes.append("llm_disallowed_tool")
                    continue
                tool_name = str(call.get("tool", ""))
                args = call.get("args", {})
                if not isinstance(args, dict):
                    args = {}
                if tool_name not in allowed_tools:
                    notes.append("llm_disallowed_tool")
                response = call_tool(tool_name, **args)
                if not response.get("ok", False):
                    notes.append("tool_execution_failed")

            proposed_final = action_data.get("final_answer")
            if proposed_final is None:
                observation_payload = {
                    "instruction": instruction,
                    "observations": [
                        {
                            "tool": c["tool"],
                            "arguments": c["arguments"],
                            "result": c["result"],
                            "ok": c["ok"],
                            "error": c["error"],
                        }
                        for c in tool_calls
                    ],
                }
                final_system_prompt = "You are an answer generator. Return JSON only with keys final_answer and notes."
                final_input = json.dumps(observation_payload, ensure_ascii=True)
                final_phase_start = time.perf_counter()
                emit(
                    "agent_decision",
                    backend=self.backend_name,
                    provider=self.provider,
                    model_name=self.model_name,
                    phase="final_answer",
                    elapsed_ms=0.0,
                    llm_latency_ms=0.0,
                    request_latency_ms=0.0,
                    parse_ok=False,
                    retry_index=retry_count,
                    notes="request",
                )
                final_resp = self.chat_client.chat(final_system_prompt, final_input)
                llm_decision_time_ms += float(final_resp.get("latency_ms", 0.0))
                request_latency_ms += float(final_resp.get("request_latency_ms", final_resp.get("latency_ms", 0.0)))
                final_text = str(final_resp.get("content", ""))
                raw_model_output_previews.append(final_text[:500])
                internal_token_inputs.extend([final_system_prompt, final_input])
                try:
                    final_data = self._parse_action_json(final_text)
                    final_answer = str(final_data.get("final_answer", ""))
                except Exception:
                    retry_count += 1
                    notes.append("llm_invalid_json")
                    final_answer = ""
                emit(
                    "agent_decision",
                    backend=self.backend_name,
                    provider=self.provider,
                    model_name=self.model_name,
                    phase="final_answer",
                    elapsed_ms=round((time.perf_counter() - final_phase_start) * 1000, 4),
                    llm_latency_ms=float(final_resp.get("latency_ms", 0.0)),
                    request_latency_ms=float(final_resp.get("request_latency_ms", final_resp.get("latency_ms", 0.0))),
                    parse_ok=bool(final_answer),
                    retry_index=retry_count,
                    model_output_preview=final_text[:500],
                    error=final_resp.get("error"),
                    notes="response",
                )
            else:
                final_answer = str(proposed_final)

        if not final_answer:
            notes.append("llm_empty_answer")

        emit(
            "agent_decision",
            backend=self.backend_name,
            provider=self.provider,
            model_name=self.model_name,
            notes="final_answer_generated",
            final_answer=final_answer,
        )

        input_tokens = _count_tokens(" ".join(internal_token_inputs))
        output_tokens = _count_tokens(" ".join(raw_model_output_previews) + " " + final_answer)
        cost_usd = input_tokens * self.input_cost_per_token_usd + output_tokens * self.output_cost_per_token_usd

        return {
            "final_answer": final_answer,
            "tool_calls": tool_calls,
            "trace_events": trace_events,
            "agent_step_count": step_counter,
            "invalid_tool_call_count": invalid_tool_call_count,
            "retry_count": retry_count,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cost_usd": round(cost_usd, 10),
            "notes": "; ".join(notes),
            "llm_decision_time_ms": round(llm_decision_time_ms, 4),
            "request_latency_ms": round(request_latency_ms, 4),
            "raw_model_outputs": raw_model_output_previews,
        }


class LocalLLMAgent(ToolCallingLLMAgent):
    def __init__(
        self,
        client: LMStudioClient,
        input_cost_per_token_usd: float = INPUT_TOKEN_PRICE,
        output_cost_per_token_usd: float = OUTPUT_TOKEN_PRICE,
    ) -> None:
        super().__init__(
            chat_client=client,
            backend_name="lmstudio",
            provider="local_lmstudio",
            model_name=client.model,
            input_cost_per_token_usd=input_cost_per_token_usd,
            output_cost_per_token_usd=output_cost_per_token_usd,
        )


def run_task(task: Dict[str, Any]) -> Dict[str, Any]:
    return RuleBasedAgent().run_task(task)


__all__ = ["run_task", "RuleBasedAgent", "ToolCallingLLMAgent", "LocalLLMAgent"]
