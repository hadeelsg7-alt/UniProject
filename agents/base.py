"""Shared agent machinery: tool calling with allow-lists, logging, and an
LLM-or-fallback reasoning step."""
from __future__ import annotations

import json
from typing import Any, Callable, Type, TypeVar

from pydantic import BaseModel

from llm import LLMClient, LLMError
from schemas import ToolCall, WorkflowState
from tools import TOOL_REGISTRY

T = TypeVar("T", bound=BaseModel)


def _summ(obj: Any, limit: int = 220) -> str:
    try:
        if isinstance(obj, BaseModel):
            obj = obj.model_dump()
        s = json.dumps(obj, ensure_ascii=False, default=str)
    except TypeError:
        s = str(obj)
    return s if len(s) <= limit else s[: limit - 3] + "..."


class BaseAgent:
    name = "Agent"
    role = ""
    instructions = ""

    def __init__(self, llm: LLMClient | None = None):
        self.llm = llm or LLMClient(provider="offline")

    # ------------------------------------------------------------------ #
    def call_tool(self, state: WorkflowState, tool: str, reason: str, **kwargs) -> Any:
        spec = TOOL_REGISTRY.get(tool)
        if spec is None:
            raise KeyError(f"Unknown tool {tool}")
        if self.name not in spec.agents:
            raise PermissionError(f"{self.name} is not allowed to use tool {tool}")
        result = spec.func(**kwargs)
        state.tool_calls.append(ToolCall(agent=self.name, tool=tool, reason=reason,
                                         input_summary=_summ(kwargs), output_summary=_summ(result)))
        state.log(self.name, "TOOL_CALL", f"{tool}: {reason} -> {_summ(result, 160)}", tool=tool)
        return result

    def reason(self, state: WorkflowState, step: str, user: str, schema: Type[T], fallback: Callable[[], T]) -> tuple[T, str]:
        """Run the LLM reasoning step; on any LLM failure fall back to the
        transparent rule-based reasoner and record it in the log."""
        if self.llm.available:
            try:
                out = self.llm.structured(self.instructions, user, schema)
                state.log(self.name, "LLM_REASONING", f"{step}: structured output from {self.llm.name}")
                return out, self.llm.name
            except LLMError as e:
                state.errors.append(f"{self.name}/{step}: {e}")
                state.log(self.name, "LLM_FALLBACK", f"{step}: LLM unavailable ({str(e)[:120]}); using rule-based reasoner")
        out = fallback()
        state.log(self.name, "RULE_REASONING", f"{step}: rule-based reasoner (offline)")
        return out, "offline-rule-based"
