"""A tool-using agent (plan -> call tools -> observe -> repeat) on Cohere's chat API,
plus a single-shot RAG baseline to compare against."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any

from .config import Settings
from .tools import TOOL_SPECS, ToolBox

SYSTEM_PROMPT = """You answer questions about FastAPI using only its official tutorial, available through tools.

How to work:
- Always call search_docs before answering. If the results are weak, search again with different words,
  or in the other language (the tutorial exists in English and Japanese).
- When a snippet is cut off or you need details (exact parameter names, defaults, code), call read_section.
- Stop as soon as you can answer. Do not call tools you don't need.

How to answer:
- Answer in the same language as the question.
- Be concise and concrete: name the exact classes, functions and parameters.
- If the tutorial does not cover the question, say so instead of guessing.
"""


@dataclass
class Step:
    tool: str
    args: dict[str, Any]
    result_preview: str


@dataclass
class AgentResult:
    question: str
    answer: str
    mode: str
    steps: list[Step] = field(default_factory=list)
    pages_seen: list[str] = field(default_factory=list)  # pages returned or read by tools, in order
    input_tokens: int = 0
    output_tokens: int = 0
    seconds: float = 0.0
    stopped_early: bool = False

    @property
    def tool_calls(self) -> int:
        return len(self.steps)


def _add_usage(result: AgentResult, resp) -> None:
    tokens = getattr(getattr(resp, "usage", None), "tokens", None)
    if tokens:
        result.input_tokens += int(tokens.input_tokens or 0)
        result.output_tokens += int(tokens.output_tokens or 0)


def _final_text(resp) -> str:
    content = getattr(resp.message, "content", None) or []
    return "".join(getattr(part, "text", "") for part in content).strip()


def _note_pages(result: AgentResult, output: dict[str, Any]) -> None:
    pages = [r["page_id"] for r in output.get("results", [])]
    if "section_id" in output:
        pages.append(output["section_id"].split("#")[0])
    for p in pages:
        if p not in result.pages_seen:
            result.pages_seen.append(p)


class DocsAgent:
    def __init__(self, client, toolbox: ToolBox, settings: Settings):
        self.client = client
        self.toolbox = toolbox
        self.settings = settings

    def ask(self, question: str) -> AgentResult:
        s = self.settings
        result = AgentResult(question=question, answer="", mode="agent")
        messages: list[Any] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ]
        started = time.perf_counter()
        for _ in range(s.max_steps):
            resp = self.client.chat(model=s.chat_model, messages=messages, tools=TOOL_SPECS, temperature=s.temperature)
            _add_usage(result, resp)
            calls = resp.message.tool_calls or []
            if not calls:
                result.answer = _final_text(resp)
                break
            messages.append({"role": "assistant", "tool_plan": resp.message.tool_plan, "tool_calls": calls})
            for call in calls:
                args = json.loads(call.function.arguments or "{}")
                output = self.toolbox.call(call.function.name, args)
                _note_pages(result, output)
                payload = json.dumps(output, ensure_ascii=False)
                result.steps.append(Step(call.function.name, args, payload[:160]))
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call.id,
                        "content": [{"type": "document", "document": {"data": payload}}],
                    }
                )
        else:
            # Out of steps: ask for the best answer from what was gathered, with tools disabled.
            result.stopped_early = True
            resp = self.client.chat(model=s.chat_model, messages=messages, temperature=s.temperature)
            _add_usage(result, resp)
            result.answer = _final_text(resp)
        result.seconds = round(time.perf_counter() - started, 2)
        return result

    def ask_baseline(self, question: str, top_k: int = 5) -> AgentResult:
        """Single-shot RAG: one search with the raw question, then answer. No planning, no follow-up reads."""
        s = self.settings
        result = AgentResult(question=question, answer="", mode="baseline")
        started = time.perf_counter()
        output = self.toolbox.call("search_docs", {"query": question, "top_k": top_k})
        _note_pages(result, output)
        result.steps.append(Step("search_docs", {"query": question}, f"{len(output.get('results', []))} results"))
        documents = [
            {"id": r["section_id"], "data": {"title": r["title"], "url": r["url"], "text": r["snippet"]}}
            for r in output.get("results", [])
        ]
        resp = self.client.chat(
            model=s.chat_model,
            messages=[{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": question}],
            documents=documents,
            temperature=s.temperature,
        )
        _add_usage(result, resp)
        result.answer = _final_text(resp)
        result.seconds = round(time.perf_counter() - started, 2)
        return result
