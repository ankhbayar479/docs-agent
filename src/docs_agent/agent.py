"""A tool-using agent (plan -> call tools -> observe -> repeat) on Cohere's chat API,
plus a single-shot RAG baseline to compare against."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any

from .config import Settings
from .text import detect_lang
from .tools import TOOL_SPECS, ToolBox

LANG_NAMES = {"en": "English", "ja": "Japanese"}

SYSTEM_PROMPT = """You answer questions about FastAPI using only its official tutorial, available through tools.

How to work:
- Always call search_docs before answering. If the results are weak, search again with different words,
  or in the other language (the tutorial exists in English and Japanese).
- When a snippet is cut off or you need details (exact parameter names, defaults, code), call read_section.
- Stop as soon as you can answer. Usually one search and at most two read_section calls are enough.

How to answer:
- Answer in the language named at the end of the user's message, even if the sources are in the other language.
- Be concise and concrete: name the exact classes, functions and parameters.
- When the tutorial shows code for it, include a short code example using the exact names from the tutorial.
- If the tutorial does not cover the question, say so instead of guessing.
"""

# Extra rules for the agent only (the single-shot baseline has no tools to act on them).
# Added after the hard question set showed the agent stopping at the first plausible page
# and filling the rest from memory.
AGENT_RULES = """
Before you answer (coverage check):
- Split the question into its parts (for example: which type, which package, which setting).
- For each part, find the sentence or code in a tool result that supports it.
- If any part is not supported yet, search for that part specifically (often it is on a different page).
- Never answer a part from memory. If the tutorial does not cover a part, say so for that part.
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

    def _chat(self, **kwargs):
        s = self.settings
        return self.client.chat(model=s.chat_model, temperature=s.temperature, thinking={"type": s.thinking}, **kwargs)

    def _prepare(self, question: str) -> str:
        lang = detect_lang(question)
        self.toolbox.default_lang = lang
        return f"{question}\n\n(Answer in {LANG_NAMES[lang]}.)"

    def ask(self, question: str) -> AgentResult:
        s = self.settings
        result = AgentResult(question=question, answer="", mode="agent")
        messages: list[Any] = [
            {"role": "system", "content": SYSTEM_PROMPT + AGENT_RULES},
            {"role": "user", "content": self._prepare(question)},
        ]
        started = time.perf_counter()
        self._seed_search(question, messages, result)
        for _ in range(s.max_steps):
            resp = self._chat(messages=messages, tools=TOOL_SPECS)
            _add_usage(result, resp)
            calls = resp.message.tool_calls or []
            if not calls:
                result.answer = _final_text(resp)
                break
            assistant: dict[str, Any] = {"role": "assistant", "tool_calls": calls}
            # Cohere only accepts a tool plan back when thinking is on.
            if s.thinking == "enabled" and resp.message.tool_plan:
                assistant["tool_plan"] = resp.message.tool_plan
            messages.append(assistant)
            for call in calls:
                args = json.loads(call.function.arguments or "{}")
                if len(result.steps) >= s.max_tool_calls:
                    output = {"error": "tool budget used up; answer now with what you have"}
                else:
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
            resp = self._chat(messages=messages)
            _add_usage(result, resp)
            result.answer = _final_text(resp)
        result.seconds = round(time.perf_counter() - started, 2)
        return result

    def _seed_search(self, question: str, messages: list[Any], result: AgentResult) -> None:
        """Run the first search in code, with the full question, before the model acts.

        The hard question set showed the agent writing narrow queries and then seeing less
        context than the single-shot baseline. Seeding guarantees it starts with the same
        evidence; the model then decides what else to look up.
        """
        args = {"query": question, "top_k": 5}
        output = self.toolbox.call("search_docs", args)
        _note_pages(result, output)
        payload = json.dumps(output, ensure_ascii=False)
        result.steps.append(Step("search_docs", args, payload[:160]))
        call_id = "seed-search"
        messages.append(
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": call_id,
                        "type": "function",
                        "function": {"name": "search_docs", "arguments": json.dumps(args)},
                    }
                ],
            }
        )
        messages.append(
            {"role": "tool", "tool_call_id": call_id, "content": [{"type": "document", "document": {"data": payload}}]}
        )

    def ask_baseline(self, question: str, top_k: int = 5) -> AgentResult:
        """Single-shot RAG: one search with the raw question, then answer. No planning, no follow-up reads."""
        result = AgentResult(question=question, answer="", mode="baseline")
        started = time.perf_counter()
        user_message = self._prepare(question)
        output = self.toolbox.call("search_docs", {"query": question, "top_k": top_k})
        _note_pages(result, output)
        result.steps.append(Step("search_docs", {"query": question}, f"{len(output.get('results', []))} results"))
        documents = [
            {"id": r["chunk_id"], "data": {"title": r["title"], "url": r["url"], "text": r["snippet"]}}
            for r in output.get("results", [])
        ]
        resp = self._chat(
            messages=[{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user_message}],
            documents=documents,
        )
        _add_usage(result, resp)
        result.answer = _final_text(resp)
        result.seconds = round(time.perf_counter() - started, 2)
        return result
