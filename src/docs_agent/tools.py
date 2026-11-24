"""Tools the agent can call, in Cohere's tool format, plus their Python implementations."""

from __future__ import annotations

from typing import Any

from .retrieval import Retriever

TOOL_SPECS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "search_docs",
            "description": (
                "Search the FastAPI tutorial (English and Japanese). Returns the best-matching passages "
                "with their page_id and section_id. Search again with different wording if results are weak."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "What to look for, in English or Japanese."},
                    "lang": {
                        "type": "string",
                        "enum": ["en", "ja", "any"],
                        "description": "Language edition. Default: the question's language. Use any for both.",
                    },
                    "top_k": {"type": "integer", "description": "Number of passages, 1-8. Default 5."},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_sections",
            "description": "List the sections of one page (section_id and title), to decide what to read next.",
            "parameters": {
                "type": "object",
                "properties": {"page_id": {"type": "string", "description": "e.g. en/tutorial/path-params"}},
                "required": ["page_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_section",
            "description": "Read the full text of one section, when a search snippet is not enough to answer.",
            "parameters": {
                "type": "object",
                "properties": {"section_id": {"type": "string", "description": "e.g. en/tutorial/path-params#3"}},
                "required": ["section_id"],
            },
        },
    },
]

SNIPPET_CHARS = 700


class ToolBox:
    def __init__(self, retriever: Retriever):
        self.retriever = retriever
        # Set per question by the agent: searches default to the question's language.
        self.default_lang = "any"

    def call(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        try:
            if name == "search_docs":
                lang = args.get("lang") or self.default_lang
                top_k = max(1, min(int(args.get("top_k") or 5), 8))
                hits = self.retriever.search(str(args["query"]), lang=None if lang == "any" else lang, top_k=top_k)
                return {
                    "results": [
                        {
                            "chunk_id": h.chunk.chunk_id,
                            "page_id": h.chunk.page_id,
                            "section_id": h.chunk.section_id,
                            "title": f"{h.chunk.page_title} > {h.chunk.section_title}",
                            "url": h.chunk.url,
                            "score": round(h.score, 3),
                            "snippet": h.chunk.text[:SNIPPET_CHARS],
                        }
                        for h in hits
                    ]
                }
            if name == "list_sections":
                outline = self.retriever.outline(str(args["page_id"]))
                if not outline:
                    return {"error": f"unknown page_id {args['page_id']!r}; use an id returned by search_docs"}
                return {"sections": [{"section_id": sid, "title": t} for sid, t in outline]}
            if name == "read_section":
                sid = str(args["section_id"])
                if "#" not in sid and self.retriever.outline(sid):
                    # The model passed a page_id: show the page's sections instead of failing.
                    return {
                        "note": "That is a page_id. Pick a section_id from this list and call read_section again.",
                        "sections": [{"section_id": s, "title": t} for s, t in self.retriever.outline(sid)],
                    }
                text = self.retriever.section_text(sid)
                if text is None:
                    return {"error": f"unknown section_id {args['section_id']!r}"}
                chunk = self.retriever.by_section[str(args["section_id"])][0]
                return {"section_id": chunk.section_id, "title": chunk.section_title, "url": chunk.url, "text": text}
            return {"error": f"unknown tool {name!r}"}
        except (KeyError, ValueError, TypeError) as exc:
            # Bad arguments go back to the model as data so it can correct itself.
            return {"error": f"bad arguments for {name}: {exc}"}
