"""Run the question set and score answers deterministically (no LLM judge)."""

from __future__ import annotations

import json
import re
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml

from .agent import AgentResult

# Phrases that count as "the tutorial doesn't cover this".
ABSTAIN_PATTERNS = [
    r"\bnot (?:covered|cover|include|mention|discuss)",
    r"\bdoes(?:n't| not) (?:cover|include|mention|discuss|say)",
    r"\bno information\b",
    r"\boutside (?:the )?(?:scope|tutorial)",
    r"記載(?:が|は)?(?:ありません|ない)",
    r"扱われていません|含まれていません|見つかりません",
]


@dataclass
class Case:
    id: str
    lang: str
    question: str
    pages: list[str]
    facts: list[list[str]]
    abstain: bool = False


@dataclass
class Score:
    id: str
    mode: str
    lang: str
    passed: bool
    facts_ok: bool
    evidence_ok: bool
    abstain_ok: bool | None
    tool_calls: int
    input_tokens: int
    output_tokens: int
    seconds: float
    answer: str


def load_cases(path: Path) -> list[Case]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    return [
        Case(
            id=r["id"],
            lang=r["lang"],
            question=r["question"],
            pages=r.get("pages", []),
            facts=[[str(x) for x in group] for group in r.get("facts", [])],
            abstain=bool(r.get("abstain", False)),
        )
        for r in raw
    ]


def facts_present(answer: str, facts: list[list[str]]) -> bool:
    low = answer.lower()
    return all(any(alt.lower() in low for alt in group) for group in facts)


def evidence_found(result: AgentResult, pages: list[str]) -> bool:
    # Accept either language edition: "en/tutorial/cors" and "ja/tutorial/cors" are the same page.
    seen = {p.split("/", 1)[1] for p in result.pages_seen}
    return any(p in seen for p in pages)


def abstained(answer: str) -> bool:
    return any(re.search(p, answer, re.I) for p in ABSTAIN_PATTERNS)


def score(case: Case, result: AgentResult) -> Score:
    if case.abstain:
        ok = abstained(result.answer)
        facts_ok = evidence_ok = ok
        abstain_ok: bool | None = ok
    else:
        facts_ok = facts_present(result.answer, case.facts)
        evidence_ok = evidence_found(result, case.pages)
        abstain_ok = None
        ok = facts_ok and evidence_ok
    return Score(
        id=case.id,
        mode=result.mode,
        lang=case.lang,
        passed=ok,
        facts_ok=facts_ok,
        evidence_ok=evidence_ok,
        abstain_ok=abstain_ok,
        tool_calls=result.tool_calls,
        input_tokens=result.input_tokens,
        output_tokens=result.output_tokens,
        seconds=result.seconds,
        answer=result.answer,
    )


def summarize(scores: list[Score]) -> dict[str, Any]:
    def pct(xs: list[bool]) -> float:
        return round(100 * sum(xs) / len(xs), 1) if xs else 0.0

    out: dict[str, Any] = {}
    for mode in sorted({s.mode for s in scores}):
        ms = [s for s in scores if s.mode == mode]
        answerable = [s for s in ms if s.abstain_ok is None]
        out[mode] = {
            "cases": len(ms),
            "pass_rate": pct([s.passed for s in ms]),
            "pass_rate_en": pct([s.passed for s in ms if s.lang == "en"]),
            "pass_rate_ja": pct([s.passed for s in ms if s.lang == "ja"]),
            "facts_correct": pct([s.facts_ok for s in answerable]),
            "right_page_found": pct([s.evidence_ok for s in answerable]),
            "abstained_when_uncovered": pct([bool(s.abstain_ok) for s in ms if s.abstain_ok is not None]),
            "avg_tool_calls": round(statistics.mean(s.tool_calls for s in ms), 2),
            "median_seconds": round(statistics.median(s.seconds for s in ms), 2),
            "avg_input_tokens": round(statistics.mean(s.input_tokens for s in ms)),
            "avg_output_tokens": round(statistics.mean(s.output_tokens for s in ms)),
        }
    return out


def write_report(scores: list[Score], summary: dict[str, Any], out_dir: Path, meta: dict[str, Any]) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "results.json").write_text(
        json.dumps(
            {"meta": meta, "summary": summary, "cases": [asdict(s) for s in scores]}, ensure_ascii=False, indent=2
        ),
        encoding="utf-8",
    )
    modes = list(summary)
    rows = [
        ("Pass rate (all)", "pass_rate", "%"),
        ("Pass rate, English questions", "pass_rate_en", "%"),
        ("Pass rate, Japanese questions", "pass_rate_ja", "%"),
        ("Required facts in answer", "facts_correct", "%"),
        ("Right page found", "right_page_found", "%"),
        ("Said 'not covered' when it wasn't", "abstained_when_uncovered", "%"),
        ("Tool calls per question", "avg_tool_calls", ""),
        ("Median seconds per question", "median_seconds", ""),
        ("Input tokens per question", "avg_input_tokens", ""),
        ("Output tokens per question", "avg_output_tokens", ""),
    ]
    lines = [
        "# Evaluation results",
        "",
        f"Chat model `{meta['chat_model']}` · embeddings `{meta['embed_model']}` · rerank `{meta['rerank_model']}` · "
        f"FastAPI docs `{meta['fastapi_tag']}` · {meta['cases']} questions",
        "",
        "| Metric | " + " | ".join(modes) + " |",
        "|---|" + "---|" * len(modes),
    ]
    for label, key, unit in rows:
        lines.append(f"| {label} | " + " | ".join(f"{summary[m][key]}{unit}" for m in modes) + " |")
    lines += ["", "## Per question", "", "| Question | " + " | ".join(modes) + " |", "|---|" + "---|" * len(modes)]
    for qid in dict.fromkeys(s.id for s in scores):
        cells = []
        for m in modes:
            s = next((x for x in scores if x.id == qid and x.mode == m), None)
            cells.append("—" if s is None else ("pass" if s.passed else "fail") + f" ({s.tool_calls} calls)")
        lines.append(f"| {qid} | " + " | ".join(cells) + " |")
    path = out_dir / "results.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path
