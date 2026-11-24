"""Command line: docs-agent ingest | index | ask "question" | eval"""

from __future__ import annotations

import argparse
import sys

from .config import PROJECT_ROOT, Settings, cohere_api_key


class RetryingClient:
    """Wraps the Cohere client: waits and retries when a rate limit (HTTP 429) is hit.

    Trial keys allow about 20 calls per minute per endpoint, so an eval run hits the limit.
    """

    def __init__(self, client, attempts: int = 6, first_wait: float = 5.0):
        self._client = client
        self._attempts = attempts
        self._first_wait = first_wait

    def __getattr__(self, name):
        method = getattr(self._client, name)
        if name not in ("chat", "embed", "rerank"):
            return method

        def call(*args, **kwargs):
            import time

            from cohere.errors import TooManyRequestsError

            wait = self._first_wait
            for attempt in range(self._attempts):
                try:
                    return method(*args, **kwargs)
                except TooManyRequestsError:
                    if attempt == self._attempts - 1:
                        raise
                    print(f"  rate limited on {name}; waiting {wait:.0f}s", file=sys.stderr)
                    time.sleep(wait)
                    wait = min(wait * 2, 60)

        return call


def _client():
    import cohere

    return RetryingClient(cohere.ClientV2(api_key=cohere_api_key()))


def _agent(settings: Settings, *, rerank: bool = True):
    from .agent import DocsAgent
    from .index import CohereEmbedder, load_index
    from .retrieval import CohereReranker, Retriever
    from .tools import ToolBox

    client = _client()
    chunks, vectors, meta = load_index(settings)
    if meta.get("embed_model") != settings.embed_model:
        sys.exit(
            f"Index was built with {meta.get('embed_model')}, settings say {settings.embed_model}. Re-run `index`."
        )
    retriever = Retriever(
        chunks,
        vectors,
        CohereEmbedder(client, settings.embed_model),
        settings,
        reranker=CohereReranker(client, settings.rerank_model) if rerank else None,
    )
    return DocsAgent(client, ToolBox(retriever), settings), meta


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="docs-agent", description="Tool-using RAG agent over the FastAPI tutorial.")
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("ingest", help="download the FastAPI tutorial (EN + JA) and split it into sections")
    sub.add_parser("index", help="chunk, embed with Cohere and build the keyword index")
    ask = sub.add_parser("ask", help="ask one question")
    ask.add_argument("question")
    ask.add_argument("--baseline", action="store_true", help="single search + answer, no agent loop")
    ask.add_argument("--trace", action="store_true", help="print each tool call")
    ev = sub.add_parser("eval", help="run eval/questions.yaml and write eval/results.md")
    ev.add_argument("--mode", choices=["agent", "baseline", "both"], default="both")
    ev.add_argument("--limit", type=int, default=0, help="only the first N questions")
    ev.add_argument("--no-rerank", action="store_true")
    args = parser.parse_args(argv)
    settings = Settings()

    if args.cmd == "ingest":
        from .ingest import ingest

        ingest(settings)
    elif args.cmd == "index":
        from .index import CohereEmbedder, build_index

        build_index(CohereEmbedder(_client(), settings.embed_model), settings)
    elif args.cmd == "ask":
        agent, _ = _agent(settings)
        result = agent.ask_baseline(args.question) if args.baseline else agent.ask(args.question)
        if args.trace:
            for i, step in enumerate(result.steps, 1):
                print(f"[{i}] {step.tool}({step.args})")
            print()
        print(result.answer)
        print(
            f"\n— {result.mode}: {result.tool_calls} tool calls, {result.seconds}s, "
            f"{result.input_tokens} in / {result.output_tokens} out tokens"
        )
        if result.pages_seen:
            print("Sources: " + ", ".join(result.pages_seen[:5]))
    elif args.cmd == "eval":
        from .evaluate import load_cases, score, summarize, write_report

        agent, meta = _agent(settings, rerank=not args.no_rerank)
        cases = load_cases(PROJECT_ROOT / "eval" / "questions.yaml")
        if args.limit:
            cases = cases[: args.limit]
        modes = ["baseline", "agent"] if args.mode == "both" else [args.mode]
        scores = []
        for case in cases:
            for mode in modes:
                try:
                    result = agent.ask(case.question) if mode == "agent" else agent.ask_baseline(case.question)
                except Exception as exc:  # one failed API call must not lose the whole run
                    from .agent import AgentResult

                    result = AgentResult(question=case.question, answer=f"ERROR: {type(exc).__name__}", mode=mode)
                    print(f"  error on {case.id} ({mode}): {str(exc)[-160:]}", file=sys.stderr)
                s = score(case, result)
                scores.append(s)
                print(f"{'PASS' if s.passed else 'FAIL'}  {mode:8} {case.id:22} {s.tool_calls} calls {s.seconds}s")
        summary = summarize(scores)
        report = write_report(
            scores,
            summary,
            PROJECT_ROOT / "eval",
            {
                "chat_model": settings.chat_model,
                "embed_model": meta.get("embed_model"),
                "rerank_model": "none" if args.no_rerank else settings.rerank_model,
                "fastapi_tag": meta.get("fastapi_tag"),
                "cases": len(cases),
            },
        )
        print(f"\nWrote {report}")
        for mode, m in summary.items():
            print(f"{mode:8} pass {m['pass_rate']}%  ·  {m['avg_tool_calls']} calls  ·  {m['median_seconds']}s median")


if __name__ == "__main__":
    main()
