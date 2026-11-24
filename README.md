# docs-agent

A tool-using RAG agent that answers questions about the [FastAPI tutorial](https://fastapi.tiangolo.com/tutorial/) in **English and Japanese**, built on Cohere (Command, Embed, Rerank), with a **deterministic evaluation** that compares the agent against single-shot RAG.

A real session (trimmed):

```
$ docs-agent ask "My React app on localhost:3000 is blocked when calling my API. What do I add to FastAPI?" --trace
[1] search_docs({'query': 'CORS React localhost 3000 FastAPI', 'lang': 'en', 'top_k': 5})
[2] read_section({'section_id': 'en/tutorial/cors#4'})

To fix your React app being blocked ... add CORS middleware:

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    ...
)
— agent: 2 tool calls
```

## Why an agent, not just RAG

Single-shot RAG embeds the user's question, takes the top chunks and answers. That fails when the question uses different words than the docs, when the answer needs a detail cut off in a snippet, or when the best source is in the other language.

The agent plans and uses three tools until it has enough evidence:

| Tool | What it does |
|---|---|
| `search_docs(query, lang, top_k)` | Hybrid, two-stage search (below). The agent rewrites the query, retries, or switches language. |
| `list_sections(page_id)` | Outline of one page, to choose what to read. |
| `read_section(section_id)` | Full text of a section, when a snippet is not enough. |

```mermaid
flowchart LR
    Q[Question EN / JA] --> A{Agent<br/>Cohere Command}
    A -- tool call --> S[search_docs]
    A -- tool call --> L[list_sections]
    A -- tool call --> R[read_section]
    S --> H[Hybrid scoring<br/>vectors 0.7 + BM25 0.3]
    H --> P[Stage 1: rank pages]
    P --> C[Stage 2: chunks from top pages]
    C --> RR[Cohere Rerank]
    RR --> A
    L --> A
    R --> A
    A -- enough evidence --> ANS[Answer in the question's language]
```

## Retrieval design

- **Bilingual corpus:** the 51 tutorial pages in both English and Japanese (FastAPI `0.165.0`, pinned), with the 183 code files they reference inlined, split on headings into ~960 sections and ~1,290 chunks of at most 1,200 characters with 150 characters of overlap.
- **Hybrid scoring:** `embed-v4.0` vector similarity combined with BM25 keyword scores (0.7 / 0.3 after min-max normalization). Keywords matter in API docs, where exact names like `response_model_exclude_unset` carry the meaning.
- **Japanese keyword search without a tokenizer dependency:** Japanese has no spaces, so the BM25 tokenizer uses character bigrams for Japanese and words for Latin script.
- **Two-stage retrieval:** stage 1 ranks whole *pages* (best chunk + ¼ of the second best), so one page with two good matches beats scattered single hits. Stage 2 takes candidate chunks only from the top 3 pages, always including some from the best page, then **Cohere Rerank** orders them.
- **Headings in front of every chunk** before embedding, because a section titled "Data validation" means more than its first paragraph.

## Evaluation

`eval/questions.yaml` has 17 questions: 10 English, 6 Japanese, and 1 the tutorial doesn't cover. Answers in either language are scored the same way. Scoring is done **in code, not by an LLM judge**, so results are reproducible:

- **Facts:** every required group of terms appears in the answer (for example `CORSMiddleware` *and* `allow_origins`).
- **Evidence:** the agent's tools returned or read the right page (either language edition counts).
- **Abstention:** for the uncovered question, the answer says the tutorial doesn't cover it instead of inventing.

Each question runs twice: as the **agent** and as the **single-shot RAG baseline** (one search with the raw question, then answer). The report also tracks tool calls, latency and tokens per question.

### Results

Chat `command-a-03-2025`, embeddings `embed-v4.0`, rerank `rerank-v3.5`, 17 questions. Full tables: [`eval/results.md`](eval/results.md) (latest) and [`eval/results-run1-no-code-examples.md`](eval/results-run1-no-code-examples.md).

| | Run 1: code examples missing from the index | Run 2: code examples inlined |
|---|---|---|
| Agent pass rate | 70.6% | **94.1%** |
| Single-shot RAG pass rate | 64.7% | **94.1%** |
| Right page found (both modes) | 100% | 100% |
| Agent tool calls / question | 1.9 | 2.8 |
| Agent vs baseline input tokens | 2.9× | 2.9× |

### What the evaluation taught me

1. **Data quality beat model cleverness.** In run 1 every failure had the right page but an incomplete answer: no `status_code=201`, no `include_router`. FastAPI's docs pull code from separate files (`{* ../../docs_src/... *}`), and my ingest step dropped those references. Inlining the 183 referenced files raised both modes from 65–71% to 94%. Retrieval was never the problem; the evaluation showed that in one run.
2. **On single-lookup questions the agent is not worth its cost.** Both modes reach 94.1%, but the agent uses ~2.9× the input tokens and about twice the time. (This run used agent v1, before the changes below.)
3. **On harder questions, the first agent was worse than plain RAG, and instructions did not fix it; code did.** See the hard set below.
4. **Operational lessons from real runs:**
   - Cohere's newest model "thinks" before every reply by default: 12 s for a one-sentence answer vs 0.7 s with thinking off. Thinking is off by default here (`COHERE_THINKING`).
   - One run made 23 tool calls on one question, because a model can request several tools in a single step. Total tool calls are now capped (8), not just steps.
   - Two languages need the language decided in code: an English question once got a Japanese answer after the first search returned Japanese pages. Searches now default to the question's language and the answer language is stated explicitly.
   - Trial keys allow ~20 calls/minute per endpoint, so the client waits and retries on HTTP 429, and one failed call no longer aborts an evaluation run.

### Hard question set

[`eval/questions-hard.yaml`](eval/questions-hard.yaml): 8 questions (6 English, 2 Japanese) that need several details from different sections, and one that needs two different pages. The single-shot baseline scored 87.5% in both of its runs (1 call, ~3,100 input tokens).

| Agent version | Pass rate | Tool calls | Input tokens | Report |
|---|---|---|---|---|
| v1: the model writes its own first query | 62.5% | 1.9 | ~8,800 | [run 1](eval/results-hard-run1.md) |
| v2: + "coverage check" instructions in the prompt | 37.5% | 2.1 | ~9,100 | [run 2](eval/results-hard-run2-coverage-prompt.md) |
| **v3: + first search done in code, with the full question** | **87.5%** | **1.4** | **~5,600** | [run 3](eval/results-hard-run3-seeded-search.md) |

- **v1 failed by stopping early.** It wrote a narrow query, found the file-upload page, and answered "no extra package is required" from memory, while the baseline saw five snippets from three pages and got `python-multipart` right. The agent was seeing *less* context than the baseline.
- **v2 asked the model to check that every part of the question was supported before answering. It did not change the behaviour** (still about 2 calls, then answer); with 8 questions, part of the drop is run-to-run noise.
- **v3 fixed it in code:** the first search always uses the full question, so the agent starts with exactly the baseline's evidence and then decides what else to look up. It matches the baseline with fewer calls and tokens than v1.
- **It still does not beat the baseline.** The one question that needs two pages (`hard-router-201`: router dependencies *and* status 201) fails in every version and in the baseline; nothing searches a second time for the missing part. Next step: split the question into parts in code and search for each.
- One question has a flaw I left in place rather than rewrite after seeing results: "how long do access tokens last by default" can be read as 15 minutes (the helper's default) or 30 (the setting the tutorial uses); answers saying 15 are scored as failures.

## Quick start

```bash
git clone https://github.com/ankhbayar479/docs-agent && cd docs-agent
uv sync
cp .env.example .env            # add your COHERE_API_KEY (a free trial key works)

uv run docs-agent ingest        # download the tutorial (EN + JA), split into sections
uv run docs-agent index         # chunk + embed with Cohere (~1,290 chunks)
uv run docs-agent ask "クエリパラメータを最大50文字に制限するには？" --trace
uv run docs-agent eval          # agent vs baseline -> eval/results.md
```

Tests run offline with fake Cohere clients (no key needed): `uv run pytest`.

## Project layout

```
src/docs_agent/
  ingest.py      download + clean markdown + split on headings
  text.py        EN/JA tokenizer, chunking with overlap
  index.py       chunks, Cohere embeddings, saved index
  retrieval.py   hybrid scoring, two-stage page -> chunk retrieval, rerank
  tools.py       tool schemas (Cohere format) + implementations
  agent.py       the plan / tool-call / observe loop, and the single-shot baseline
  evaluate.py    deterministic scoring and the results report
  cli.py         docs-agent ingest | index | ask | eval
eval/questions.yaml
tests/           offline tests with scripted fake clients
```

## Limits and next steps

- Questions are about one product's docs; a broader set would need more pages and harder multi-hop questions.
- Fact checks are string matches: strict and reproducible, but they can miss a correct answer phrased differently.
- The eval sets are small (17 + 8 questions), so differences of one or two questions are within run-to-run noise.
- The 17-question results above were measured with agent v1; v3 has not been re-run on them yet (trial-key call limits).
- Next: question decomposition for multi-page questions, streaming answers with citations, and caching query embeddings.

## License

Code: MIT. The FastAPI documentation is © Sebastián Ramírez and contributors, MIT-licensed; it is downloaded at run time, not stored in this repository.
