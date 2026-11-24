# Evaluation results

Chat model `command-a-03-2025` · embeddings `embed-v4.0` · rerank `rerank-v3.5` · FastAPI docs `0.165.0` · 17 questions

| Metric | agent | baseline |
|---|---|---|
| Pass rate (all) | 94.1% | 94.1% |
| Pass rate, English questions | 90.9% | 90.9% |
| Pass rate, Japanese questions | 100.0% | 100.0% |
| Required facts in answer | 93.8% | 93.8% |
| Right page found | 100.0% | 100.0% |
| Said 'not covered' when it wasn't | 100.0% | 100.0% |
| Tool calls per question | 2.82 | 1 |
| Median seconds per question | 13.51 | 6.05 |
| Input tokens per question | 8876 | 3087 |
| Output tokens per question | 585 | 322 |

## Per question

| Question | agent | baseline |
|---|---|---|
| en-status-201 | pass (1 calls) | pass (1 calls) |
| en-query-maxlen | pass (2 calls) | pass (1 calls) |
| en-exclude-unset | pass (2 calls) | pass (1 calls) |
| en-background | fail (2 calls) | pass (1 calls) |
| en-cors | pass (2 calls) | fail (1 calls) |
| en-upload | pass (1 calls) | pass (1 calls) |
| en-routers | pass (2 calls) | pass (1 calls) |
| en-yield-cleanup | pass (2 calls) | pass (1 calls) |
| en-oauth2-token-url | pass (2 calls) | pass (1 calls) |
| en-enum-path | pass (2 calls) | pass (1 calls) |
| ja-status-201 | pass (1 calls) | pass (1 calls) |
| ja-query-maxlen | pass (1 calls) | pass (1 calls) |
| ja-background | pass (1 calls) | pass (1 calls) |
| ja-cors | pass (2 calls) | pass (1 calls) |
| ja-routers | pass (23 calls) | pass (1 calls) |
| ja-yield-cleanup | pass (1 calls) | pass (1 calls) |
| en-out-of-scope | pass (1 calls) | pass (1 calls) |
