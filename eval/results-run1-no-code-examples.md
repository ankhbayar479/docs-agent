# Evaluation results

Chat model `command-a-03-2025` · embeddings `embed-v4.0` · rerank `rerank-v3.5` · FastAPI docs `0.165.0` · 17 questions

| Metric | agent | baseline |
|---|---|---|
| Pass rate (all) | 70.6% | 64.7% |
| Pass rate, English questions | 72.7% | 54.5% |
| Pass rate, Japanese questions | 66.7% | 83.3% |
| Required facts in answer | 68.8% | 68.8% |
| Right page found | 100.0% | 100.0% |
| Said 'not covered' when it wasn't | 100.0% | 0.0% |
| Tool calls per question | 1.88 | 1 |
| Median seconds per question | 5.6 | 3.88 |
| Input tokens per question | 8246 | 2884 |
| Output tokens per question | 342 | 190 |

## Per question

| Question | agent | baseline |
|---|---|---|
| en-status-201 | fail (2 calls) | fail (1 calls) |
| en-query-maxlen | pass (2 calls) | pass (1 calls) |
| en-exclude-unset | pass (1 calls) | pass (1 calls) |
| en-background | fail (2 calls) | fail (1 calls) |
| en-cors | pass (2 calls) | fail (1 calls) |
| en-upload | pass (1 calls) | pass (1 calls) |
| en-routers | fail (2 calls) | fail (1 calls) |
| en-yield-cleanup | pass (2 calls) | pass (1 calls) |
| en-oauth2-token-url | pass (2 calls) | pass (1 calls) |
| en-enum-path | pass (2 calls) | pass (1 calls) |
| ja-status-201 | fail (2 calls) | fail (1 calls) |
| ja-query-maxlen | pass (4 calls) | pass (1 calls) |
| ja-background | pass (1 calls) | pass (1 calls) |
| ja-cors | pass (2 calls) | pass (1 calls) |
| ja-routers | fail (2 calls) | pass (1 calls) |
| ja-yield-cleanup | pass (2 calls) | pass (1 calls) |
| en-out-of-scope | pass (1 calls) | fail (1 calls) |
