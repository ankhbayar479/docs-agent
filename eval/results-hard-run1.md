# Evaluation results

Chat model `command-a-03-2025` · embeddings `embed-v4.0` · rerank `rerank-v3.5` · FastAPI docs `0.165.0` · 8 questions

| Metric | agent | baseline |
|---|---|---|
| Pass rate (all) | 62.5% | 87.5% |
| Pass rate, English questions | 66.7% | 83.3% |
| Pass rate, Japanese questions | 50.0% | 100.0% |
| Required facts in answer | 62.5% | 100.0% |
| Right page found | 87.5% | 87.5% |
| Said 'not covered' when it wasn't | 0.0% | 0.0% |
| Tool calls per question | 1.88 | 1 |
| Median seconds per question | 13.69 | 4.6 |
| Input tokens per question | 8751 | 3131 |
| Output tokens per question | 441 | 311 |

## Per question

| Question | agent | baseline |
|---|---|---|
| hard-jwt | pass (2 calls) | pass (1 calls) |
| hard-jwt-ja | fail (3 calls) | pass (1 calls) |
| hard-router-201 | fail (2 calls) | fail (1 calls) |
| hard-patch | pass (2 calls) | pass (1 calls) |
| hard-file-form | fail (2 calls) | pass (1 calls) |
| hard-file-form-ja | pass (1 calls) | pass (1 calls) |
| hard-hide-param | pass (1 calls) | pass (1 calls) |
| hard-deprecated | pass (2 calls) | pass (1 calls) |
