# Evaluation results

Chat model `command-a-03-2025` · embeddings `embed-v4.0` · rerank `rerank-v3.5` · FastAPI docs `0.165.0` · 8 questions

| Metric | agent |
|---|---|
| Pass rate (all) | 87.5% |
| Pass rate, English questions | 83.3% |
| Pass rate, Japanese questions | 100.0% |
| Required facts in answer | 87.5% |
| Right page found | 87.5% |
| Said 'not covered' when it wasn't | 0.0% |
| Tool calls per question | 1.38 |
| Median seconds per question | 6.58 |
| Input tokens per question | 5615 |
| Output tokens per question | 442 |

## Per question

| Question | agent |
|---|---|
| hard-jwt | pass (1 calls) |
| hard-jwt-ja | pass (1 calls) |
| hard-router-201 | fail (2 calls) |
| hard-patch | pass (2 calls) |
| hard-file-form | pass (1 calls) |
| hard-file-form-ja | pass (1 calls) |
| hard-hide-param | pass (1 calls) |
| hard-deprecated | pass (2 calls) |
