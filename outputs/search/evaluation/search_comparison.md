# Search Relevance Evaluation

| Model | ndcg@5 | ndcg@10 | ndcg@20 | mrr | latency_p50_ms | latency_p99_ms |
|---|---|---|---|---|---|---|
| Full Pipeline | 0.7646 | 0.8170 | 0.8170 | 0.8303 | 39.2 | 48.3 |
| Bi + Cross-Encoder | 0.7632 | 0.8159 | 0.8159 | 0.8303 | 39.4 | 66.9 |
| BM25 | 0.7343 | 0.7651 | 0.8629 | 0.8681 | 32.8 | 90.8 |
| Bi-Encoder | 0.6038 | 0.6418 | 0.7466 | 0.7630 | 7.0 | 163.4 |
