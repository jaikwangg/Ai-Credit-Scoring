# Advisor Ablation Study

Total records analysed: **83**

### Overall

| Approach | N | Latency (s) | Checks | Pass | Fail | Unkn | Actions | Sources | Keyword Recall | Verdict Acc. | IsSup |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A1 — Profile-conditioned (single-hop) | 25 | 41.0 ± 19.2 | 5.3 ± 1.5 | 3.48 | 0.48 | 1.12 | 3.36 | 9.12 | 0.90 (n=24) | 0.56 (n=18) | — |
| A2 — A1 + Multi-hop decomposition | 27 | 64.9 ± 20.8 | 5.4 ± 1.3 | 3.37 | 0.52 | 1.37 | 3.48 | 9.04 | 0.85 (n=25) | 0.68 (n=19) | — |
| A3 — A2 + Self-RAG reflection | 31 | 85.6 ± 21.6 | 5.4 ± 1.8 | 3.00 | 0.52 | 1.52 | 3.55 | 8.68 | 0.79 (n=30) | 0.46 (n=24) | 3.30 |

### By type: advice

| Approach | N | Latency (s) | Checks | Pass | Fail | Unkn | Actions | Sources | Keyword Recall | Verdict Acc. | IsSup |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A1 — Profile-conditioned (single-hop) | 3 | 36.6 ± 7.2 | 5.3 ± 0.5 | 4.00 | 0.00 | 1.00 | 3.67 | 7.67 | 1.00 (n=2) | — (n=0) | — |
| A2 — A1 + Multi-hop decomposition | 4 | 68.3 ± 30.7 | 5.8 ± 0.8 | 4.00 | 0.00 | 1.25 | 3.50 | 6.50 | 1.00 (n=2) | — (n=0) | — |
| A3 — A2 + Self-RAG reflection | 3 | 89.0 ± 15.1 | 5.7 ± 0.5 | 4.00 | 0.00 | 1.33 | 4.00 | 9.00 | 1.00 (n=2) | — (n=0) | 3.00 |

### By type: factual

| Approach | N | Latency (s) | Checks | Pass | Fail | Unkn | Actions | Sources | Keyword Recall | Verdict Acc. | IsSup |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A1 — Profile-conditioned (single-hop) | 4 | 39.5 ± 18.8 | 4.2 ± 1.6 | 3.50 | 0.00 | 0.75 | 3.00 | 8.75 | 0.92 (n=4) | — (n=0) | — |
| A2 — A1 + Multi-hop decomposition | 4 | 55.9 ± 6.3 | 5.8 ± 1.8 | 3.50 | 0.00 | 1.75 | 3.25 | 10.50 | 0.83 (n=4) | — (n=0) | — |
| A3 — A2 + Self-RAG reflection | 4 | 73.3 ± 11.0 | 5.5 ± 1.5 | 3.00 | 0.00 | 2.00 | 3.25 | 11.00 | 0.58 (n=4) | — (n=0) | 3.67 |

### By type: multi_eligibility

| Approach | N | Latency (s) | Checks | Pass | Fail | Unkn | Actions | Sources | Keyword Recall | Verdict Acc. | IsSup |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A1 — Profile-conditioned (single-hop) | 9 | 46.4 ± 26.7 | 5.3 ± 1.4 | 2.89 | 0.89 | 1.11 | 3.67 | 10.67 | 0.96 (n=9) | 0.67 (n=9) | — |
| A2 — A1 + Multi-hop decomposition | 8 | 70.7 ± 21.4 | 5.9 ± 1.3 | 3.12 | 1.25 | 1.50 | 3.62 | 9.25 | 0.96 (n=8) | 0.62 (n=8) | — |
| A3 — A2 + Self-RAG reflection | 10 | 84.8 ± 24.9 | 6.8 ± 1.2 | 3.30 | 1.00 | 2.10 | 3.80 | 9.90 | 0.90 (n=10) | 0.50 (n=10) | 3.89 |

### By type: single_eligibility

| Approach | N | Latency (s) | Checks | Pass | Fail | Unkn | Actions | Sources | Keyword Recall | Verdict Acc. | IsSup |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A1 — Profile-conditioned (single-hop) | 9 | 37.8 ± 9.8 | 5.8 ± 1.5 | 3.89 | 0.44 | 1.33 | 3.11 | 8.22 | 0.81 (n=9) | 0.44 (n=9) | — |
| A2 — A1 + Multi-hop decomposition | 11 | 62.6 ± 17.5 | 4.8 ± 1.1 | 3.27 | 0.36 | 1.18 | 3.45 | 9.27 | 0.76 (n=11) | 0.73 (n=11) | — |
| A3 — A2 + Self-RAG reflection | 14 | 89.0 ± 21.2 | 4.3 ± 1.7 | 2.57 | 0.43 | 1.00 | 3.36 | 7.07 | 0.74 (n=14) | 0.43 (n=14) | 2.67 |
