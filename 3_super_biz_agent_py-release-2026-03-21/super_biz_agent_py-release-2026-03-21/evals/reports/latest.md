# Offline Evaluation Report

- Generated at: `2026-07-03T07:29:19.726717+00:00`
- Overall score: **0.8781**
- RAG cases: **15**
- Tool-calling cases: **12**
- AIOps cases: **9**

## Summary

| Area | Metric | Value |
|---|---:|---:|
| RAG | Hit@1 | 0.8667 |
| RAG | Hit@3 | 1.0 |
| RAG | MRR | 0.9222 |
| RAG | Context Precision@3 | 0.3333 |
| RAG | Required Term Recall | 1.0 |
| RAG | Answer Term Coverage | 1.0 |
| Tool Calling | Exact Match | 0.5 |
| Tool Calling | Micro F1 | 0.8571 |
| AIOps | Root Cause Hit@1 | 0.8889 |
| AIOps | Evidence Coverage | 1.0 |
| AIOps | Action Coverage | 0.8889 |
| AIOps | Diagnosis Pass Rate | 0.8889 |

## Notes

- This is a local deterministic baseline, not an LLM-as-judge score.
- Use it to catch regressions and to define the evaluation vocabulary before connecting real Agent outputs.
- Higher-value follow-up: export real `/api/chat_stream` and `/api/aiops` traces, then compare them against the same datasets.
