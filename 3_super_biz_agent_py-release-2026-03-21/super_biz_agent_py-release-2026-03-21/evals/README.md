# Evaluation Suite

This directory contains a lightweight evaluation baseline for the Smart OnCall
Agent project. It is designed to run without Qwen, Milvus, MCP servers, Docker,
or network access.

## What It Evaluates

1. **RAG retrieval**
   - `Hit@1`
   - `Hit@K`
   - `MRR`
   - `Context Precision@K`
   - `Required Term Recall`
   - `Answer Term Coverage`

2. **Agent tool calling**
   - Exact tool-set match
   - Micro precision / recall / F1
   - Missing and extra tool calls per case

3. **AIOps diagnosis**
   - Root cause hit@1
   - Evidence coverage
   - Action coverage
   - Diagnosis pass rate

## Run

From the Python project root:

```powershell
.venv\Scripts\python.exe evals\run_offline_evals.py --write-report
```

If you do not want to use the local virtual environment:

```powershell
python evals\run_offline_evals.py --write-report
```

Reports are written to:

- `evals/reports/latest.json`
- `evals/reports/latest.md`

## Dataset Files

- `datasets/rag_cases.json`: question-to-source expectations for the local
  `aiops-docs` knowledge base.
- `datasets/tool_cases.json`: expected tool choices for representative Agent
  prompts.
- `datasets/aiops_cases.json`: incident cases with expected root causes,
  evidence terms, and remediation actions.

## How To Use In Interviews

The key point is not that this offline baseline is perfect. The key point is
that the project has a measurable quality loop:

- Change chunk size, `top_k`, prompt wording, or tool descriptions.
- Run the same evaluation.
- Compare `latest.json` metrics before and after.

For the real Agent, keep these datasets and replace the deterministic baseline
with exported traces from `/api/chat_stream` and `/api/aiops`.
