"""Offline evaluation suite for the Smart OnCall Agent project.

The evaluator is intentionally dependency-free. It provides a reproducible
baseline for three interview-relevant surfaces:

1. RAG retrieval over local aiops-docs.
2. Agent tool-selection expectations.
3. AIOps diagnosis coverage over curated incident cases.

It does not call Qwen, Milvus, MCP servers, or external APIs. That makes it safe
to run in CI and on a laptop before the real services are available.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


EVAL_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = EVAL_ROOT.parent
DOCS_DIR = PROJECT_ROOT / "aiops-docs"
DATASET_DIR = EVAL_ROOT / "datasets"
REPORT_DIR = EVAL_ROOT / "reports"


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def normalize_text(text: str) -> str:
    return text.lower().replace("％", "%")


def contains_term(text: str, term: str) -> bool:
    return normalize_text(term) in normalize_text(text)


def safe_div(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def round4(value: float) -> float:
    return round(value, 4)


def tokenize(text: str) -> Counter[str]:
    """Tokenize Chinese/English incident text without external segmenters.

    Chinese is represented by 2-4 character ngrams; English and alert IDs are
    represented as lowercase words. This is not meant to replace embeddings, but
    it is good enough to catch obvious regressions in the local knowledge base.
    """

    text = normalize_text(text)
    tokens: list[str] = []

    tokens.extend(re.findall(r"[a-z0-9_]+%?", text))
    tokens.extend(re.findall(r"\d+(?:\.\d+)?%?", text))

    for sequence in re.findall(r"[\u4e00-\u9fff]+", text):
        for ngram_size in (2, 3, 4):
            if len(sequence) < ngram_size:
                continue
            tokens.extend(sequence[i : i + ngram_size] for i in range(len(sequence) - ngram_size + 1))

    return Counter(tokens)


@dataclass
class KnowledgeDoc:
    source: str
    text: str
    tokens: Counter[str]


def load_docs() -> list[KnowledgeDoc]:
    docs: list[KnowledgeDoc] = []
    for path in sorted(DOCS_DIR.glob("*.md")):
        text = path.read_text(encoding="utf-8")
        docs.append(KnowledgeDoc(source=path.name, text=text, tokens=tokenize(path.name + "\n" + text)))
    if not docs:
        raise RuntimeError(f"No Markdown docs found under {DOCS_DIR}")
    return docs


def build_idf(docs: list[KnowledgeDoc]) -> dict[str, float]:
    doc_count = len(docs)
    df: Counter[str] = Counter()
    for doc in docs:
        df.update(doc.tokens.keys())
    return {token: math.log((doc_count + 1) / (count + 0.5)) + 1 for token, count in df.items()}


def retrieve_docs(query: str, docs: list[KnowledgeDoc], idf: dict[str, float], top_k: int) -> list[dict[str, Any]]:
    query_tokens = tokenize(query)
    scored: list[dict[str, Any]] = []

    for doc in docs:
        score = 0.0
        overlap_terms: list[str] = []
        for token, query_count in query_tokens.items():
            doc_count = doc.tokens.get(token, 0)
            if not doc_count:
                continue
            overlap_terms.append(token)
            score += (1 + math.log(query_count + 1)) * (1 + math.log(doc_count + 1)) * idf.get(token, 1.0)

        # Small phrase bonus keeps alert names and explicit filenames stable.
        doc_text = normalize_text(doc.source + "\n" + doc.text)
        for phrase in re.findall(r"[a-z][a-z0-9_]+", normalize_text(query)):
            if phrase in doc_text:
                score += 2.5

        scored.append(
            {
                "source": doc.source,
                "score": round4(score),
                "overlap_terms": sorted(overlap_terms, key=len, reverse=True)[:12],
                "text": doc.text,
            }
        )

    return sorted(scored, key=lambda item: item["score"], reverse=True)[:top_k]


def evaluate_rag(top_k: int) -> dict[str, Any]:
    cases = load_json(DATASET_DIR / "rag_cases.json")
    docs = load_docs()
    idf = build_idf(docs)

    rows: list[dict[str, Any]] = []
    hit_at_1 = 0
    hit_at_k = 0
    reciprocal_ranks: list[float] = []
    context_precisions: list[float] = []
    context_recalls: list[float] = []
    answer_term_coverages: list[float] = []

    for case in cases:
        retrieved = retrieve_docs(case["question"], docs, idf, top_k=top_k)
        retrieved_sources = [item["source"] for item in retrieved]
        expected_sources = set(case["expected_sources"])

        first_relevant_rank = None
        for index, source in enumerate(retrieved_sources, start=1):
            if source in expected_sources:
                first_relevant_rank = index
                break

        hit1 = retrieved_sources[0] in expected_sources if retrieved_sources else False
        hitk = first_relevant_rank is not None
        precision = safe_div(sum(1 for source in retrieved_sources if source in expected_sources), len(retrieved_sources))

        retrieved_context = "\n".join(item["text"] for item in retrieved)
        required_hits = [term for term in case["required_terms"] if contains_term(retrieved_context, term)]
        answer_hits = [term for term in case["answer_terms"] if contains_term(retrieved_context, term)]

        required_recall = safe_div(len(required_hits), len(case["required_terms"]))
        answer_coverage = safe_div(len(answer_hits), len(case["answer_terms"]))

        hit_at_1 += int(hit1)
        hit_at_k += int(hitk)
        reciprocal_ranks.append(safe_div(1, first_relevant_rank or 0))
        context_precisions.append(precision)
        context_recalls.append(required_recall)
        answer_term_coverages.append(answer_coverage)

        rows.append(
            {
                "id": case["id"],
                "question": case["question"],
                "expected_sources": case["expected_sources"],
                "retrieved_sources": retrieved_sources,
                "hit_at_1": hit1,
                f"hit_at_{top_k}": hitk,
                "rank": first_relevant_rank,
                f"context_precision_at_{top_k}": round4(precision),
                "required_term_recall": round4(required_recall),
                "answer_term_coverage": round4(answer_coverage),
                "missing_required_terms": [term for term in case["required_terms"] if term not in required_hits],
                "top_scores": [{k: v for k, v in item.items() if k != "text"} for item in retrieved],
            }
        )

    count = len(cases)
    return {
        "summary": {
            "cases": count,
            "hit_at_1": round4(hit_at_1 / count),
            f"hit_at_{top_k}": round4(hit_at_k / count),
            "mrr": round4(sum(reciprocal_ranks) / count),
            f"context_precision_at_{top_k}": round4(sum(context_precisions) / count),
            "required_term_recall": round4(sum(context_recalls) / count),
            "answer_term_coverage": round4(sum(answer_term_coverages) / count),
        },
        "cases": rows,
    }


def predict_tools(prompt: str) -> list[str]:
    text = normalize_text(prompt)
    tools: set[str] = set()

    if any(keyword in text for keyword in ["现在", "当前", "最近", "分钟", "小时", "时间窗口", "告警时间"]):
        tools.add("get_current_time")

    if any(keyword in text for keyword in ["知识库", "处理方案", "建议", "怎么", "如何", "排查", "判断", "告警"]):
        tools.add("retrieve_knowledge")

    if any(keyword in text for keyword in ["cpu", "负载", "使用率超过 80", "高cpu"]):
        tools.add("query_cpu_metrics")

    if any(keyword in text for keyword in ["内存", "memory", "oom", "gc"]):
        tools.add("query_memory_metrics")

    log_keywords = [
        "日志",
        "error",
        "错误",
        "fatal",
        "慢查询",
        "健康检查",
        "数据库连接",
        "服务不可用",
        "serviceunavailable",
        "磁盘",
    ]
    if any(keyword in text for keyword in log_keywords):
        tools.add("search_topic_by_service_name")
        tools.add("search_log")

    return sorted(tools)


def evaluate_tools() -> dict[str, Any]:
    cases = load_json(DATASET_DIR / "tool_cases.json")

    rows: list[dict[str, Any]] = []
    exact_matches = 0
    total_tp = 0
    total_fp = 0
    total_fn = 0

    for case in cases:
        expected = set(case["expected_tools"])
        predicted = set(predict_tools(case["prompt"]))
        tp = len(expected & predicted)
        fp = len(predicted - expected)
        fn = len(expected - predicted)

        exact = expected == predicted
        exact_matches += int(exact)
        total_tp += tp
        total_fp += fp
        total_fn += fn

        rows.append(
            {
                "id": case["id"],
                "prompt": case["prompt"],
                "expected_tools": sorted(expected),
                "predicted_tools": sorted(predicted),
                "exact_match": exact,
                "precision": round4(safe_div(tp, tp + fp)),
                "recall": round4(safe_div(tp, tp + fn)),
                "f1": round4(safe_div(2 * tp, 2 * tp + fp + fn)),
                "missing_tools": sorted(expected - predicted),
                "extra_tools": sorted(predicted - expected),
            }
        )

    precision = safe_div(total_tp, total_tp + total_fp)
    recall = safe_div(total_tp, total_tp + total_fn)
    f1 = safe_div(2 * total_tp, 2 * total_tp + total_fp + total_fn)

    return {
        "summary": {
            "cases": len(cases),
            "exact_match": round4(exact_matches / len(cases)),
            "micro_precision": round4(precision),
            "micro_recall": round4(recall),
            "micro_f1": round4(f1),
        },
        "cases": rows,
    }


ROOT_CAUSE_RULES = [
    ("cpu_spin_loop", ["重复错误堆栈", "单个进程", "占满 cpu", "pid-12345"]),
    ("traffic_spike", ["请求量", "增长", "多个实例", "无明显 error"]),
    ("memory_leak", ["full gc", "无法释放", "持续缓慢上升", "gc 记录"]),
    ("large_file_memory_pressure", ["大文件", "oom", "一次性加载"]),
    ("log_bloat", ["/var/log", "debug", "日志持续增长"]),
    ("db_connection_failure", ["too many connections", "connection refused", "数据库连接"]),
    ("database_slow_query", ["慢查询", "query_time", "全表扫描", "p99"]),
    ("cache_miss_or_penetration", ["缓存命中率", "数据库查询量", "35%"]),
]

ACTION_RULES = {
    "cpu_spin_loop": ["restart_instance", "rollback_release", "inspect_error_stack"],
    "traffic_spike": ["scale_out", "enable_rate_limit", "watch_cpu"],
    "memory_leak": ["dump_heap", "restart_instance", "analyze_heap"],
    "large_file_memory_pressure": ["streaming_processing", "limit_file_size", "batch_processing"],
    "log_bloat": ["clear_large_logs", "enable_logrotate", "lower_log_level"],
    "db_connection_failure": ["check_database", "restart_app_pool", "tune_connection_pool"],
    "database_slow_query": ["explain_sql", "add_index", "optimize_sql"],
    "cache_miss_or_penetration": ["warm_cache", "cache_null_values", "add_bloom_filter"],
}


def diagnose_case(case: dict[str, Any]) -> dict[str, Any]:
    text = normalize_text(" ".join([case["alert_name"], case["service_name"], case["input"]]))
    causes: list[str] = []
    for cause, keywords in ROOT_CAUSE_RULES:
        if sum(1 for keyword in keywords if keyword in text) >= 2:
            causes.append(cause)

    if not causes:
        alert = normalize_text(case["alert_name"])
        if "highcpuusage" in alert:
            causes.append("traffic_spike")
        elif "highmemoryusage" in alert:
            causes.append("memory_leak")
        elif "highdiskusage" in alert:
            causes.append("log_bloat")
        elif "serviceunavailable" in alert:
            causes.append("db_connection_failure")
        elif "slowresponse" in alert:
            causes.append("database_slow_query")

    actions: list[str] = []
    for cause in causes:
        actions.extend(ACTION_RULES.get(cause, []))

    evidence_hits = [term for term in case["expected_evidence_terms"] if contains_term(case["input"], term)]

    return {
        "predicted_root_causes": causes,
        "predicted_actions": sorted(set(actions)),
        "evidence_hits": evidence_hits,
    }


def evaluate_aiops() -> dict[str, Any]:
    cases = load_json(DATASET_DIR / "aiops_cases.json")

    rows: list[dict[str, Any]] = []
    root_hits = 0
    evidence_coverages: list[float] = []
    action_coverages: list[float] = []
    pass_count = 0

    for case in cases:
        predicted = diagnose_case(case)
        expected_causes = set(case["expected_root_causes"])
        predicted_causes = predicted["predicted_root_causes"]
        predicted_actions = set(predicted["predicted_actions"])

        root_cause_hit = bool(predicted_causes and predicted_causes[0] in expected_causes)
        evidence_coverage = safe_div(len(predicted["evidence_hits"]), len(case["expected_evidence_terms"]))
        action_coverage = safe_div(
            len(set(case["expected_actions"]) & predicted_actions),
            len(case["expected_actions"]),
        )
        passed = root_cause_hit and evidence_coverage >= 0.67 and action_coverage >= 0.67

        root_hits += int(root_cause_hit)
        evidence_coverages.append(evidence_coverage)
        action_coverages.append(action_coverage)
        pass_count += int(passed)

        rows.append(
            {
                "id": case["id"],
                "alert_name": case["alert_name"],
                "service_name": case["service_name"],
                "expected_root_causes": case["expected_root_causes"],
                "predicted_root_causes": predicted_causes,
                "root_cause_hit_at_1": root_cause_hit,
                "evidence_coverage": round4(evidence_coverage),
                "action_coverage": round4(action_coverage),
                "passed": passed,
                "missing_evidence_terms": [
                    term for term in case["expected_evidence_terms"] if term not in predicted["evidence_hits"]
                ],
                "missing_actions": sorted(set(case["expected_actions"]) - predicted_actions),
            }
        )

    count = len(cases)
    return {
        "summary": {
            "cases": count,
            "root_cause_hit_at_1": round4(root_hits / count),
            "mean_evidence_coverage": round4(sum(evidence_coverages) / count),
            "mean_action_coverage": round4(sum(action_coverages) / count),
            "diagnosis_pass_rate": round4(pass_count / count),
        },
        "cases": rows,
    }


def weighted_overall_score(rag: dict[str, Any], tools: dict[str, Any], aiops: dict[str, Any], top_k: int) -> float:
    rag_summary = rag["summary"]
    tool_summary = tools["summary"]
    aiops_summary = aiops["summary"]

    rag_score = (
        0.35 * rag_summary["hit_at_1"]
        + 0.25 * rag_summary[f"hit_at_{top_k}"]
        + 0.2 * rag_summary["required_term_recall"]
        + 0.2 * rag_summary["answer_term_coverage"]
    )
    tool_score = 0.45 * tool_summary["exact_match"] + 0.55 * tool_summary["micro_f1"]
    aiops_score = (
        0.4 * aiops_summary["root_cause_hit_at_1"]
        + 0.25 * aiops_summary["mean_evidence_coverage"]
        + 0.25 * aiops_summary["mean_action_coverage"]
        + 0.1 * aiops_summary["diagnosis_pass_rate"]
    )

    return round4(0.45 * rag_score + 0.25 * tool_score + 0.30 * aiops_score)


def render_markdown_report(report: dict[str, Any]) -> str:
    top_k = report["config"]["top_k"]
    rag = report["rag"]["summary"]
    tools = report["tool_calling"]["summary"]
    aiops = report["aiops_diagnosis"]["summary"]

    lines = [
        "# Offline Evaluation Report",
        "",
        f"- Generated at: `{report['generated_at']}`",
        f"- Overall score: **{report['overall_score']}**",
        f"- RAG cases: **{rag['cases']}**",
        f"- Tool-calling cases: **{tools['cases']}**",
        f"- AIOps cases: **{aiops['cases']}**",
        "",
        "## Summary",
        "",
        "| Area | Metric | Value |",
        "|---|---:|---:|",
        f"| RAG | Hit@1 | {rag['hit_at_1']} |",
        f"| RAG | Hit@{top_k} | {rag[f'hit_at_{top_k}']} |",
        f"| RAG | MRR | {rag['mrr']} |",
        f"| RAG | Context Precision@{top_k} | {rag[f'context_precision_at_{top_k}']} |",
        f"| RAG | Required Term Recall | {rag['required_term_recall']} |",
        f"| RAG | Answer Term Coverage | {rag['answer_term_coverage']} |",
        f"| Tool Calling | Exact Match | {tools['exact_match']} |",
        f"| Tool Calling | Micro F1 | {tools['micro_f1']} |",
        f"| AIOps | Root Cause Hit@1 | {aiops['root_cause_hit_at_1']} |",
        f"| AIOps | Evidence Coverage | {aiops['mean_evidence_coverage']} |",
        f"| AIOps | Action Coverage | {aiops['mean_action_coverage']} |",
        f"| AIOps | Diagnosis Pass Rate | {aiops['diagnosis_pass_rate']} |",
        "",
        "## Notes",
        "",
        "- This is a local deterministic baseline, not an LLM-as-judge score.",
        "- Use it to catch regressions and to define the evaluation vocabulary before connecting real Agent outputs.",
        "- Higher-value follow-up: export real `/api/chat_stream` and `/api/aiops` traces, then compare them against the same datasets.",
        "",
    ]
    return "\n".join(lines)


def run(top_k: int, write_report: bool) -> dict[str, Any]:
    rag = evaluate_rag(top_k=top_k)
    tools = evaluate_tools()
    aiops = evaluate_aiops()

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "config": {
            "top_k": top_k,
            "docs_dir": str(DOCS_DIR.relative_to(PROJECT_ROOT)),
            "datasets_dir": str(DATASET_DIR.relative_to(PROJECT_ROOT)),
        },
        "overall_score": weighted_overall_score(rag, tools, aiops, top_k=top_k),
        "rag": rag,
        "tool_calling": tools,
        "aiops_diagnosis": aiops,
    }

    if write_report:
        REPORT_DIR.mkdir(parents=True, exist_ok=True)
        write_json(REPORT_DIR / "latest.json", report)
        (REPORT_DIR / "latest.md").write_text(render_markdown_report(report), encoding="utf-8")

    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Run offline evals for the Smart OnCall Agent project.")
    parser.add_argument("--top-k", type=int, default=3, help="Number of retrieved docs for RAG metrics.")
    parser.add_argument("--write-report", action="store_true", help="Write evals/reports/latest.json and latest.md.")
    args = parser.parse_args()

    report = run(top_k=args.top_k, write_report=args.write_report)
    print(json.dumps(report["rag"]["summary"], ensure_ascii=False, indent=2))
    print(json.dumps(report["tool_calling"]["summary"], ensure_ascii=False, indent=2))
    print(json.dumps(report["aiops_diagnosis"]["summary"], ensure_ascii=False, indent=2))
    print(json.dumps({"overall_score": report["overall_score"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
