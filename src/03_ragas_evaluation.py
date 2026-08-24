"""
Bước 3 — RAGAS Evaluation  (có checkpoint/resume)
===================================================
NHIỆM VỤ:
  1. Chạy 20 QA pairs qua CẢ 2 prompt version, lưu answers + contexts
     → Cache vào data/rag_cache_v1.json & rag_cache_v2.json (tự động skip nếu đã có)
  2. Tạo EvaluationDataset với các SingleTurnSample object
  3. Đánh giá với 4 RAGAS metrics: faithfulness, answer_relevancy,
     context_recall, context_precision
     → Lưu checkpoint sau MỖI sample vào data/ragas_ckpt_v1.json & v2.json
     → Nếu bị ngắt, chạy lại sẽ TỰ ĐỘNG resume từ điểm dừng
  4. In bảng so sánh V1 vs V2
  5. Lưu kết quả vào data/ragas_report.json

DELIVERABLE: faithfulness ≥ 0.8 cho ít nhất 1 prompt version
             + file data/ragas_report.json được tạo ra

⏰ LƯU Ý: Bước này mất ~10-15 phút (20 samples × tuần tự).
"""
import sys
import json
import time
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import config  # ⚠️ phải import trước LangChain

import numpy as np
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from ragas import evaluate, EvaluationDataset, SingleTurnSample
from ragas.run_config import RunConfig
from ragas.metrics import faithfulness, answer_relevancy, context_recall, context_precision

from utils.llm_factory import get_llm, get_embeddings
from utils.data_loader import load_knowledge_base, split_text, build_vectorstore
from qa_pairs import QA_PAIRS


# ── Cấu hình ──────────────────────────────────────────────────────────────────
DATA_DIR        = Path(__file__).parent.parent / "data"
RAG_SAMPLE_SIZE = 50   # số câu hỏi RAG mỗi prompt
RAGAS_EVAL_SIZE = 50   # số samples đưa vào RAGAS evaluate
METRIC_NAMES    = ["faithfulness", "answer_relevancy", "context_recall", "context_precision"]
METRICS         = [faithfulness, answer_relevancy, context_recall, context_precision]

# Checkpoint files
RAG_CACHE_FILE  = {"v1": DATA_DIR / "rag_cache_v1.json",
                   "v2": DATA_DIR / "rag_cache_v2.json"}
RAGAS_CKPT_FILE = {"v1": DATA_DIR / "ragas_ckpt_v1.json",
                   "v2": DATA_DIR / "ragas_ckpt_v2.json"}


# ── 1. Prompt Templates ────────────────────────────────────────────────────────
SYSTEM_V1 = (
    "Bạn là trợ lý AI hữu ích. Chỉ dùng context sau để trả lời. "
    "Giữ câu trả lời ngắn gọn (2-4 câu).\n\nContext:\n{context}"
)
PROMPT_V1 = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_V1),
    ("human",  "{question}"),
])

SYSTEM_V2 = (
    "Bạn là một chuyên gia phân tích dữ liệu. Hãy đọc kỹ context, trích xuất "
    "các thông tin chính xác và trả lời một cách rõ ràng, có cấu trúc logic (3-5 câu).\n\nContext:\n{context}"
)
PROMPT_V2 = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_V2),
    ("human",  "{question}"),
])

PROMPTS = {"v1": PROMPT_V1, "v2": PROMPT_V2}


# ── 2. Setup Vectorstore ───────────────────────────────────────────────────────
def setup_vectorstore():
    """Tái sử dụng — tạo FAISS vectorstore từ knowledge base."""
    embeddings  = get_embeddings()
    text        = load_knowledge_base()
    chunks      = split_text(text)
    return build_vectorstore(chunks, embeddings)


# ── 3. RAG Collection với cache ────────────────────────────────────────────────
def run_rag(retriever, llm, prompt, question: str) -> dict:
    """Chạy RAG chain cho 1 câu hỏi."""
    docs     = retriever.invoke(question)
    contexts = [doc.page_content for doc in docs]
    ctx_str  = "\n\n".join(contexts)
    answer   = (prompt | llm | StrOutputParser()).invoke({
        "context":  ctx_str,
        "question": question,
    })
    return {"answer": answer, "contexts": contexts}


def collect_rag_outputs(vectorstore, prompt_version: str) -> list:
    """
    Chạy QA pairs và lưu cache. Nếu cache đã có → load và skip.
    Trả về: list of dict với keys: question, reference, answer, contexts
    """
    cache_file = RAG_CACHE_FILE[prompt_version]

    # ── Resume: load cache nếu đã có ──
    if cache_file.exists():
        print(f"\n✅ Dùng RAG cache [{prompt_version}] từ {cache_file.name} (skip re-collection)")
        return json.loads(cache_file.read_text(encoding="utf-8"))

    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    llm       = get_llm()
    prompt    = PROMPTS[prompt_version]
    qa_subset = QA_PAIRS[:RAG_SAMPLE_SIZE]
    results   = []

    print(f"\n🚀 Đang chạy {RAG_SAMPLE_SIZE} câu hỏi với prompt {prompt_version} ...")
    for i, qa in enumerate(qa_subset, 1):
        out = run_rag(retriever, llm, prompt, qa["question"])
        results.append({
            "question":  qa["question"],
            "reference": qa["reference"],
            "answer":    out["answer"],
            "contexts":  out["contexts"],
        })
        print(f"  [{i:02d}/{RAG_SAMPLE_SIZE}] {qa['question'][:60]}")

    # Lưu cache
    DATA_DIR.mkdir(exist_ok=True)
    cache_file.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"  💾 Đã lưu RAG cache → {cache_file.name}")

    return results


# ── 4. RAGAS Evaluation với checkpoint per-sample ─────────────────────────────
def build_ragas_dataset(rag_results: list) -> EvaluationDataset:
    samples = [
        SingleTurnSample(
            user_input=r["question"],
            response=r["answer"],
            retrieved_contexts=r["contexts"],
            reference=r["reference"],
        )
        for r in rag_results
    ]
    return EvaluationDataset(samples=samples)


def run_ragas_eval_with_checkpoint(rag_results: list, version: str) -> dict:
    """
    Đánh giá từng sample một, lưu checkpoint sau mỗi sample.
    Nếu bị ngắt → chạy lại sẽ resume từ sample chưa đánh giá.
    """
    ckpt_file = RAGAS_CKPT_FILE[version]
    subset    = rag_results[:RAGAS_EVAL_SIZE]
    total     = len(subset)

    # ── Resume: load checkpoint nếu có ──
    if ckpt_file.exists():
        ckpt_data = json.loads(ckpt_file.read_text(encoding="utf-8"))
        scores_per_sample = ckpt_data.get("scores", [])
        start_idx = len(scores_per_sample)
        if start_idx >= total:
            print(f"\n✅ RAGAS [{version}] đã hoàn thành từ checkpoint ({total} samples)")
            return _aggregate(scores_per_sample, version, total)
        print(f"\n📂 Resume RAGAS [{version}] từ checkpoint: {start_idx}/{total} samples đã xong")
    else:
        scores_per_sample = []
        start_idx = 0

    print(f"📐 Đánh giá RAGAS [{version}]: còn {total - start_idx}/{total} samples ...")

    llm_eval = get_llm(temperature=0)
    emb_eval = get_embeddings()
    run_cfg  = RunConfig(max_workers=2, timeout=120)

    for i in range(start_idx, total):
        sample_data = subset[i]
        print(f"  [{i+1:02d}/{total}] Evaluating: {sample_data['question'][:50]} ...", end=" ", flush=True)

        try:
            single_ds = build_ragas_dataset([sample_data])
            result    = evaluate(
                single_ds,
                metrics=METRICS,
                llm=llm_eval,
                embeddings=emb_eval,
                run_config=run_cfg,
            )
            sample_scores = {}
            for k in METRIC_NAMES:
                raw = result[k]
                v   = raw[0] if raw else None
                sample_scores[k] = float(v) if (v is not None and not np.isnan(float(v))) else None
            print(f"✅ faith={sample_scores.get('faithfulness', 'N/A')}")

        except Exception as e:
            print(f"⚠️  lỗi: {type(e).__name__}")
            sample_scores = {k: None for k in METRIC_NAMES}

        scores_per_sample.append(sample_scores)

        # ── Lưu checkpoint ngay sau mỗi sample ──
        DATA_DIR.mkdir(exist_ok=True)
        ckpt_file.write_text(
            json.dumps({"version": version, "total": total, "scores": scores_per_sample},
                       indent=2, ensure_ascii=False),
            encoding="utf-8"
        )

        # Nghỉ nhỏ để tránh rate limit
        time.sleep(1)

    return _aggregate(scores_per_sample, version, total)


def _aggregate(scores_per_sample: list, version: str, total: int) -> dict:
    """Tổng hợp điểm từ danh sách per-sample scores."""
    final_scores = {}
    for k in METRIC_NAMES:
        valid = [
            s[k] for s in scores_per_sample
            if s.get(k) is not None and not np.isnan(s[k])
        ]
        final_scores[k] = float(np.mean(valid)) if valid else 0.0

    n_valid = sum(
        1 for s in scores_per_sample
        if any(s.get(k) is not None for k in METRIC_NAMES)
    )
    print(f"\n📊 Kết quả RAGAS — Prompt {version.upper()} ({n_valid}/{total} samples hợp lệ):")
    for k, v in final_scores.items():
        star = " ⭐" if k == "faithfulness" and v >= 0.8 else ""
        print(f"  {k:30s}: {v:.4f}{star}")

    return final_scores


# ── 5. Main ────────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  Bước 3: RAGAS Evaluation  (checkpoint mode)")
    print("=" * 60)

    if not config.validate():
        sys.exit(1)

    vectorstore = setup_vectorstore()

    v1_results = collect_rag_outputs(vectorstore, "v1")
    v2_results = collect_rag_outputs(vectorstore, "v2")

    v1_scores = run_ragas_eval_with_checkpoint(v1_results, "v1")
    v2_scores = run_ragas_eval_with_checkpoint(v2_results, "v2")

    # ── Bảng so sánh ──
    print("\n" + "=" * 65)
    print(f"  {'Metric':30s}  {'V1':>8}  {'V2':>8}  Winner")
    print("=" * 65)
    for metric in METRIC_NAMES:
        s1, s2 = v1_scores[metric], v2_scores[metric]
        winner = "← V1" if s1 > s2 else "← V2"
        print(f"  {metric:30s}  {s1:>8.4f}  {s2:>8.4f}  {winner}")

    best_faith = max(v1_scores["faithfulness"], v2_scores["faithfulness"])
    if best_faith >= 0.8:
        print(f"\n✅ Đạt mục tiêu: faithfulness = {best_faith:.4f} ≥ 0.8")
    else:
        print(f"\n⚠️  Chưa đạt mục tiêu ({best_faith:.4f} < 0.8).")
        print("   Gợi ý: giảm chunk_size, tăng k, hoặc điều chỉnh prompt.")

    # ── Lưu report ──
    report = {
        "prompt_v1_scores": v1_scores,
        "prompt_v2_scores": v2_scores,
        "target_met": best_faith >= 0.8,
        "samples_evaluated": RAGAS_EVAL_SIZE,
    }
    report_path = DATA_DIR / "ragas_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\n💾 Đã lưu báo cáo vào {report_path}")
    print("💡 Tip: Xóa data/ragas_ckpt_v*.json để chạy lại từ đầu.")


if __name__ == "__main__":
    main()
