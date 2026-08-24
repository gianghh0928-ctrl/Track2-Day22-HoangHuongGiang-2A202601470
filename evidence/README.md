# Day 22 Evaluation Report & Evidence Overview

## Project Summary
- **Project Name:** Day 22 - LangSmith, Prompt Hub, RAGAS Evaluation & Guardrails AI
- **LangSmith Project:** `day22-lab`
- **Evaluated Models:** GPT-4o-mini / Gemini-3.1-Flash-Lite

---

## 📊 RAGAS Evaluation Analysis (Prompt V1 vs Prompt V2)

### Comparison Table
| Metric | Prompt V1 | Prompt V2 | Winner |
| :--- | :---: | :---: | :---: |
| **Faithfulness** | **0.9681** | 0.9018 | **← Prompt V1** |
| **Answer Relevancy** | **0.8892** | 0.8722 | **← Prompt V1** |
| **Context Recall** | **1.0000** | **1.0000** | **Tie** |
| **Context Precision** | **0.9467** | 0.9400 | **← Prompt V1** |

### Key Findings & Analysis
1. **Faithfulness Performance (Target ≥ 0.8):**
   - Both prompt versions successfully passed the deliverable requirement (`faithfulness ≥ 0.8`).
   - **Prompt V1 achieved an outstanding 0.9681**, outperforming Prompt V2 (0.9018).
   
2. **Why Prompt V1 Performed Better:**
   - **Prompt V1** explicitly instructed the model: *"Chỉ dùng context sau để trả lời. Giữ câu trả lời ngắn gọn (2-4 câu)"*. This strictly constrained the model from introducing external hallucinations or verbose elaborations.
   - **Prompt V2** instructed the model to act as an data analysis expert and format answers logically in 3-5 sentences. While clear, the longer structure occasionally led to slight paraphrasing that marginally reduced strict faithfulness alignment against the raw context chunks.

3. **Retrieval Metrics (Context Recall & Precision):**
   - Both prompts used the same FAISS vector retriever ($k=3$), resulting in identical **Context Recall (1.0000)** and nearly identical **Context Precision (~0.94)**.

---

## 📁 Evidence Files Summary
- `01_langsmith_traces.png`: Screenshot of LangSmith Dashboard with 100+ traced executions.
- `02_prompt_hub.png`: Screenshot of LangSmith Prompt Hub displaying published prompt versions.
- `02_ab_routing_log.txt`: Console log demonstrating deterministic A/B routing execution across 50 questions.
- `03_ragas_scores.png`: Screenshot of RAGAS evaluation terminal output.
- `03_ragas_report.json`: JSON output containing raw scores for V1 & V2.
- `04_pii_demo_log.txt`: Log showing PII detection and redaction (FIX action).
- `04_json_demo_log.txt`: Log showing automated JSON schema validation and error repair.
