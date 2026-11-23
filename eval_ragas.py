"""
RAGAS evaluation for SETU Compliance RAG.

Metrics (all LLM-based, no string matching):
  - Faithfulness       : claims in the answer are grounded in retrieved context
  - Answer Relevancy   : answer directly addresses the question asked
  - Context Recall     : retrieved context covers the ground-truth answer
  - Context Precision  : most relevant chunks are ranked highest

Ragas uses our own Groq LLM + Jina embeddings — no OpenAI API key needed.

Run:  python eval_ragas.py
Results saved to: eval_ragas_results.json
"""
import json
import sys
import time
from dotenv import load_dotenv

load_dotenv()

# ── Reference answers for context_recall / context_precision ─────────────────

GROUND_TRUTH = {
    "E1": "SETU staff are entitled to 26 weeks of maternity leave at full pay under the Leave Management Policy.",
    "E2": "SETU uses legitimate interest as a lawful basis for processing personal data under GDPR, as stated in the Data Protection Policy.",
    "E3": "Garda vetting is required at SETU for all roles involving contact with children or vulnerable adults. All staff in qualifying positions must comply.",
    "E4": "Staff must follow SETU's responsible use guidelines and disclose the use of Gen AI tools when producing official work, as per the Gen AI Guidelines.",
    "E5": "SETU is committed to equality, diversity, and inclusion, ensuring all staff are treated fairly and without discrimination under the EDI Policy.",
    "M1": "SETU staff with caring responsibilities can access flexible working arrangements and carer's leave as set out in the Caring Responsibilities Policy.",
    "M2": "Research misconduct allegations are handled through formal stages: initial assessment, formal investigation, and an appeal process.",
    "M3": "Intellectual property created by staff in the course of their employment belongs to SETU, as defined in the Intellectual Property Policy.",
    "M4": "Breaches of the Data Protection Policy may result in disciplinary action up to and including dismissal for serious breaches.",
    "M5": "The Garda Vetting Policy applies to all SETU staff and volunteers in roles involving access to children or vulnerable persons.",
    "H1": "Both the Equality Policy and the Gender Identity Policy require SETU to respect and support staff requesting name changes related to gender identity.",
    "H2": "Academic misconduct at SETU includes plagiarism, fabrication of data, and dishonest conduct as defined in the Academic Integrity Policy.",
    "H3": "SETU may share employee personal data with a third-party recruitment agency only when there is a lawful basis and appropriate data protection safeguards are in place.",
    "H4": "Harassment under SETU's Dignity and Respect Policy includes any conduct that violates a person's dignity or creates an intimidating, hostile, or offensive environment.",
    "H5": "Staff using AI to write research papers must disclose this use and adhere to SETU's Gen AI Guidelines, which emphasise responsible and transparent AI use.",
}


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    from information_agent import load_vectorstore, get_information_with_context
    from config.llm_config import get_llm, get_embeddings
    from eval_metrics import GROUND_TRUTH as EVAL_QUERIES

    print("#" * 70)
    print("# SETU Compliance RAG — RAGAS Evaluation")
    print(f"# {len(EVAL_QUERIES)} queries  |  faithfulness, relevancy, recall, precision")
    print("#" * 70)

    vs = load_vectorstore()

    # ── Configure Ragas to use our LLM/embeddings ─────────────────────────────
    try:
        from ragas.llms import LangchainLLMWrapper
        from ragas.embeddings import LangchainEmbeddingsWrapper
        from ragas import evaluate, EvaluationDataset
        from ragas.dataset_schema import SingleTurnSample
        from ragas.metrics import Faithfulness, AnswerRelevancy, LLMContextRecall, LLMContextPrecisionWithReference

        llm_wrapper = LangchainLLMWrapper(get_llm(temperature=0))
        emb_wrapper = LangchainEmbeddingsWrapper(get_embeddings())

        metrics = [
            Faithfulness(llm=llm_wrapper),
            AnswerRelevancy(llm=llm_wrapper, embeddings=emb_wrapper),
            LLMContextRecall(llm=llm_wrapper),
            LLMContextPrecisionWithReference(llm=llm_wrapper),
        ]
        use_v2 = True
        print("[OK] Ragas 0.2.x API loaded\n")
    except ImportError:
        from ragas.llms import LangchainLLMWrapper
        from ragas.embeddings import LangchainEmbeddingsWrapper
        from ragas import evaluate
        from ragas.metrics import faithfulness, answer_relevancy, context_recall, context_precision
        from datasets import Dataset

        llm_wrapper = LangchainLLMWrapper(get_llm(temperature=0))
        emb_wrapper = LangchainEmbeddingsWrapper(get_embeddings())
        faithfulness.llm = llm_wrapper
        answer_relevancy.llm = llm_wrapper
        answer_relevancy.embeddings = emb_wrapper
        context_recall.llm = llm_wrapper
        context_precision.llm = llm_wrapper
        metrics = [faithfulness, answer_relevancy, context_recall, context_precision]
        use_v2 = False
        print("[OK] Ragas 0.1.x API loaded\n")

    # ── Collect answers + contexts ────────────────────────────────────────────
    rows = []
    for entry in EVAL_QUERIES:
        qid   = entry["id"]
        query = entry["query"]
        ref   = GROUND_TRUTH.get(qid, "")
        print(f"[{qid}] ({entry['difficulty']}) {query[:70]}")

        t0 = time.perf_counter()
        try:
            answer, contexts = get_information_with_context(vs, query)
        except Exception as exc:
            print(f"  ERROR: {exc}")
            answer, contexts = f"ERROR: {exc}", []
        latency_ms = round((time.perf_counter() - t0) * 1000)

        print(f"  {latency_ms}ms | {len(contexts)} contexts | {answer[:80].replace(chr(10),' ')}...")
        rows.append({
            "id": qid,
            "difficulty": entry["difficulty"],
            "query": query,
            "answer": answer,
            "contexts": contexts,
            "ground_truth": ref,
            "latency_ms": latency_ms,
        })
        time.sleep(3)  # Groq TPM guard

    # ── Build Ragas dataset ───────────────────────────────────────────────────
    print("\nBuilding Ragas dataset and running metrics (this takes a few minutes)...")

    if use_v2:
        samples = [
            SingleTurnSample(
                user_input=r["query"],
                response=r["answer"],
                retrieved_contexts=r["contexts"],
                reference=r["ground_truth"],
            )
            for r in rows
        ]
        dataset = EvaluationDataset(samples=samples)
    else:
        dataset = Dataset.from_dict({
            "question":    [r["query"]        for r in rows],
            "answer":      [r["answer"]       for r in rows],
            "contexts":    [r["contexts"]     for r in rows],
            "ground_truth":[r["ground_truth"] for r in rows],
        })

    result = evaluate(dataset, metrics=metrics)

    # ── Print results ─────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("RAGAS AGGREGATE METRICS")
    print("=" * 70)
    try:
        agg = {
            "faithfulness":      round(float(result["faithfulness"]),      3),
            "answer_relevancy":  round(float(result["answer_relevancy"]),  3),
            "context_recall":    round(float(result["context_recall"] if "context_recall" in result
                                             else result["llm_context_recall"]), 3),
            "context_precision": round(float(result["context_precision"] if "context_precision" in result
                                             else result["llm_context_precision_with_reference"]), 3),
        }
    except Exception:
        agg = dict(result)

    for k, v in agg.items():
        bar = "█" * int(v * 20) + "░" * (20 - int(v * 20))
        print(f"  {k:<28} {bar}  {v:.3f}")

    # Per-difficulty breakdown
    df = result.to_pandas() if hasattr(result, "to_pandas") else None
    if df is not None:
        df["difficulty"] = [r["difficulty"] for r in rows]
        df["id"]         = [r["id"]         for r in rows]
        df["latency_ms"] = [r["latency_ms"] for r in rows]
        print()
        for label, diff in [("Easy (E1-E5)", "easy"), ("Medium (M1-M5)", "medium"), ("Hard (H1-H5)", "hard")]:
            sub = df[df["difficulty"] == diff]
            if not sub.empty:
                print(f"\n{label}:")
                for col in [c for c in agg if c in sub.columns]:
                    print(f"  {col:<28} {sub[col].mean():.3f}")
                print(f"  {'avg_latency_ms':<28} {sub['latency_ms'].mean():.0f}ms")

    # ── Save results ──────────────────────────────────────────────────────────
    per_query = []
    if df is not None:
        per_query = df.to_dict(orient="records")
    else:
        per_query = rows

    out_path = "eval_ragas_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"aggregate": agg, "per_query": per_query}, f, ensure_ascii=False, indent=2)
    print(f"\nResults saved → {out_path}")

    # ── Side-by-side with custom eval ─────────────────────────────────────────
    try:
        with open("eval_results.json") as f:
            prev = json.load(f)["aggregate"]
        print("\n── Comparison with custom eval ────────────────────────────────")
        print(f"  Custom  answer_recall    : {prev.get('answer_recall', '?')}")
        print(f"  Custom  citation_accuracy: {prev.get('citation_accuracy', '?')}")
        print(f"  Ragas   faithfulness     : {agg.get('faithfulness', '?')}")
        print(f"  Ragas   context_recall   : {agg.get('context_recall', '?')}")
    except Exception:
        pass


if __name__ == "__main__":
    main()
