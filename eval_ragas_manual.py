"""
RAGAS-equivalent evaluation using the project's own LLM — no ragas library needed.

Implements the same four LLM-based metrics as RAGAS:
  - Faithfulness       : fraction of answer claims supported by retrieved context
  - Answer Relevancy   : how directly the answer addresses the question (0-1)
  - Context Recall     : fraction of ground-truth facts covered by retrieved context
  - Context Precision  : fraction of retrieved chunks that are relevant

Each metric uses the same Groq LLM already powering the chatbot.

Run:  python eval_ragas_manual.py
Results saved to: eval_ragas_results.json
"""
import json
import re
import sys
import time
from dotenv import load_dotenv

load_dotenv()

# ── Reference answers (ground truth) ─────────────────────────────────────────

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


# ── Metric implementations ────────────────────────────────────────────────────

def _ask_01(llm, prompt: str) -> float:
    """Ask LLM a yes/no question; return 1.0 for yes, 0.0 for no."""
    try:
        r = llm.invoke(prompt).content.strip().lower()
        if r.startswith("1") or r.startswith("yes"):
            return 1.0
        return 0.0
    except Exception:
        return 0.0


def faithfulness(question: str, answer: str, contexts: list[str], llm) -> float:
    """Fraction of answer sentences that are supported by the retrieved context.

    RAGAS faithfulness: decompose the answer into atomic claims, then for
    each claim ask whether it can be inferred from the context.
    """
    sentences = [s.strip() for s in re.split(r"[.!?]", answer) if len(s.strip()) > 15]
    if not sentences:
        return 1.0
    ctx_text = "\n---\n".join(contexts[:5])[:2000]
    scores = []
    for sent in sentences:
        prompt = (
            f"Context:\n{ctx_text}\n\n"
            f"Claim: {sent}\n\n"
            "Is this claim fully supported by the context above? "
            "Reply with only 1 (yes) or 0 (no)."
        )
        scores.append(_ask_01(llm, prompt))
        time.sleep(0.5)
    return sum(scores) / len(scores) if scores else 1.0


def answer_relevancy(question: str, answer: str, llm) -> float:
    """How directly and completely the answer addresses the question.

    RAGAS approach: ask the LLM to score relevance and also to regenerate
    the question from the answer; similarity of regenerated vs original = score.
    We use a simpler direct scoring for speed.
    """
    prompt = (
        f"Question: {question}\n\n"
        f"Answer: {answer}\n\n"
        "Score how directly and completely the answer addresses the question. "
        "Penalise answers that are off-topic or refuse to answer. "
        "Return only a decimal between 0.0 (irrelevant) and 1.0 (fully relevant)."
    )
    try:
        r = llm.invoke(prompt).content.strip()
        m = re.search(r"0?\.\d+|1\.0|1", r)
        return min(1.0, max(0.0, float(m.group()))) if m else 0.5
    except Exception:
        return 0.5


def context_recall(question: str, ground_truth: str, contexts: list[str], llm) -> float:
    """Fraction of ground-truth sentences that are covered by the retrieved context.

    RAGAS: for each sentence in the reference answer, check whether it can
    be attributed to the retrieved context.
    """
    if not ground_truth.strip():
        return 1.0
    sentences = [s.strip() for s in re.split(r"[.!?]", ground_truth) if len(s.strip()) > 10]
    if not sentences:
        return 1.0
    ctx_text = "\n---\n".join(contexts[:5])[:2000]
    scores = []
    for sent in sentences:
        prompt = (
            f"Context:\n{ctx_text}\n\n"
            f"Statement: {sent}\n\n"
            "Can this statement be inferred from the context above? "
            "Reply with only 1 (yes) or 0 (no)."
        )
        scores.append(_ask_01(llm, prompt))
        time.sleep(0.5)
    return sum(scores) / len(scores) if scores else 1.0


def context_precision(question: str, contexts: list[str], llm) -> float:
    """Fraction of retrieved chunks that are relevant to answering the question.

    RAGAS: for each chunk, ask if it contains information useful for the question.
    Weighted by rank (earlier chunks count more).
    """
    if not contexts:
        return 0.0
    scores = []
    for i, ctx in enumerate(contexts[:5]):
        prompt = (
            f"Question: {question}\n\n"
            f"Passage:\n{ctx[:400]}\n\n"
            "Does this passage contain information that is useful for answering the question? "
            "Reply with only 1 (yes) or 0 (no)."
        )
        score = _ask_01(llm, prompt)
        scores.append(score)
        time.sleep(0.5)
    # Weighted precision@k: earlier hits matter more
    weighted = sum(
        scores[i] * (sum(scores[:i+1]) / (i+1))
        for i in range(len(scores))
        if scores[i] == 1.0
    )
    return weighted / len(scores) if any(scores) else 0.0


# ── Main evaluation loop ──────────────────────────────────────────────────────

def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    from information_agent import load_vectorstore, get_information_with_context
    from config.llm_config import get_llm
    from eval_metrics import GROUND_TRUTH as EVAL_QUERIES

    print("#" * 70)
    print("# SETU Compliance RAG — RAGAS-equivalent Evaluation")
    print(f"# {len(EVAL_QUERIES)} queries | faithfulness, relevancy, recall, precision")
    print("#" * 70)

    vs  = load_vectorstore()
    llm = get_llm(temperature=0)

    results = []

    for entry in EVAL_QUERIES:
        qid   = entry["id"]
        query = entry["query"]
        ref   = GROUND_TRUTH.get(qid, "")
        print(f"\n[{qid}] ({entry['difficulty']}) {query[:70]}")

        # Retrieve answer + contexts
        t0 = time.perf_counter()
        try:
            answer, contexts = get_information_with_context(vs, query)
        except Exception as exc:
            answer, contexts = f"ERROR: {exc}", []
        latency_ms = round((time.perf_counter() - t0) * 1000)
        print(f"  {latency_ms}ms | {len(contexts)} contexts retrieved")
        time.sleep(5)  # extra guard: FactCorrector adds 2-3 LLM calls per query

        # Compute metrics
        print("  computing faithfulness...", end=" ", flush=True)
        faith = faithfulness(query, answer, contexts, llm)
        print(f"{faith:.2f}")

        print("  computing answer_relevancy...", end=" ", flush=True)
        relevancy = answer_relevancy(query, answer, llm)
        print(f"{relevancy:.2f}")
        time.sleep(1)

        print("  computing context_recall...", end=" ", flush=True)
        recall = context_recall(query, ref, contexts, llm)
        print(f"{recall:.2f}")

        print("  computing context_precision...", end=" ", flush=True)
        precision = context_precision(query, contexts, llm)
        print(f"{precision:.2f}")
        time.sleep(2)

        results.append({
            "id": qid,
            "difficulty": entry["difficulty"],
            "query": query,
            "answer": answer,
            "latency_ms": latency_ms,
            "faithfulness": round(faith, 3),
            "answer_relevancy": round(relevancy, 3),
            "context_recall": round(recall, 3),
            "context_precision": round(precision, 3),
        })

    # ── Aggregate ──────────────────────────────────────────────────────────────
    n = len(results)
    agg = {
        k: round(sum(r[k] for r in results) / n, 3)
        for k in ("faithfulness", "answer_relevancy", "context_recall", "context_precision")
    }

    print("\n" + "=" * 70)
    print("RAGAS-EQUIVALENT AGGREGATE METRICS")
    print("=" * 70)
    for k, v in agg.items():
        bar = "█" * int(v * 20) + "░" * (20 - int(v * 20))
        print(f"  {k:<28} {bar}  {v:.3f}")

    for label, diff in [("Easy", "easy"), ("Medium", "medium"), ("Hard", "hard")]:
        g = [r for r in results if r["difficulty"] == diff]
        if g:
            gn = len(g)
            print(f"\n{label} ({gn} queries):")
            for k in agg:
                print(f"  {k:<28} {sum(r[k] for r in g)/gn:.3f}")
            print(f"  {'avg_latency_ms':<28} {sum(r['latency_ms'] for r in g)/gn:.0f}ms")

    # ── Compare with custom eval ───────────────────────────────────────────────
    try:
        with open("eval_results.json") as f:
            prev = json.load(f)["aggregate"]
        print("\n── Comparison with custom (keyword) eval ──────────────────────────")
        print(f"  Custom  answer_recall    : {prev.get('answer_recall', '?')}")
        print(f"  Custom  citation_accuracy: {prev.get('citation_accuracy', '?')}")
        print(f"  Ragas   context_recall   : {agg['context_recall']}")
        print(f"  Ragas   faithfulness     : {agg['faithfulness']}")
        print("  (Ragas metrics are LLM-judged; custom metrics are keyword-based)")
    except Exception:
        pass

    # ── Save ───────────────────────────────────────────────────────────────────
    out = {"aggregate": agg, "per_query": results}
    with open("eval_ragas_results.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\nSaved → eval_ragas_results.json")


if __name__ == "__main__":
    main()
