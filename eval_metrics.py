"""
Formal evaluation script for the SETU Compliance RAG system.

Metrics computed per query:
  - Retrieval Precision  : fraction of retrieved chunks relevant to the query (LLM-scored)
  - Answer Recall        : fraction of ground-truth key facts present in the answer
  - Answer Precision     : fraction of answer sentences grounded in expected facts
  - Citation Accuracy    : does the answer name the correct source policy?
  - Latency (ms)         : end-to-end response time

Run:  python eval_metrics.py
"""
import json
import re
import sys
import time
from dotenv import load_dotenv

load_dotenv()

# ── Ground truth: 15 queries across easy / medium / hard ─────────────────────
GROUND_TRUTH = [
    # Easy — direct fact lookup
    {
        "id": "E1",
        "query": "How many weeks of maternity leave is a SETU staff member entitled to?",
        "expected_facts": ["26", "weeks", "maternity", "full pay"],
        "expected_source": "Leave Management Policy",
        "difficulty": "easy",
    },
    {
        "id": "E2",
        "query": "What lawful bases does SETU use to process personal data under GDPR?",
        "expected_facts": ["legitimate interest", "personal data", "process"],
        "expected_source": "Data Protection Policy",
        "difficulty": "easy",
    },
    {
        "id": "E3",
        "query": "What is SETU's policy on Garda vetting — when is it required and who must comply?",
        "expected_facts": ["vetting", "required", "position"],
        "expected_source": "Garda Vetting",
        "difficulty": "easy",
    },
    {
        "id": "E4",
        "query": "What must SETU staff disclose when using Gen AI tools for official work?",
        "expected_facts": ["responsible", "guidelines", "disclose"],
        "expected_source": "SETU Staff Guidelines on the Use of Gen AI",
        "difficulty": "easy",
    },
    {
        "id": "E5",
        "query": "What are SETU's equality and diversity obligations towards staff under the EDI Policy?",
        "expected_facts": ["equality", "diversity", "inclusive"],
        "expected_source": "Equality, Diversity",
        "difficulty": "easy",
    },
    # Medium — multi-step or less direct
    {
        "id": "M1",
        "query": "What leave is available to SETU staff with caring responsibilities?",
        "expected_facts": ["carer", "flexible", "leave"],
        "expected_source": "Caring Responsibilities",
        "difficulty": "medium",
    },
    {
        "id": "M2",
        "query": "What are the formal stages in reporting research misconduct at SETU?",
        "expected_facts": ["Stage", "investigation", "misconduct"],
        "expected_source": "Procedures for Managing Allegations",
        "difficulty": "medium",
    },
    {
        "id": "M3",
        "query": "Who owns intellectual property created by staff during employment at SETU?",
        "expected_facts": ["SETU", "University", "employment"],
        "expected_source": "Intellectual Property",
        "difficulty": "medium",
    },
    {
        "id": "M4",
        "query": "What are the consequences for a SETU staff member who breaches the Data Protection Policy?",
        "expected_facts": ["data", "breach", "policy"],
        "expected_source": "Data Protection Policy",
        "difficulty": "medium",
    },
    {
        "id": "M5",
        "query": "What is the Garda Vetting Policy scope — which roles does it apply to?",
        "expected_facts": ["children", "vulnerable", "vetting"],
        "expected_source": "Garda Vetting",
        "difficulty": "medium",
    },
    # Hard — cross-policy, nuanced, or edge cases
    {
        "id": "H1",
        "query": "How do the Equality Policy and Gender Identity Policy interact for a staff member requesting a name change?",
        "expected_facts": ["equality", "gender", "identity"],
        "expected_source": "Equality",
        "difficulty": "hard",
    },
    {
        "id": "H2",
        "query": "What is SETU's approach to academic integrity and what counts as misconduct?",
        "expected_facts": ["plagiarism", "misconduct", "academic"],
        "expected_source": "Academic",
        "difficulty": "hard",
    },
    {
        "id": "H3",
        "query": "Can SETU share employee personal data with a third-party recruitment agency?",
        "expected_facts": ["legitimate", "data", "personal"],
        "expected_source": "Data Protection Policy",
        "difficulty": "hard",
    },
    {
        "id": "H4",
        "query": "What types of behaviour constitute harassment under SETU's Dignity and Respect Policy?",
        "expected_facts": ["harassment", "dignity", "conduct"],
        "expected_source": "Dignity",
        "difficulty": "hard",
    },
    {
        "id": "H5",
        "query": "What does SETU's policy say about using AI to write a research paper, and what must be disclosed?",
        "expected_facts": ["responsible", "disclose", "guidelines"],
        "expected_source": "SETU Staff Guidelines on the Use of Gen AI",
        "difficulty": "hard",
    },
]


# ── Metric functions ──────────────────────────────────────────────────────────

def recall(answer: str, expected_facts: list[str]) -> float:
    """Fraction of expected key facts found anywhere in the answer."""
    a = answer.lower()
    hits = sum(1 for f in expected_facts if f.lower() in a)
    return hits / len(expected_facts) if expected_facts else 0.0


def precision(answer: str, expected_facts: list[str]) -> float:
    """
    Fraction of answer sentences that contain at least one expected fact.
    Approximates how much of the answer is grounded in the expected content.
    """
    sentences = [s.strip() for s in re.split(r"[.!?]", answer) if len(s.strip()) > 10]
    if not sentences:
        return 0.0
    relevant = sum(
        1 for s in sentences
        if any(f.lower() in s.lower() for f in expected_facts)
    )
    return relevant / len(sentences)


def citation_accuracy(answer: str, expected_source: str) -> float:
    """Fraction of significant words from the expected source name in the answer."""
    words = [w for w in expected_source.split() if len(w) > 3]
    if not words:
        return 0.0
    hits = sum(1 for w in words if w.lower() in answer.lower())
    return hits / len(words)


def retrieval_precision(query: str, retrieved_docs: list, llm) -> float:
    """LLM-scored fraction of retrieved chunks relevant to the query."""
    scores = []
    for doc in retrieved_docs[:3]:
        excerpt = doc.page_content[:350]
        prompt = (
            f"Question: {query}\n\nPolicy excerpt:\n{excerpt}\n\n"
            "Does this excerpt contain information useful for answering the question? "
            "Reply with only 1 (yes) or 0 (no)."
        )
        try:
            r = llm.invoke(prompt).content.strip()
            scores.append(int(r[0]))
        except Exception:
            scores.append(0)
    return sum(scores) / len(scores) if scores else 0.0


# ── Run evaluation ────────────────────────────────────────────────────────────

def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    from information_agent import load_vectorstore, get_information, _hybrid_retrieve
    from config.llm_config import get_llm

    vs  = load_vectorstore()
    llm = get_llm(temperature=0)

    print("#" * 70)
    print("# SETU Compliance RAG — Formal Evaluation")
    print(f"# {len(GROUND_TRUTH)} queries  |  precision, recall, citation, latency")
    print("#" * 70)

    results = []

    for entry in GROUND_TRUTH:
        qid   = entry["id"]
        query = entry["query"]
        print(f"\n[{qid}] ({entry['difficulty']}) {query[:72]}")

        # Retrieve chunks for retrieval-precision metric
        try:
            raw = _hybrid_retrieve(vs, query)
        except Exception:
            raw = []

        # Timed end-to-end answer
        t0 = time.perf_counter()
        try:
            answer = get_information(vs, query)
        except Exception as exc:
            answer = f"ERROR: {exc}"
        latency_ms = (time.perf_counter() - t0) * 1000

        time.sleep(3)  # avoid Groq TPM limit between queries

        rec  = recall(answer, entry["expected_facts"])
        prec = precision(answer, entry["expected_facts"])
        cit  = citation_accuracy(answer, entry["expected_source"])
        ret_prec = retrieval_precision(query, raw, llm)

        time.sleep(2)

        results.append({
            "id": qid,
            "difficulty": entry["difficulty"],
            "query": query,
            "answer": answer,
            "latency_ms": round(latency_ms),
            "retrieval_precision": round(ret_prec, 3),
            "answer_precision": round(prec, 3),
            "answer_recall": round(rec, 3),
            "citation_accuracy": round(cit, 3),
        })

        print(f"  Latency={latency_ms:.0f}ms  RetPrec={ret_prec:.2f}  "
              f"Prec={prec:.2f}  Recall={rec:.2f}  Cit={cit:.2f}")
        print(f"  A: {answer[:140].replace(chr(10),' ')}...")

    # ── Aggregate ──────────────────────────────────────────────────────────────
    n = len(results)
    avg_lat      = sum(r["latency_ms"]          for r in results) / n
    avg_ret_prec = sum(r["retrieval_precision"] for r in results) / n
    avg_prec     = sum(r["answer_precision"]    for r in results) / n
    avg_rec      = sum(r["answer_recall"]       for r in results) / n
    avg_cit      = sum(r["citation_accuracy"]   for r in results) / n

    print("\n" + "=" * 70)
    print("AGGREGATE METRICS")
    print("=" * 70)
    print(f"Queries evaluated    : {n}")
    print(f"Avg latency          : {avg_lat:.0f} ms")
    print(f"Retrieval precision  : {avg_ret_prec:.2f}  (fraction of retrieved chunks that are relevant)")
    print(f"Answer precision     : {avg_prec:.2f}  (fraction of answer sentences grounded in expected facts)")
    print(f"Answer recall        : {avg_rec:.2f}  (fraction of expected key facts covered by answer)")
    print(f"Citation accuracy    : {avg_cit:.2f}  (fraction of correct source names cited)")

    for label, diff in [("Easy (5)", "easy"), ("Medium (5)", "medium"), ("Hard (5)", "hard")]:
        g = [r for r in results if r["difficulty"] == diff]
        if g:
            gn = len(g)
            print(f"\n{label}:")
            print(f"  Precision={sum(r['answer_precision'] for r in g)/gn:.2f}  "
                  f"Recall={sum(r['answer_recall'] for r in g)/gn:.2f}  "
                  f"Citation={sum(r['citation_accuracy'] for r in g)/gn:.2f}  "
                  f"Latency={sum(r['latency_ms'] for r in g)/gn:.0f}ms")

    print("\n" + "-" * 70)
    print(f"{'ID':<4} {'Diff':<7} {'Prec':>5} {'Rec':>5} {'Cit':>5} {'ms':>6}  Query")
    print("-" * 70)
    for r in results:
        print(f"{r['id']:<4} {r['difficulty']:<7} "
              f"{r['answer_precision']:>5.2f} {r['answer_recall']:>5.2f} "
              f"{r['citation_accuracy']:>5.2f} {r['latency_ms']:>6}  "
              f"{r['query'][:40]}")

    out_path = "eval_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "aggregate": {
                "n": n,
                "avg_latency_ms": round(avg_lat),
                "retrieval_precision": round(avg_ret_prec, 3),
                "answer_precision": round(avg_prec, 3),
                "answer_recall": round(avg_rec, 3),
                "citation_accuracy": round(avg_cit, 3),
            },
            "per_query": results,
        }, f, ensure_ascii=False, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
