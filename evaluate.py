"""
Production evaluation framework for the SETU Compliance RAG system.

Metrics (inspired by RAGAS — no external dependency needed):
  - Latency          : end-to-end response time (ms)
  - Context Precision: fraction of retrieved chunks that are actually relevant
  - Faithfulness     : does the answer stay within the retrieved context?
  - Citation Quality : does the answer name the source policy document?
  - Answer Relevance : does the answer address the actual question asked?

Run:  python evaluate.py
"""
import time
from dotenv import load_dotenv
load_dotenv()

from information_agent import load_vectorstore, get_information, _hybrid_retrieve
from config.llm_config import get_llm

vs  = load_vectorstore()
llm = get_llm(temperature=0)

# ── 10-query test set ─────────────────────────────────────────────────────────
test_set = [
    {
        "query":    "What does SETU's Data Protection Policy say about lawful basis for processing personal data?",
        "expected": "GDPR lawful basis: consent, legal obligation, or legitimate interest",
        "source":   "Data Protection Policy",
        "difficulty": "easy",
    },
    {
        "query":    "What is the purpose of the CCTV Policy at SETU and who can access the footage?",
        "expected": "security and crime prevention; controlled access by authorised personnel",
        "source":   "Closed Circuit Television (CCTV) Policy",
        "difficulty": "easy",
    },
    {
        "query":    "What are staff obligations under the Conflict of Interest Policy?",
        "expected": "declare conflicts to manager or HR; recuse from affected decisions",
        "source":   "Conflict of Interest Policy",
        "difficulty": "easy",
    },
    {
        "query":    "What does the Gen AI guidelines document say about staff using AI tools for official work?",
        "expected": "staff must disclose AI use, check accuracy, not share sensitive data",
        "source":   "SETU Staff Guidelines on the Use of Gen AI",
        "difficulty": "easy",
    },
    {
        "query":    "What leave entitlements exist for employees with caring responsibilities?",
        "expected": "carers leave or flexible arrangements under leave or caring responsibilities policy",
        "source":   "Supporting Employees with Caring Responsibilities Policy",
        "difficulty": "medium",
    },
    {
        "query":    "How long does SETU retain different categories of personal data?",
        "expected": "retention periods defined by category (student records, HR, financial)",
        "source":   "Data Retention Policy",
        "difficulty": "medium",
    },
    {
        "query":    "What are the formal stages in reporting research misconduct at SETU?",
        "expected": "initial report, assessment, investigation panel, outcome, appeal",
        "source":   "Procedures for Managing Allegations of Misconduct in Research",
        "difficulty": "medium",
    },
    {
        "query":    "What does the Intellectual Property Policy say about IP created during employment?",
        "expected": "IP created in course of employment belongs to SETU",
        "source":   "Intellectual Property Policy",
        "difficulty": "medium",
    },
    {
        "query":    "What are the consequences of breaching the Data Protection Policy for a SETU staff member?",
        "expected": "disciplinary action under HR policy; possible termination; regulatory fine",
        "source":   "Data Protection Policy",
        "difficulty": "hard",
    },
    {
        "query":    "How do the Equality Policy and the Gender Identity Policy interact for a staff member requesting a name change?",
        "expected": "both policies apply: equality protects against discrimination; gender identity supports name/pronoun change",
        "source":   "Equality, Diversity & Inclusion Policy / Gender Identity & Expression Policy",
        "difficulty": "hard",
    },
]


# ── Metric helpers ────────────────────────────────────────────────────────────

def measure_context_precision(query: str, retrieved_docs: list, llm) -> float:
    """
    Context Precision: what fraction of retrieved chunks contain information
    relevant to answering the query?
    Scores each chunk 0 or 1 with an LLM call, averages them.
    """
    scores = []
    for doc in retrieved_docs:
        chunk_text = doc.metadata.get("parent_doc", doc.page_content)[:400]
        prompt = (
            f"Question: {query}\n\n"
            f"Policy excerpt:\n{chunk_text}\n\n"
            "Does this excerpt contain information useful for answering the question?\n"
            "Reply with only 1 (yes) or 0 (no)."
        )
        try:
            resp = llm.invoke(prompt).content.strip()[0]
            scores.append(int(resp))
        except Exception:
            scores.append(0)
    return sum(scores) / len(scores) if scores else 0.0


def measure_faithfulness(answer: str, context: str, llm) -> bool:
    """
    Faithfulness: does the answer make claims NOT supported by the context?
    Returns True (faithful) if no contradictions found.
    """
    prompt = (
        f"Context (policy excerpts):\n{context[:800]}\n\n"
        f"Answer: {answer[:400]}\n\n"
        "Does the answer make any claim that contradicts or is absent from the context?\n"
        "Reply CONTRADICT or SUPPORTED."
    )
    result = llm.invoke(prompt).content.strip().upper()
    return "CONTRADICT" not in result


def measure_citation_quality(answer: str, expected_source: str) -> float:
    """
    Citation Quality: does the answer mention the expected source policy?
    Simple string check — fast, no LLM call.
    """
    # Check if any significant word from the expected source appears in the answer
    key_words = [w for w in expected_source.split() if len(w) > 3]
    hits = sum(1 for w in key_words if w.lower() in answer.lower())
    return hits / len(key_words) if key_words else 0.0


def measure_answer_relevance(query: str, answer: str, llm) -> int:
    """
    Answer Relevance (0-2): does the answer directly address the question?
    """
    prompt = (
        f"Question: {query}\n\nAnswer: {answer[:400]}\n\n"
        "Rate how directly the answer addresses the question:\n"
        "2 = directly and completely answers the question\n"
        "1 = partially answers or is too vague\n"
        "0 = does not answer the question\n"
        "Reply with a single digit only."
    )
    try:
        return max(0, min(2, int(llm.invoke(prompt).content.strip()[0])))
    except Exception:
        return 1


# ── Run evaluation ────────────────────────────────────────────────────────────

print("#" * 65)
print("# SETU Compliance RAG — Production Evaluation")
print("#" * 65)
print(f"Queries: {len(test_set)} | Metrics: latency, context_precision, faithfulness, citation, relevance\n")

results = []

for i, entry in enumerate(test_set, 1):
    query      = entry["query"]
    expected   = entry["expected"]
    source     = entry["source"]
    difficulty = entry["difficulty"]

    print(f"[{i:02d}/{len(test_set)}] {query[:70]}")

    # Retrieve raw chunks for context-level metrics
    raw_chunks = _hybrid_retrieve(vs, query)
    context    = "\n\n".join([
        d.metadata.get("parent_doc", d.page_content) for d in raw_chunks[:3]
    ])

    # Timed answer generation
    t0     = time.perf_counter()
    answer = get_information(vs, query)
    latency_ms = (time.perf_counter() - t0) * 1000

    # Compute metrics
    ctx_precision = measure_context_precision(query, raw_chunks[:3], llm)
    faithful      = measure_faithfulness(answer, context, llm)
    citation      = measure_citation_quality(answer, source)
    relevance     = measure_answer_relevance(query, answer, llm)

    results.append({
        "query":         query[:55],
        "difficulty":    difficulty,
        "latency_ms":    latency_ms,
        "ctx_precision": ctx_precision,
        "faithful":      faithful,
        "citation":      citation,
        "relevance":     relevance,
    })

    print(f"       A: {answer[:120].replace(chr(10), ' ')}...")
    print(f"       Latency={latency_ms:.0f}ms  CtxPrec={ctx_precision:.2f}  "
          f"Faithful={'YES' if faithful else 'NO'}  "
          f"Citation={citation:.2f}  Relevance={relevance}/2")
    print()


# ── Aggregate ─────────────────────────────────────────────────────────────────
print("=" * 65)
print("AGGREGATE METRICS")
print("=" * 65)

n = len(results)
avg_lat  = sum(r["latency_ms"]    for r in results) / n
avg_prec = sum(r["ctx_precision"] for r in results) / n
faith_ok = sum(1 for r in results if r["faithful"])
avg_cit  = sum(r["citation"]      for r in results) / n
avg_rel  = sum(r["relevance"]     for r in results) / n

print(f"Avg latency          : {avg_lat:.0f} ms")
print(f"Avg context precision: {avg_prec:.2f}  (fraction of retrieved chunks that are relevant)")
print(f"Faithfulness         : {faith_ok}/{n} answers grounded in context")
print(f"Avg citation quality : {avg_cit:.2f}  (0-1, does answer name the right policy?)")
print(f"Avg answer relevance : {avg_rel:.2f}/2.0")

for label, diff in [("Easy", "easy"), ("Medium", "medium"), ("Hard", "hard")]:
    group = [r for r in results if r["difficulty"] == diff]
    if group:
        avg = sum(r["relevance"] for r in group) / len(group)
        lat = sum(r["latency_ms"] for r in group) / len(group)
        print(f"{label} ({len(group)}): relevance={avg:.2f}/2.0  latency={lat:.0f}ms")

print()
print("-" * 95)
print(f"{'Query':<57} {'Diff':<8} {'ms':>5} {'Prec':>5} {'Faith':>6} {'Cit':>5} {'Rel':>4}")
print("-" * 95)
for r in results:
    print(
        f"{r['query']:<57} {r['difficulty']:<8} "
        f"{r['latency_ms']:>5.0f} {r['ctx_precision']:>5.2f} "
        f"{'[OK]':>6} " if r['faithful'] else f"{'[NO]':>6} ",
        end="",
    )
    print(f"{r['citation']:>5.2f} {r['relevance']:>4}/2")
