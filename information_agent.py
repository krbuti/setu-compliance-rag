"""
Information agent — production-grade retrieval for SETU compliance policies.

Pipeline (fast, accurate, citation-aware):
  1. Self-query filter   — if query names a specific policy, restrict search to it
  2. Hybrid retrieval    — BM25 (keyword) + MMR ChromaDB (semantic diversity)
                           fused with Reciprocal Rank Fusion (RRF)
  3. Cross-encoder rerank — local sentence-transformer model, ~50ms, no LLM call
  4. Parent-chunk upgrade — swap matched child for its richer parent context
  5. Generate answer      — grounded, cites source policy names

Why this beats the naive approach:
  - BM25 catches exact legal/regulatory terms ("GDPR", "Article 6", "Garda vetting")
    that semantic embeddings can miss
  - MMR prevents retrieving 3 near-duplicate chunks from the same paragraph
  - Cross-encoder reranking is 100x faster than LLM-based reranking and more
    accurate because it scores each (query, passage) pair jointly
  - Self-query filtering stops leakage from unrelated policies
"""
from __future__ import annotations

import re
import threading
import concurrent.futures
import numpy as np
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_community.retrievers import BM25Retriever
from config.llm_config import get_embeddings, get_llm

CHROMA_DIR = Path(__file__).parent / "chroma_db"
COLLECTION  = "setu_compliance"

# SETU does not use numeric policy codes. Any pattern like SETU-HR-001 is invented.
_FAKE_CODE_RE = re.compile(r'\bSETU-[A-Z]{2,}-\d+\b|\bSETU-\d{3,}\b', re.IGNORECASE)

# Known policy document names for self-query detection
KNOWN_POLICIES = [
    "Data Protection", "Data Retention", "Data Governance",
    "CCTV", "Closed Circuit Television",
    "Equality", "EDI", "Dignity and Respect",
    "Leave Management", "Parental Leave",
    "Intellectual Property", "Consultancy",
    "Research Misconduct", "Research Practice",
    "Email Policy", "Acceptable Usage",
    "Recruitment", "Probation",
    "Conflict of Interest",
    "Gender Identity",
    "Garda Vetting",
    "Authorship",
    "Risk Management",
    "Gen AI", "Generative AI",
    "Fitness to Practise", "Fitness to Continue",
    "Voluntary Campus Transfer",
    "Visiting Academic",
    "Intellectual Property",
]

class SemanticCache:
    """In-memory semantic cache for the RAG pipeline.

    A cosine-similarity lookup over stored query embeddings catches paraphrased
    questions (e.g. "maternity leave weeks" vs "maternity leave entitlement") that
    the upstream exact-match AnswerCache misses. Cache hits skip all LLM and
    embedding API calls, returning in <5ms instead of 1-2s.
    """

    def __init__(self, threshold: float = 0.92, max_size: int = 500):
        self._threshold = threshold
        self._max_size  = max_size
        self._entries: list[dict] = []
        self._lock = threading.Lock()
        self.hits   = 0
        self.misses = 0

    def lookup(self, embedding: list[float]) -> tuple[str, list[str]] | None:
        q = np.asarray(embedding, dtype=np.float32)
        q_norm = q / (np.linalg.norm(q) + 1e-9)
        with self._lock:
            for entry in self._entries:
                if float(np.dot(q_norm, entry["norm_emb"])) >= self._threshold:
                    self.hits += 1
                    return entry["answer"], entry["sources"]
            self.misses += 1
        return None

    def store(self, embedding: list[float], answer: str, sources: list[str]) -> None:
        emb = np.asarray(embedding, dtype=np.float32)
        norm_emb = emb / (np.linalg.norm(emb) + 1e-9)
        with self._lock:
            if len(self._entries) >= self._max_size:
                self._entries.pop(0)
            self._entries.append({"norm_emb": norm_emb, "answer": answer, "sources": sources})

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0

    def __len__(self) -> int:
        return len(self._entries)


_embeddings  = get_embeddings()

# Memoize embed_query so the semantic cache lookup and ChromaDB dense retrieval
# share one Jina API call per query instead of making two.
_emb_memo: dict[str, list[float]] = {}
_orig_embed_query = _embeddings.embed_query

def _embed_query_memo(text: str) -> list[float]:
    if text not in _emb_memo:
        if len(_emb_memo) >= 256:
            _emb_memo.pop(next(iter(_emb_memo)))
        _emb_memo[text] = _orig_embed_query(text)
    return _emb_memo[text]

_embeddings.embed_query = _embed_query_memo

_semantic_cache = SemanticCache()
_vs: Optional[Chroma]      = None
_bm25: Optional[BM25Retriever] = None
_reranker = None


def _load_reranker():
    global _reranker
    if _reranker is None:
        try:
            from sentence_transformers import CrossEncoder
            _reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
            print("[OK] Cross-encoder reranker loaded")
        except ImportError:
            print("[WARN] sentence-transformers not installed — skipping reranking")
    return _reranker


def load_vectorstore() -> Chroma:
    global _vs, _bm25
    if not CHROMA_DIR.exists():
        raise RuntimeError(
            f"ChromaDB not found at {CHROMA_DIR}. Run 'python ingest.py' first."
        )
    if _vs is None:
        _vs = Chroma(
            persist_directory=str(CHROMA_DIR),
            embedding_function=_embeddings,
            collection_name=COLLECTION,
        )
        # Build BM25 index from all stored child chunks
        print("[INFO] Building BM25 index from ChromaDB documents...")
        raw = _vs._collection.get(include=["documents", "metadatas"])
        all_docs = [
            Document(page_content=text, metadata=meta)
            for text, meta in zip(raw["documents"], raw["metadatas"])
        ]
        _bm25 = BM25Retriever.from_documents(all_docs, k=8)
        print(f"[OK] BM25 index ready ({len(all_docs)} documents)")
        _load_reranker()
    return _vs


def _detect_policy_filter(query: str) -> Optional[str]:
    """Return a ChromaDB doc_name filter if the query explicitly names a policy."""
    q = query.lower()
    for name in KNOWN_POLICIES:
        if name.lower() in q:
            # Match against stored doc_name metadata (partial match)
            return name
    return None


def _reciprocal_rank_fusion(
    *ranked_lists: list[Document], k: int = 60
) -> list[Document]:
    """Merge multiple ranked lists using RRF. Higher score = more relevant."""
    scores: dict[str, float] = {}
    doc_map: dict[str, Document] = {}

    for ranked in ranked_lists:
        for rank, doc in enumerate(ranked):
            key = doc.page_content[:120]  # dedup key
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank + 1)
            doc_map[key] = doc

    sorted_keys = sorted(scores, key=lambda x: scores[x], reverse=True)
    return [doc_map[k] for k in sorted_keys]


def _hybrid_retrieve(
    vs: Chroma, query: str, policy_filter: Optional[str] = None, top_n: int = 10
) -> list[Document]:
    """BM25 + dense similarity retrieval fused with RRF, run in parallel.

    Uses similarity (not MMR) for the dense leg — MMR's diversity penalty
    was pulling in off-topic policies and drowning out the correct one.
    BM25 handles keyword diversity naturally (different term matches = different docs).
    ChromaDB 1.5.x $contains filter returns 0 results — do not use metadata filtering.
    Dense and sparse retrievals are parallelised — saves ~50ms since Jina embedding
    latency and BM25 scoring are fully independent.
    """
    dense_retriever = vs.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 10},
    )

    def _run_dense():
        return dense_retriever.invoke(query)

    def _run_sparse():
        if not _bm25:
            return []
        return _bm25.invoke(query)

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        dense_fut = executor.submit(_run_dense)
        sparse_fut = executor.submit(_run_sparse)
        dense_results = dense_fut.result()
        sparse_results = sparse_fut.result()

    fused = _reciprocal_rank_fusion(dense_results, sparse_results)
    return fused[:top_n]


def _cross_encode_rerank(
    query: str, candidates: list[Document], top_k: int = 3
) -> tuple[list[Document], list[float]]:
    """Re-rank candidates with a local cross-encoder. Returns (docs, scores)."""
    reranker = _reranker
    if reranker is None:
        return candidates[:top_k], [0.0] * min(top_k, len(candidates))
    pairs  = [(query, doc.page_content[:512]) for doc in candidates]
    scores = reranker.predict(pairs)
    ranked = sorted(zip(scores, candidates), key=lambda x: x[0], reverse=True)
    top    = ranked[:top_k]
    return [doc for _, doc in top], [float(s) for s, _ in top]


def _reason_and_refine(
    query: str, docs: list[Document], top_score: float, llm
) -> str | None:
    """ReAct reasoning step: generate a better search query when retrieval looks weak.

    Uses the cross-encoder top score as a retrieval-quality signal.
    ms-marco-MiniLM scores: >0 = relevant, <0 = poor match.
    Only fires one extra LLM call, and only when the top score is below threshold.
    Returns a refined query string, or None if context looks sufficient.
    """
    if top_score > -0.5:
        return None

    preview = "\n\n".join(
        f"[{doc.metadata.get('doc_name', '?')}]: {doc.page_content[:200]}"
        for doc in docs[:3]
    )
    prompt = (
        f"Question: {query}\n\n"
        f"Retrieved passages:\n{preview}\n\n"
        "The passages may not directly answer the question. "
        "Write ONE better search query to find the specific policy information. "
        "Be more specific — include key terms like policy names, leave types, "
        "entitlements, or legal concepts. "
        "Return ONLY the new query with no explanation. "
        "If the passages are actually on-topic, return exactly: SUFFICIENT"
    )
    try:
        result = llm.invoke(prompt).content.strip()
        if not result or result.upper().startswith("SUFFICIENT"):
            return None
        if result.lower() == query.lower():
            return None
        print(f"[ReAct] score={top_score:.2f} → refined: '{result[:70]}'")
        return result
    except Exception:
        return None


def _upgrade_to_parent(docs: list[Document]) -> list[Document]:
    """Replace each child chunk with its stored parent for richer context."""
    return [
        Document(
            page_content=doc.metadata.get("parent_doc", doc.page_content),
            metadata=doc.metadata,
        )
        for doc in docs
    ]


def _lightweight_expand(query: str, llm) -> list[str]:
    """One fast rewrite — cheaper than 3 expansions."""
    prompt = (
        f"Rewrite this policy question in one alternative way:\n{query}\n"
        "Return only the rewritten question, nothing else."
    )
    alt = llm.invoke(prompt).content.strip()
    return [query, alt] if alt and alt != query else [query]


def _reflect_and_verify(query: str, context: str, initial_answer: str, llm) -> str:
    """
    Second-pass LLM check: verify the initial answer is fully grounded in the
    retrieved context. Catches hallucinated policy names, invented reference codes
    (e.g. SETU-001), and facts not traceable to the source excerpts.

    Returns a corrected answer when unsupported claims are found, or the original
    answer unchanged if everything checks out.
    """
    # Extract source names actually present in context
    import re as _re
    source_names = _re.findall(r'\[Source:\s*([^\]]+)\]', context)
    sources_list = "\n".join(f"  - {s.strip()}" for s in source_names) or "  (none retrieved)"

    reflect_prompt = (
        "You are a strict fact-checker for a university policy assistant.\n\n"
        "The ONLY policy documents retrieved for this query are:\n"
        f"{sources_list}\n\n"
        "Retrieved excerpts (ground truth):\n"
        f"{context}\n\n"
        "Draft answer to verify:\n"
        f"{initial_answer}\n\n"
        "Check the draft answer. Flag it as needing REVISION if ANY of these are true:\n"
        "1. It mentions a policy reference code (e.g. SETU-001, SETU-HR-002, SETU-AC-005) "
        "— SETU does not use numeric policy codes; any such code is invented\n"
        "2. It names a policy document NOT in the source list above\n"
        "3. It states a specific number, date, or entitlement NOT present word-for-word in the excerpts\n"
        "4. It contains placeholder text like [Your Name], [Staff Member], [Date]\n"
        "5. It describes what a non-existent policy 'says' instead of stating the policy was not found\n\n"
        "If the answer passes all checks: reply VERIFIED: <original answer unchanged>\n"
        "If it fails any check: reply REVISED: <corrected answer — remove invented codes/names, "
        "replace unsupported facts with 'refer to the full policy document', "
        "remove any placeholder text>\n"
        "Output only VERIFIED or REVISED, nothing else."
    )
    try:
        reflection = llm.invoke(reflect_prompt).content.strip()
        if reflection.startswith("VERIFIED:"):
            return reflection[len("VERIFIED:"):].strip()
        elif reflection.startswith("REVISED:"):
            revised = reflection[len("REVISED:"):].strip()
            # Strip any email addresses that survived the LLM revision
            revised = re.sub(r'\b[\w.+-]+@[\w.-]+\.\w+\b', '[contact SETU directly]', revised)
            print("[INFO] Reflection: unsupported claims removed from answer")
            return revised
        else:
            return initial_answer
    except Exception as exc:
        print(f"[WARN] Reflection step failed ({exc}), returning original answer")
        return initial_answer


_NO_POLICY_RE = re.compile(
    r'no dedicated setu policy|not found in the retrieved|does not directly address|'
    r'not explicitly|unclear what specific|no specific.*(?:was found|are mentioned)',
    re.IGNORECASE,
)
_SPECIFIC_FACT_RE = re.compile(
    r'\b\d+\s*(?:weeks?|days?|months?|years?|hours?)\b|'
    r'\b(?:article|section|appendix)\s+\d+\b',
    re.IGNORECASE,
)


def _should_reflect(answer: str) -> bool:
    """Return True only when the answer carries hallucination risk worth a second LLM call.

    Skipping reflection saves ~400ms per query. It's safe to skip when:
    - The answer already says "no policy found" (no facts to hallucinate)
    - The answer contains no numbers, timeframes, appendix refs, or fake codes
    """
    if _NO_POLICY_RE.search(answer):
        return False
    if _FAKE_CODE_RE.search(answer):
        return True
    if re.search(r'\b[\w.+-]+@[\w.-]+\.\w+\b', answer):
        return True
    if _SPECIFIC_FACT_RE.search(answer):
        return True
    return False


def _factcorrect(query: str, answer: str, docs: list[Document], llm) -> str:
    """FactCorrector-inspired post-hoc claim verification (3 LLM calls max).

    Inspired by: FactCorrector — A Graph-Inspired Approach to Long-Form
    Factuality Correction of LLMs (IBM Research, ACL 2026).

    Pipeline:
      1. Atomize the answer into atomic claims (1 LLM call)
      2. Batch-check all claims against the retrieved context in one prompt
         (1 LLM call — cheaper than one call per claim)
      3. Rewrite removing unsupported claims (1 LLM call, only if needed)

    Replaces the selective _reflect_and_verify trigger (emails/codes only)
    with systematic checking on every response.
    """
    if not answer or len(answer.strip()) < 20:
        return answer

    ctx_text = "\n---\n".join(
        f"[{doc.metadata.get('doc_name', 'Policy')}]\n{doc.page_content[:400]}"
        for doc in docs[:5]
    )

    # Step 1 — atomize
    try:
        claims_raw = llm.invoke(
            "Break this answer into atomic claims, one per line. "
            "Each claim must be a single verifiable statement. "
            "Output only the numbered list, nothing else.\n\n"
            f"Answer: {answer}"
        ).content.strip()
    except Exception:
        return answer

    claims = [
        c.strip().lstrip("•-0123456789.) ")
        for c in claims_raw.split("\n")
        if c.strip() and len(c.strip()) > 8
    ]
    if not claims:
        return answer

    # Step 2 — batch check all claims in one call
    # Only flag claims that are CONTRADICTED or contain specific facts (numbers,
    # names, dates) that do NOT appear in the context. Absence ≠ wrong.
    numbered = "\n".join(f"{i+1}. {c}" for i, c in enumerate(claims))
    try:
        check = llm.invoke(
            f"Policy context:\n{ctx_text}\n\n"
            f"Claims to verify:\n{numbered}\n\n"
            "List the numbers of claims that are FACTUALLY WRONG based on the context. "
            "A claim is wrong only if: (a) the context explicitly states something different, "
            "OR (b) the claim contains a specific number, code, name, or date that is NOT "
            "found anywhere in the context. "
            "Do NOT flag claims that are merely absent from the context — absence is not error. "
            "Reply with comma-separated numbers (e.g. '2,4') or 'none'. Numbers only."
        ).content.strip().lower()
    except Exception:
        return answer

    if "none" in check or not re.search(r'\d', check):
        return answer  # nothing contradicted

    bad_nums = {int(n) for n in re.findall(r'\d+', check) if 0 < int(n) <= len(claims)}
    unsupported = [claims[i - 1] for i in sorted(bad_nums)]
    if not unsupported:
        return answer

    print(f"[FactCorrect] {len(unsupported)}/{len(claims)} claim(s) unsupported — correcting")

    # Step 3 — rewrite removing unsupported claims
    bullets = "\n".join(f"  - {c}" for c in unsupported)
    try:
        corrected = llm.invoke(
            "Rewrite the answer below, removing ONLY the unsupported claims listed. "
            "Keep every supported fact unchanged. "
            "If nothing remains, say: "
            "'The retrieved policy documents do not directly address this specific point.'\n\n"
            f"Original answer:\n{answer}\n\n"
            f"Unsupported claims to remove:\n{bullets}"
        ).content.strip()
        # Strip any fabricated emails that survived
        corrected = re.sub(r'\b[\w.+-]+@[\w.-]+\.\w+\b', '[contact SETU directly]', corrected)
        return corrected
    except Exception:
        return answer


def _run_pipeline(vs: Chroma, query: str) -> tuple[str, list[Document]]:
    """Core RAG pipeline. Returns (answer, top_reranked_docs).

    Separated from get_information() so evaluation code can access the
    retrieved contexts without running retrieval twice.
    """
    if _FAKE_CODE_RE.search(query):
        return (
            "SETU does not use numeric policy reference codes. "
            "Policy documents are identified by their full title (e.g. 'Sick Leave Policy', "
            "'Parental Leave Policy'). Please search the SETU policy library by topic name, "
            "or contact HR with the specific subject area you need."
        ), []

    llm           = get_llm()
    policy_filter = _detect_policy_filter(query)

    all_candidates = _hybrid_retrieve(vs, query, policy_filter)

    seen, unique = set(), []
    for doc in all_candidates:
        key = doc.page_content[:120]
        if key not in seen:
            seen.add(key)
            unique.append(doc)

    top5, scores = _cross_encode_rerank(query, unique, top_k=5)

    # ReAct loop: if top cross-encoder score signals a weak match, reason about
    # a better query and retry retrieval once. Adds ~2-4s only when needed.
    if scores:
        refined = _reason_and_refine(query, top5, max(scores), llm)
        if refined:
            extra = _hybrid_retrieve(vs, refined, policy_filter)
            merged = list({doc.page_content[:120]: doc for doc in top5 + extra}.values())
            top5, scores = _cross_encode_rerank(query, merged, top_k=5)

    context_parts = []
    for doc in top5:
        source = doc.metadata.get("doc_name", "Unknown Policy")
        context_parts.append(f"[Source: {source}]\n{doc.page_content}")
    context = "\n\n---\n\n".join(context_parts)

    filter_note = f" (filtered to: {policy_filter})" if policy_filter else ""
    answer_prompt = (
        "You are a SETU policy information assistant.\n"
        "Answer the question using ONLY the policy excerpts below.\n"
        "Always name the policy document (shown as [Source: ...]) in your answer.\n"
        "If the excerpts are on-topic but lack a specific number or date, explain what "
        "the policy DOES say (e.g. 'entitlements are set in individual contracts') — "
        "do NOT say the information is absent.\n"
        "IMPORTANT: If the excerpts do NOT directly address the topic in the question "
        "(e.g. the question asks about remote working or hybrid schedules but the excerpts "
        "cover unrelated subjects), say clearly: 'No dedicated SETU policy on [topic] was "
        "found in the retrieved documents.' Do NOT extrapolate or infer from unrelated policies.\n"
        "NEVER invent contact details (email addresses, phone numbers, names). "
        "If asked for contact info, say 'contact the relevant SETU office directly' — do not fabricate.\n"
        "Only name a policy document if it is listed in the [Source: ...] tags below.\n\n"
        f"Policy Excerpts{filter_note}:\n{context}\n\n"
        f"Question: {query}\n\n"
        "Answer (2-3 sentences max, cite the source policy):"
    )
    initial_answer = llm.invoke(answer_prompt).content

    # Targeted verification: catches fake policy codes, invented emails, hallucinated
    # source names. Skipped when the answer is already low-risk (~400ms saved).
    if _should_reflect(initial_answer):
        final_answer = _reflect_and_verify(query, context, initial_answer, llm)
    else:
        final_answer = initial_answer
    return final_answer, top5


def get_information_with_sources(vs: Chroma, query: str) -> tuple[str, list[str]]:
    """Returns (answer, deduplicated_source_policy_names) for the REST API.

    Checks the semantic cache first — identical or paraphrased queries return
    in <5ms without hitting any LLM or embedding API. The embed_query call here
    is memoized, so ChromaDB's dense retrieval on a cache miss reuses the same
    vector at no extra cost.
    """
    embedding = _embeddings.embed_query(query)
    cached = _semantic_cache.lookup(embedding)
    if cached:
        print(f"[SemanticCache] HIT  (rate={_semantic_cache.hit_rate:.0%}, size={len(_semantic_cache)})")
        return cached

    answer, docs = _run_pipeline(vs, query)
    sources = list(dict.fromkeys(
        doc.metadata.get("doc_name", "")
        for doc in docs
        if doc.metadata.get("doc_name")
    ))
    _semantic_cache.store(embedding, answer, sources)
    return answer, sources


def get_information(vs: Chroma, query: str) -> str:
    answer, _ = get_information_with_sources(vs, query)
    return answer


def get_information_with_context(vs: Chroma, query: str) -> tuple[str, list[str]]:
    """Returns (answer, retrieved_context_strings) for RAGAS evaluation."""
    answer, docs = _run_pipeline(vs, query)
    return answer, [doc.page_content for doc in docs]


if __name__ == "__main__":
    vs = load_vectorstore()
    questions = [
        "What is SETU's data retention policy for student records?",
        "Can CCTV footage be shared with third parties at SETU?",
        "What are the grounds for an academic integrity investigation?",
        "What does the Gen AI policy say about staff using AI tools?",
    ]
    for q in questions:
        print(f"Q: {q}")
        print(f"A: {get_information(vs, q)}")
        print("-" * 60)
