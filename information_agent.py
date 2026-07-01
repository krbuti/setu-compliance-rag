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

_embeddings  = get_embeddings()
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
    vs: Chroma, query: str, policy_filter: Optional[str] = None, top_n: int = 12
) -> list[Document]:
    """BM25 + MMR dense retrieval fused with RRF."""
    # Dense: MMR ensures diversity across the retrieved set
    dense_kwargs = {
        "search_type": "mmr",
        "search_kwargs": {"k": 8, "fetch_k": 30, "lambda_mult": 0.7},
    }
    if policy_filter:
        dense_kwargs["search_kwargs"]["filter"] = {"doc_name": {"$contains": policy_filter}}

    dense_results = vs.as_retriever(**dense_kwargs).invoke(query)

    # Sparse: BM25 keyword retrieval
    sparse_results = _bm25.invoke(query) if _bm25 else []

    # Fuse with RRF
    fused = _reciprocal_rank_fusion(dense_results, sparse_results)
    return fused[:top_n]


def _cross_encode_rerank(
    query: str, candidates: list[Document], top_k: int = 3
) -> list[Document]:
    """Re-rank candidates with a local cross-encoder. Falls back to RRF order."""
    reranker = _reranker
    if reranker is None:
        return candidates[:top_k]
    pairs  = [(query, doc.page_content[:512]) for doc in candidates]
    scores = reranker.predict(pairs)
    ranked = sorted(zip(scores, candidates), key=lambda x: x[0], reverse=True)
    return [doc for _, doc in ranked[:top_k]]


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
            print("[INFO] Reflection: unsupported claims removed from answer")
            return revised
        else:
            return initial_answer
    except Exception as exc:
        print(f"[WARN] Reflection step failed ({exc}), returning original answer")
        return initial_answer


def get_information(vs: Chroma, query: str) -> str:
    llm             = get_llm()
    policy_filter   = _detect_policy_filter(query)
    queries         = _lightweight_expand(query, llm)

    # Hybrid retrieval for each query variant, then fuse
    all_candidates: list[Document] = []
    for q in queries:
        all_candidates.extend(_hybrid_retrieve(vs, q, policy_filter))

    # Deduplicate by content
    seen, unique = set(), []
    for doc in all_candidates:
        key = doc.page_content[:120]
        if key not in seen:
            seen.add(key)
            unique.append(doc)

    # Cross-encoder rerank -> top 5 then use children directly
    # (parent-chunk upgrade skipped: stored parent metadata may point to wrong sections)
    top5 = _cross_encode_rerank(query, unique, top_k=5)
    context_docs = top5

    # Build cited context block
    context_parts = []
    for doc in context_docs:
        source = doc.metadata.get("doc_name", "Unknown Policy")
        context_parts.append(f"[Source: {source}]\n{doc.page_content}")
    context = "\n\n---\n\n".join(context_parts)

    filter_note = f" (filtered to: {policy_filter})" if policy_filter else ""
    answer_prompt = (
        "You are a SETU policy information assistant.\n"
        "Answer the question using ONLY the policy excerpts below.\n"
        "Always name the policy document (shown as [Source: ...]) in your answer.\n"
        "If the policy excerpts are relevant but do not give a specific number or date, "
        "explain what the policy DOES say (e.g. 'entitlements are set in individual contracts' "
        "or 'refer to the relevant circular') — do NOT say the information is absent.\n"
        "Only say 'This is not covered in the retrieved policies.' if the excerpts are "
        "completely unrelated to the question.\n\n"
        f"Policy Excerpts{filter_note}:\n{context}\n\n"
        f"Question: {query}\n\n"
        "Answer (2-4 sentences, cite the source policy):"
    )
    initial_answer = llm.invoke(answer_prompt).content

    # Reflection pass: verify answer is grounded in the retrieved context
    return _reflect_and_verify(query, context, initial_answer, llm)


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
