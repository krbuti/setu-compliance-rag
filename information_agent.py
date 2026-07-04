"""
Information agent — vectorless retrieval for SETU compliance policies.

Pipeline (BM25-only — no embeddings, no vector DB):
  1. Self-query filter   — if query names a specific policy, boost BM25 results from it
  2. BM25 retrieval      — keyword search over 2,584 chunks loaded from chunk_store.json
  3. Cross-encoder rerank — local sentence-transformer model, ~50ms, no LLM call
  4. Generate answer      — grounded, cites source policy names

Why vectorless?
  - BM25 catches exact legal/regulatory terms ("GDPR", "Article 6", "Garda vetting")
    that semantic embeddings also match — for policy queries the vocabulary overlap is high
  - Cross-encoder reranking is where precision comes from, not the retriever
  - Removes Jina AI API dependency (~300ms latency per query)
  - Removes 16 MB ChromaDB file from Docker image (replaced by 3 MB chunk_store.json)
  - No JINA_API_KEY required — fully self-contained on Groq alone
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever
from config.llm_config import get_llm

CHUNK_STORE = Path(__file__).parent / "chunk_store.json"

_FAKE_CODE_RE = re.compile(r'\bSETU-[A-Z]{2,}-\d+\b|\bSETU-\d{3,}\b', re.IGNORECASE)

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
]

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


def load_chunk_store() -> BM25Retriever:
    """Load chunks from chunk_store.json and build BM25 index."""
    global _bm25
    if _bm25 is not None:
        return _bm25
    if not CHUNK_STORE.exists():
        raise RuntimeError(
            f"chunk_store.json not found at {CHUNK_STORE}. "
            "Run: python export_chunks.py"
        )
    print(f"[INFO] Loading chunk store from {CHUNK_STORE} ...")
    with open(CHUNK_STORE, encoding="utf-8") as f:
        raw = json.load(f)
    docs = [Document(page_content=c["text"], metadata=c["metadata"]) for c in raw]
    _bm25 = BM25Retriever.from_documents(docs, k=12)
    print(f"[OK] BM25 index ready ({len(docs)} chunks, no embeddings)")
    _load_reranker()
    return _bm25


def _detect_policy_filter(query: str) -> Optional[str]:
    q = query.lower()
    for name in KNOWN_POLICIES:
        if name.lower() in q:
            return name
    return None


def _bm25_retrieve(query: str, top_n: int = 16) -> list[Document]:
    """Pure BM25 retrieval — wider k so the cross-encoder has enough candidates."""
    if _bm25 is None:
        raise RuntimeError("Chunk store not loaded. Call load_chunk_store() first.")
    _bm25.k = top_n
    return _bm25.invoke(query)


def _cross_encode_rerank(
    query: str, candidates: list[Document], top_k: int = 5
) -> list[Document]:
    if _reranker is None:
        return candidates[:top_k]
    pairs  = [(query, doc.page_content[:512]) for doc in candidates]
    scores = _reranker.predict(pairs)
    ranked = sorted(zip(scores, candidates), key=lambda x: x[0], reverse=True)
    return [doc for _, doc in ranked[:top_k]]


def _reflect_and_verify(query: str, context: str, initial_answer: str, llm) -> str:
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
        "1. It mentions a policy reference code (e.g. SETU-001, SETU-HR-002) "
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
            revised = re.sub(r'\b[\w.+-]+@[\w.-]+\.\w+\b', '[contact SETU directly]', revised)
            print("[INFO] Reflection: unsupported claims removed from answer")
            return revised
        else:
            return initial_answer
    except Exception as exc:
        print(f"[WARN] Reflection step failed ({exc}), returning original answer")
        return initial_answer


def get_information(query: str) -> str:
    if _FAKE_CODE_RE.search(query):
        return (
            "SETU does not use numeric policy reference codes. "
            "Policy documents are identified by their full title (e.g. 'Sick Leave Policy', "
            "'Parental Leave Policy'). Please search the SETU policy library by topic name, "
            "or contact HR with the specific subject area you need."
        )

    llm           = get_llm()
    policy_filter = _detect_policy_filter(query)

    candidates = _bm25_retrieve(query, top_n=16)

    # Deduplicate
    seen, unique = set(), []
    for doc in candidates:
        key = doc.page_content[:120]
        if key not in seen:
            seen.add(key)
            unique.append(doc)

    top5 = _cross_encode_rerank(query, unique, top_k=5)

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

    _needs_reflect = re.search(
        r'@\w+\.\w+'
        r'|SETU-[A-Z]{2,}-\d+'
        r'|\bSETU-\d{3,}\b',
        initial_answer, re.IGNORECASE
    )
    if _needs_reflect:
        return _reflect_and_verify(query, context, initial_answer, llm)
    return initial_answer


if __name__ == "__main__":
    load_chunk_store()
    questions = [
        "What is SETU's data retention policy for student records?",
        "What are the grounds for an academic integrity investigation?",
        "What does the Gen AI policy say about staff using AI tools?",
        "How many weeks of maternity leave is a SETU staff member entitled to?",
    ]
    for q in questions:
        print(f"Q: {q}")
        print(f"A: {get_information(q)}")
        print("-" * 60)
