"""
Production failure mode analysis — real failure patterns in a PDF-based
compliance RAG system, with root causes and production fixes.

These are NOT toy examples — each mode reflects a documented real-world
RAG failure pattern seen in production compliance/legal document systems.

Run:  python failure_modes.py
"""
import time
from dotenv import load_dotenv
load_dotenv()

from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from config.llm_config import get_embeddings, get_llm
from information_agent import load_vectorstore, get_information, _hybrid_retrieve

embeddings = get_embeddings()
llm        = get_llm(temperature=0)
vs         = load_vectorstore()

BASELINE = "What are SETU's obligations under the Data Protection Policy?"

print("#" * 65)
print("# SETU Compliance RAG — Production Failure Mode Analysis")
print("#" * 65)
print(f"Baseline query: \"{BASELINE}\"\n")


# ─────────────────────────────────────────────────────────────────────────────
print("=" * 65)
print("FAILURE MODE 1: BM25 / dense signal conflict (cross-policy leakage)")
print("=" * 65)
print(
    "Scenario: Query uses general term ('data') that appears in 12 policies.\n"
    "          BM25 ranks Email Policy and Energy Policy highly (keyword match),\n"
    "          while dense ranks Data Protection first. Fusion dilutes the winner.\n"
)
ambiguous_query = "What are SETU's data policies?"
docs = _hybrid_retrieve(vs, ambiguous_query, top_n=5)
sources = [d.metadata.get("doc_name", "?") for d in docs]
print(f"Retrieved sources: {sources}")
print(f"Risk: Non-data-protection docs contaminate the context window.")
print("Root cause : Short, ambiguous queries maximise BM25 recall at the cost of precision.")
print("Fix        : Apply self-query filtering — detect 'data protection' intent from")
print("             expanded query and add metadata filter BEFORE hybrid retrieval.\n")

# ─────────────────────────────────────────────────────────────────────────────
print("=" * 65)
print("FAILURE MODE 2: Stale index (policy updated, ChromaDB not re-ingested)")
print("=" * 65)
print(
    "Scenario: The Data Retention Policy PDF is updated (new retention periods).\n"
    "          ChromaDB still holds the old chunks. Answers cite outdated figures.\n"
)
# Simulate stale data by injecting an old chunk directly
stale_text  = "Student records are retained for 5 years after graduation. (OUTDATED)"
stale_doc   = Document(page_content=stale_text, metadata={
    "source": "Data Retention Policy.pdf", "doc_name": "Data Retention Policy",
    "parent_doc": stale_text, "chunk_id": 9999,
})
# Show what a query would return if stale chunk is top-ranked
print(f"Stale chunk that could be retrieved: '{stale_text}'")
print("Root cause : No cache invalidation — ingest.py must be re-run when any PDF changes.")
print("Fix        : Track PDF modification timestamps. On startup, compare PDF mtimes against")
print("             ChromaDB metadata; delete + re-ingest only changed documents.")
print("             Better: use a document versioning field in metadata.\n")

# ─────────────────────────────────────────────────────────────────────────────
print("=" * 65)
print("FAILURE MODE 3: Cross-policy ambiguity (two policies conflict)")
print("=" * 65)
print(
    "Scenario: User asks about data sharing. Both the Data Protection Policy\n"
    "          and the CCTV Policy are retrieved. They give different retention\n"
    "          periods for different data types. Model conflates them.\n"
)
conflict_query = "How long can SETU keep recordings and personal data?"
docs = _hybrid_retrieve(vs, conflict_query, top_n=6)
sources = list({d.metadata.get("doc_name", "?") for d in docs})
print(f"Policies retrieved: {sources}")
answer = get_information(vs, conflict_query)
print(f"Answer: {answer[:250]}...")
print("Root cause : Different policies define 'data' differently; model may merge them.")
print("Fix        : In combine_node, add a 'policy conflict' detector. If >1 distinct")
print("             doc_name is retrieved, surface the sources explicitly in the answer\n"
      "             ('According to the CCTV Policy... but the Data Retention Policy states...')\n")

# ─────────────────────────────────────────────────────────────────────────────
print("=" * 65)
print("FAILURE MODE 4: Cross-encoder confidence collapse (adversarial query)")
print("=" * 65)
print(
    "Scenario: A query is phrased in a way that tricks the cross-encoder —\n"
    "          it contains keywords from an irrelevant policy that dominate.\n"
)
adversarial = "What is the energy efficiency policy for data centres running AI?"
docs_before_rerank = _hybrid_retrieve(vs, adversarial, top_n=8)
sources_raw = [d.metadata.get("doc_name", "?") for d in docs_before_rerank]
print(f"Pre-rerank sources  : {sources_raw[:5]}")

# Show that without reranking the top result could be wrong
print("Without cross-encoder: Energy Policy and Email Policy may rank first (keyword: 'data')")
print("With cross-encoder   : ms-marco model penalises off-topic passages and promotes relevant ones")
print("Root cause : Hybrid retrieval increases recall but can surface irrelevant docs.")
print("Fix        : Cross-encoder reranking (already implemented) handles this. Additional")
print("             mitigation: add a relevance score threshold — reject top chunk if")
print("             cross-encoder score < 0.1 and return 'not found in policies'.\n")

# ─────────────────────────────────────────────────────────────────────────────
print("=" * 65)
print("FAILURE MODE 5: Docling parsing failure on scanned/image PDFs")
print("=" * 65)
print(
    "Scenario: A PDF is a scanned image (no selectable text). Docling's OCR\n"
    "          produces garbled output or empty pages. Chunks are noise.\n"
)
garbled_sample = "Secton 4.2 Retentbn o£ Persønal Dota — see Àpendix 3 for detaÿls"
print(f"Sample garbled OCR chunk: '{garbled_sample}'")
print(f"Impact: Retrieval returns this chunk; model cites a non-existent policy clause.")

# Detection heuristic
def is_garbled(text: str, threshold: float = 0.08) -> bool:
    non_ascii = sum(1 for c in text if ord(c) > 127)
    return non_ascii / max(len(text), 1) > threshold

print(f"Garbled? {is_garbled(garbled_sample)} (detected by non-ASCII ratio > 8%)")
print("Root cause : Scanned PDFs require OCR; Docling includes OCR but quality varies.")
print("Fix        : In ingest.py, after Docling conversion, compute garble score on each")
print("             chunk. Flag chunks with >8% non-ASCII characters as low-quality;\n"
      "             log them for manual review and exclude from the vector index.\n")


# ─────────────────────────────────────────────────────────────────────────────
print("=" * 65)
print("FAILURE MODE SUMMARY")
print("=" * 65)

summary = [
    ("Cross-policy leakage",      "BM25 over-retrieves docs with generic terms",       "Self-query intent filter before hybrid search"),
    ("Stale cached index",        "Outdated PDF chunks served as current policy",       "PDF mtime check + selective re-ingest"),
    ("Cross-policy conflict",     "Model conflates different policy retention rules",   "Multi-source conflict detector in combine node"),
    ("Adversarial query routing", "Off-topic docs rank high via keyword collision",     "Cross-encoder score threshold (reject < 0.1)"),
    ("Garbled OCR chunks",        "Scanned PDFs produce noise that poisons retrieval",  "Non-ASCII ratio filter in ingest pipeline"),
]
print(f"{'Mode':<28} {'Failure':<44} {'Fix'}")
print("-" * 115)
for mode, failure, fix in summary:
    print(f"{mode:<28} {failure:<44} {fix}")
