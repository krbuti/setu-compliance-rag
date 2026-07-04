"""
Step 1 — Load dataset JSONs into ChromaDB.

Run AFTER build_dataset.py:
    python ingest.py

Pipeline:
  dataset/<doc>.json  ->  parent-child chunking  (child=256 chars for matching,
                                                   parent=1024 chars in metadata)
                      ->  Persistent ChromaDB collection

This is fast (seconds) because Docling already ran in build_dataset.py.
"""
import json
import shutil
import argparse
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.documents import Document

load_dotenv()

from config.llm_config import get_embeddings
from data_preprocessing import create_parent_child_chunks, create_semantic_chunks

DATASET_DIR = Path(__file__).parent / "dataset"
CHROMA_DIR  = Path(__file__).parent / "chroma_db"
COLLECTION  = "setu_compliance"


def load_dataset(dataset_dir: Path = DATASET_DIR) -> list[Document]:
    json_files = sorted(dataset_dir.glob("*.json"))
    if not json_files:
        raise FileNotFoundError(
            f"No JSON files found in {dataset_dir}\n"
            "Run 'python build_dataset.py' first."
        )

    docs = []
    for path in json_files:
        record = json.loads(path.read_text(encoding="utf-8"))
        docs.append(Document(
            page_content=record["content"],
            metadata={
                "source":   record["source"],
                "doc_name": record["doc_name"],
                "type":     record.get("type", "policy"),
            },
        ))
    return docs


def ingest(
    dataset_dir: Path = DATASET_DIR,
    chroma_dir: Path = CHROMA_DIR,
    semantic: bool = False,
    reset: bool = False,
) -> int:
    from langchain_community.vectorstores import Chroma

    if reset and chroma_dir.exists():
        print(f"[INFO] --reset: deleting existing ChromaDB at {chroma_dir}")
        shutil.rmtree(chroma_dir)

    raw_docs = load_dataset(dataset_dir)
    print(f"[INFO] Loaded {len(raw_docs)} documents from {dataset_dir}")

    embeddings = get_embeddings()

    if semantic:
        print("[INFO] Creating semantic (section-aware) chunks...")
        child_docs, _ = create_semantic_chunks(raw_docs, child_size=512, parent_size=1536)
    else:
        print("[INFO] Creating fixed-size parent-child chunks...")
        child_docs, _ = create_parent_child_chunks(raw_docs, child_size=256, parent_size=1024)

    child_docs = [c for c in child_docs if len(c["text"].strip()) >= 40]
    print(f"[INFO] Total child chunks to index: {len(child_docs)}")

    texts     = [c["text"]     for c in child_docs]
    metadatas = [c["metadata"] for c in child_docs]
    ids       = [str(i)        for i in range(len(child_docs))]

    print(f"[INFO] Writing to ChromaDB at {chroma_dir} ...")

    BATCH = 100

    # Resume support: if ChromaDB already exists, skip IDs already present
    vectorstore = Chroma(
        persist_directory=str(chroma_dir),
        embedding_function=embeddings,
        collection_name=COLLECTION,
    )
    existing_ids = set(vectorstore._collection.get(include=[])["ids"])
    if existing_ids:
        print(f"[INFO] Resuming — {len(existing_ids)} chunks already indexed, skipping them.")

    pending = [
        (t, m, id_)
        for t, m, id_ in zip(texts, metadatas, ids)
        if id_ not in existing_ids
    ]

    for start in range(0, len(pending), BATCH):
        batch = pending[start : start + BATCH]
        b_texts = [x[0] for x in batch]
        b_meta  = [x[1] for x in batch]
        b_ids   = [x[2] for x in batch]
        vectorstore.add_texts(texts=b_texts, metadatas=b_meta, ids=b_ids)
        done = min(start + BATCH, len(pending))
        print(f"  Indexed {len(existing_ids) + done}/{len(texts)} chunks", end="\r")

    print(f"\n[OK] Done. {len(texts)} chunks indexed from {len(raw_docs)} documents.")
    print(f"[OK] Vector store saved to {chroma_dir}")
    return len(texts)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest SETU policy documents into ChromaDB.")
    parser.add_argument(
        "--semantic", action="store_true",
        help="Use section-aware semantic chunking (512-char child, 1536-char parent) "
             "instead of fixed character-count splits.",
    )
    parser.add_argument(
        "--reset", action="store_true",
        help="Delete and recreate ChromaDB before ingesting. Required when changing chunk strategy.",
    )
    args = parser.parse_args()
    ingest(semantic=args.semantic, reset=args.reset)
