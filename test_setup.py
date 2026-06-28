"""
Health check — run before anything else:  python test_setup.py
"""
import sys
from pathlib import Path

PDF_DIR = Path(__file__).parent.parent / "All Data"
CHROMA_DIR = Path(__file__).parent / "chroma_db"


def check(label, fn):
    try:
        result = fn()
        print(f"[OK] {label}" + (f": {result}" if result else ""))
        return True
    except Exception as exc:
        print(f"[FAIL] {label}: {exc}")
        return False


def main():
    ok = True

    ok &= check("Python version", lambda: sys.version)

    ok &= check(
        f"PDF directory exists ({PDF_DIR})",
        lambda: f"{len(list(PDF_DIR.glob('*.pdf')))} PDFs found",
    )

    ok &= check("docling importable", lambda: __import__("docling") and None)

    ok &= check("chromadb importable", lambda: __import__("chromadb") and None)

    ok &= check("langchain importable", lambda: __import__("langchain") and None)

    # LLM + embeddings factory
    from config.llm_config import get_embeddings, get_llm

    def _embeddings():
        emb = get_embeddings()
        result = emb.embed_query("test")
        return f"{len(result)} dimensions"

    ok &= check("Embeddings working", _embeddings)

    def _llm():
        llm = get_llm(temperature=0)
        resp = llm.invoke("Reply with: yes I am working")
        return resp.content.strip()[:40]

    ok &= check("LLM working", _llm)

    def _chroma():
        import chromadb
        client = chromadb.Client()
        col = client.create_collection("healthcheck")
        client.delete_collection("healthcheck")
        return "in-memory client OK"

    ok &= check("ChromaDB working", _chroma)

    print()
    if ok:
        print("[OK] All checks passed. Run: python ingest.py")
    else:
        print("[WARN] Some checks failed. Fix issues above before running ingest.py")


if __name__ == "__main__":
    main()
