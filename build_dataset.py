"""
Step 0 — Convert all 48 SETU PDFs to JSON (run once, ever).

    python build_dataset.py           # skip already-extracted files
    python build_dataset.py --force   # re-extract every PDF

Output: dataset/<doc_name>.json  (one file per PDF)

Each JSON contains:
  {
    "doc_name": "AcademicIntegrityPolicy",
    "source":   "AcademicIntegrityPolicy.pdf",
    "type":     "policy",
    "parser":   "docling" | "pymupdf",
    "content":  "# Academic Integrity Policy\\n\\n..."
  }

Tables are exported as markdown tables (| col | col |) by Docling so that
the downstream table-aware chunker in data_preprocessing.py can keep them
atomic and prepend heading context.

After this runs, commit the dataset/ folder to git.
Subsequent ingest.py runs read from JSON — no Docling re-parse needed.
"""
import argparse
import json
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

PDF_DIR     = Path(__file__).parent.parent / "All Data"
DATASET_DIR = Path(__file__).parent / "dataset"


def _docling_to_markdown(pdf_path: Path) -> str | None:
    """Parse with Docling and return clean markdown, or None on failure."""
    try:
        from docling.document_converter import DocumentConverter
        converter = DocumentConverter()
        result    = converter.convert(str(pdf_path))
        doc       = result.document

        # export_to_markdown() renders tables as | col | rows |
        # strip image placeholders that add noise without content
        markdown = doc.export_to_markdown(image_placeholder="")

        if len(markdown.strip()) < 300:
            print(
                f"  [WARN] Docling returned too little content "
                f"({len(markdown)} chars), falling back to PyMuPDF..."
            )
            return None

        return markdown

    except Exception as exc:
        print(f"  [WARN] Docling failed ({exc}), falling back to PyMuPDF...")
        return None


def _pymupdf_to_markdown(pdf_path: Path) -> str | None:
    """Plain-text extraction via PyMuPDF as a fallback."""
    try:
        import fitz  # PyMuPDF
        doc   = fitz.open(str(pdf_path))
        pages = [page.get_text() for page in doc]
        doc.close()
        return "\n\n".join(p.strip() for p in pages if p.strip())
    except Exception as exc:
        print(f"  [ERROR] PyMuPDF also failed: {exc}")
        return None


def build(
    pdf_dir: Path = PDF_DIR,
    dataset_dir: Path = DATASET_DIR,
    force: bool = False,
) -> int:
    pdf_files = sorted(pdf_dir.glob("*.pdf"))
    if not pdf_files:
        raise FileNotFoundError(f"No PDFs found in {pdf_dir}")

    dataset_dir.mkdir(exist_ok=True)

    print(f"[INFO] Found {len(pdf_files)} PDFs  (force={force})")
    print("[INFO] Initialising Docling (first run downloads AI models ~500 MB)...\n")

    written = skipped = 0

    for i, pdf_path in enumerate(pdf_files, 1):
        out_path = dataset_dir / f"{pdf_path.stem}.json"

        if out_path.exists() and not force:
            print(f"[{i:02d}/{len(pdf_files)}] SKIP (exists): {pdf_path.name}")
            skipped += 1
            continue

        print(f"[{i:02d}/{len(pdf_files)}] Parsing: {pdf_path.name}")

        markdown = _docling_to_markdown(pdf_path)
        parser   = "docling"

        if not markdown:
            markdown = _pymupdf_to_markdown(pdf_path)
            parser   = "pymupdf"

        if markdown:
            record = {
                "doc_name": pdf_path.stem,
                "source":   pdf_path.name,
                "type":     "policy",
                "parser":   parser,
                "content":  markdown,
            }
            out_path.write_text(
                json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            print(
                f"         -> {len(markdown):,} chars  [{parser}]  "
                f"saved to {out_path.name}"
            )
            written += 1
        else:
            print(f"  [ERROR] Could not extract {pdf_path.name} — skipping")

    print(f"\n[OK] Done. {written} new/updated JSONs written, {skipped} skipped.")
    print(f"[OK] Dataset at: {dataset_dir}")
    print("\nNext step: python ingest.py")
    return written


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract SETU PDFs to JSON dataset")
    parser.add_argument(
        "--force", action="store_true",
        help="Re-extract even if the JSON already exists"
    )
    args = parser.parse_args()
    build(force=args.force)
