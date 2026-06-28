"""
Step 0 — Convert all 48 SETU PDFs to JSON (run once, ever).

    python build_dataset.py

Output: dataset/<doc_name>.json  (one file per PDF)

Each JSON contains:
  {
    "doc_name": "AcademicIntegrityPolicy",
    "source":   "AcademicIntegrityPolicy.pdf",
    "type":     "policy",
    "content":  "# Academic Integrity Policy\\n\\n..."
  }

After this runs, commit the dataset/ folder to git.
Subsequent ingest.py runs read from JSON — no Docling re-parse needed.
"""
import json
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

PDF_DIR     = Path(__file__).parent.parent / "All Data"
DATASET_DIR = Path(__file__).parent / "dataset"


def build(pdf_dir: Path = PDF_DIR, dataset_dir: Path = DATASET_DIR) -> int:
    from docling.document_converter import DocumentConverter

    pdf_files = sorted(pdf_dir.glob("*.pdf"))
    if not pdf_files:
        raise FileNotFoundError(f"No PDFs found in {pdf_dir}")

    dataset_dir.mkdir(exist_ok=True)

    print(f"[INFO] Found {len(pdf_files)} PDFs")
    print("[INFO] Initialising Docling (first run downloads AI models ~500MB)...\n")

    converter = DocumentConverter()
    written   = 0
    skipped   = 0

    for i, pdf_path in enumerate(pdf_files, 1):
        out_path = dataset_dir / f"{pdf_path.stem}.json"

        if out_path.exists():
            print(f"[{i:02d}/{len(pdf_files)}] SKIP (already exists): {pdf_path.name}")
            skipped += 1
            continue

        print(f"[{i:02d}/{len(pdf_files)}] Parsing: {pdf_path.name}")
        markdown = None

        # Primary: Docling (structure-aware: headings, tables, lists)
        try:
            result   = converter.convert(str(pdf_path))
            markdown = result.document.export_to_markdown()
            parser   = "docling"
            # Docling sometimes returns only image placeholders for certain PDFs;
            # treat content under 300 chars as a failed extraction
            if len(markdown.strip()) < 300:
                print(f"  [WARN] Docling returned too little content ({len(markdown)} chars), falling back to PyMuPDF...")
                markdown = None
        except Exception as exc:
            print(f"  [WARN] Docling failed ({exc}), falling back to PyMuPDF...")

        # Fallback: PyMuPDF (plain text, no layout AI)
        if not markdown:
            try:
                import fitz
                doc   = fitz.open(str(pdf_path))
                pages = [page.get_text() for page in doc]
                doc.close()
                markdown = "\n\n".join(p.strip() for p in pages if p.strip())
                parser   = "pymupdf"
            except Exception as exc2:
                print(f"  [ERROR] PyMuPDF also failed: {exc2}")

        if markdown:
            record = {
                "doc_name": pdf_path.stem,
                "source":   pdf_path.name,
                "type":     "policy",
                "parser":   parser,
                "content":  markdown,
            }
            out_path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"         -> {len(markdown):,} chars  [{parser}]  saved to {out_path.name}")
            written += 1
        else:
            print(f"  [ERROR] Could not extract text from {pdf_path.name} — skipping")

    print(f"\n[OK] Done. {written} new JSONs written, {skipped} skipped.")
    print(f"[OK] Dataset at: {dataset_dir}")
    print("\nNext step: python ingest.py")
    return written


if __name__ == "__main__":
    build()
