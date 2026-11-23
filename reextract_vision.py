"""
Re-extract thin policy PDFs using PyMuPDF direct text extraction (free, local).

Scans dataset/*.json for content < CHAR_THRESHOLD, re-parses each PDF with
PyMuPDF's built-in text extraction (which handles embedded text that Docling
missed), and writes the updated JSON with parser="pymupdf".

Usage:
    python reextract_vision.py
    python reextract_vision.py --threshold 5000
    python reextract_vision.py --dry-run
"""
import argparse
import json
import sys
from pathlib import Path

import fitz  # PyMuPDF

CHAR_THRESHOLD = 3_500
DATASET_DIR = Path(__file__).parent / "dataset"
PDF_DIR = Path(__file__).parent.parent / "All Data"


def extract_pdf_text(pdf_path: Path) -> str:
    doc = fitz.open(pdf_path)
    pages = []
    for page in doc:
        text = page.get_text("text").strip()
        if text:
            pages.append(text)
    doc.close()
    return "\n\n".join(pages)


def find_thin_docs(threshold: int) -> list[tuple[Path, dict, int]]:
    results = []
    for jf in sorted(DATASET_DIR.glob("*.json")):
        data = json.loads(jf.read_text(encoding="utf-8"))
        clen = len(data.get("content", ""))
        if clen < threshold:
            results.append((jf, data, clen))
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--threshold", type=int, default=CHAR_THRESHOLD,
                        help=f"Content length threshold (default: {CHAR_THRESHOLD})")
    parser.add_argument("--dry-run", action="store_true",
                        help="List thin docs without writing any files")
    args = parser.parse_args()

    thin = find_thin_docs(args.threshold)

    if not thin:
        print(f"No dataset JSONs below {args.threshold} chars. Nothing to do.")
        return

    print(f"Thin documents (< {args.threshold} chars): {len(thin)}\n")
    for jf, _, clen in thin:
        pdf_exists = (PDF_DIR / f"{jf.stem}.pdf").exists()
        status = "OK" if pdf_exists else "NO PDF"
        print(f"  [{status}] {jf.stem[:55]:55s} {clen:,} chars")

    if args.dry_run:
        return

    print()
    processed = skipped = 0

    for jf, data, before_len in thin:
        pdf_path = PDF_DIR / f"{jf.stem}.pdf"
        if not pdf_path.exists():
            print(f"[SKIP]  {jf.stem} — PDF not found")
            skipped += 1
            continue

        try:
            text = extract_pdf_text(pdf_path)
        except Exception as exc:
            print(f"[ERROR] {jf.stem}: {exc}")
            skipped += 1
            continue

        after_len = len(text)
        gain = after_len - before_len
        print(f"[OK]  {jf.stem[:55]:55s} {before_len:,} -> {after_len:,} chars  (+{gain:,})")

        data["content"] = text
        data["parser"] = "pymupdf"
        jf.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        processed += 1

    print(f"\nDone. Processed: {processed}  Skipped: {skipped}")
    if processed:
        print("Next: python ingest.py --reset --semantic")


if __name__ == "__main__":
    main()
