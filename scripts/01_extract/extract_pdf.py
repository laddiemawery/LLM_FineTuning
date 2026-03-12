"""Extract structured text from PDF files (text-based with OCR fallback)."""

import json
import sys
from pathlib import Path

import fitz  # pymupdf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from scripts.utils.text_cleaning import TextCleaner


def extract_pdf(pdf_path: Path, source_id: str, ocr_fallback: bool = True) -> dict:
    """Extract text from a PDF file.

    Uses pymupdf for text extraction. Falls back to OCR via pytesseract
    if a page has very little text (likely a scanned page).
    """
    doc = fitz.open(str(pdf_path))
    sections = []
    current_section = {"title": "", "text_parts": []}

    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text("text")

        # Check if page is mostly images (scanned) - fallback to OCR
        if ocr_fallback and len(text.strip()) < 50:
            text = _ocr_page(page)

        if not text.strip():
            continue

        cleaned = TextCleaner.full_clean(text)

        # Detect chapter/section breaks by looking for heading-like patterns
        lines = cleaned.split("\n")
        for line in lines:
            stripped = line.strip()
            if _is_heading(stripped):
                # Save previous section
                if current_section["text_parts"]:
                    full_text = "\n".join(current_section["text_parts"])
                    if TextCleaner.is_meaningful_text(full_text):
                        sections.append({
                            "title": current_section["title"] or f"Page {page_num + 1}",
                            "text": full_text,
                            "metadata": {
                                "start_page": page_num + 1,
                                "content_type": "prose",
                            },
                        })
                current_section = {"title": stripped, "text_parts": []}
            else:
                current_section["text_parts"].append(stripped)

    # Save last section
    if current_section["text_parts"]:
        full_text = "\n".join(current_section["text_parts"])
        if TextCleaner.is_meaningful_text(full_text):
            sections.append({
                "title": current_section["title"] or "Final Section",
                "text": full_text,
                "metadata": {"content_type": "prose"},
            })

    doc.close()

    return {
        "source_id": source_id,
        "source_type": "pdf",
        "content_type": "prose",
        "sections": sections,
    }


def _is_heading(text: str) -> bool:
    """Heuristic to detect if a line is a section heading."""
    if not text or len(text) > 100:
        return False
    if text.isupper() and len(text.split()) <= 8:
        return True
    import re
    if re.match(r"^(Chapter|CHAPTER|Section|SECTION|Part|PART)\s+\d+", text):
        return True
    if re.match(r"^\d+\.\d*\s+[A-Z]", text):
        return True
    return False


def _ocr_page(page) -> str:
    """OCR a PDF page using pytesseract."""
    try:
        import pytesseract
        from PIL import Image
        import io

        # Render page to image at 300 DPI
        pix = page.get_pixmap(dpi=300)
        img = Image.open(io.BytesIO(pix.tobytes("png")))
        text = pytesseract.image_to_string(img)
        return text
    except ImportError:
        return ""


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Extract text from PDF files")
    parser.add_argument("pdf_path", type=Path, help="Path to the PDF file")
    parser.add_argument("--source-id", required=True, help="Source identifier")
    parser.add_argument("--output-dir", type=Path, help="Output directory")
    parser.add_argument("--no-ocr", action="store_true", help="Disable OCR fallback")
    args = parser.parse_args()

    from scripts.utils.config import Config

    output_dir = args.output_dir or Config.EXTRACTED_DIR / "textbooks"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Extracting: {args.pdf_path}")
    result = extract_pdf(args.pdf_path, args.source_id, ocr_fallback=not args.no_ocr)
    print(f"  Found {len(result['sections'])} sections")

    output_path = output_dir / f"{args.source_id}.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"  Saved to: {output_path}")


if __name__ == "__main__":
    main()
