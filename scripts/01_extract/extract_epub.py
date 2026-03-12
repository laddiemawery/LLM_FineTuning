"""Extract structured text from EPUB files."""

import json
import re
import sys
from pathlib import Path

import ebooklib
from bs4 import BeautifulSoup
from ebooklib import epub

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from scripts.utils.text_cleaning import TextCleaner


def _split_by_headings(text: str, min_section_words: int = 20) -> list[dict]:
    """Split a large block of text into sections based on heading-like lines.

    Detects headings as short lines (<=10 words) that are either title-cased,
    all-uppercase, or match common structural patterns (Domain, Chapter, etc.).
    """
    lines = text.split("\n")
    sections = []
    current_title = ""
    current_lines = []

    heading_pattern = re.compile(
        r"^(Scientific Foundations|Practical/Applied|Domain\s+\d|Chapter\s+\d|Section\s+\d|Part\s+\d)",
        re.IGNORECASE,
    )

    def _is_heading(line: str) -> bool:
        stripped = line.strip()
        if not stripped or len(stripped) < 3:
            return False
        words = stripped.split()
        if len(words) > 10:
            return False
        if heading_pattern.match(stripped):
            return True
        if stripped.isupper() and len(words) >= 2:
            return True
        if stripped.istitle() and len(words) >= 2 and len(stripped) < 60:
            return True
        return False

    def _flush():
        if current_lines:
            body = "\n".join(current_lines).strip()
            if len(body.split()) >= min_section_words:
                sections.append({
                    "title": current_title or "Untitled Section",
                    "text": body,
                })

    for line in lines:
        if _is_heading(line):
            _flush()
            current_title = line.strip()
            current_lines = []
        else:
            current_lines.append(line)

    _flush()
    return sections


def extract_epub(epub_path: Path, source_id: str) -> dict:
    """Extract text content from an EPUB file.

    Returns a dict with source metadata and a list of sections.
    If a document item yields a single large block, splits it by internal headings.
    """
    book = epub.read_epub(str(epub_path))
    sections = []

    for item in book.get_items():
        if item.get_type() != ebooklib.ITEM_DOCUMENT:
            continue

        soup = BeautifulSoup(item.get_content(), "html.parser")
        raw_text = soup.get_text(separator="\n")
        cleaned = TextCleaner.full_clean(raw_text)

        if not TextCleaner.is_meaningful_text(cleaned):
            continue

        # If the document is large (>5000 chars), try splitting by headings
        if len(cleaned) > 5000:
            sub_sections = _split_by_headings(cleaned)
            if len(sub_sections) > 1:
                for sub in sub_sections:
                    sections.append({
                        "title": sub["title"],
                        "text": sub["text"],
                        "metadata": {
                            "source_file": item.get_name(),
                            "content_type": "prose",
                        },
                    })
                continue

        # Fallback: use the whole document as one section
        title = ""
        for tag in ["h1", "h2", "h3"]:
            heading = soup.find(tag)
            if heading:
                title = heading.get_text(strip=True)
                break

        sections.append({
            "title": title or item.get_name(),
            "text": cleaned,
            "metadata": {
                "source_file": item.get_name(),
                "content_type": "prose",
            },
        })

    return {
        "source_id": source_id,
        "source_type": "epub",
        "content_type": "prose",
        "sections": sections,
    }


def main():
    """CLI entry point for standalone EPUB extraction."""
    import argparse

    parser = argparse.ArgumentParser(description="Extract text from EPUB files")
    parser.add_argument("epub_path", type=Path, help="Path to the EPUB file")
    parser.add_argument("--source-id", required=True, help="Source identifier")
    parser.add_argument("--output-dir", type=Path, help="Output directory")
    args = parser.parse_args()

    from scripts.utils.config import Config

    output_dir = args.output_dir or Config.EXTRACTED_DIR / "textbooks"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Extracting: {args.epub_path}")
    result = extract_epub(args.epub_path, args.source_id)
    print(f"  Found {len(result['sections'])} sections")

    output_path = output_dir / f"{args.source_id}.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"  Saved to: {output_path}")


if __name__ == "__main__":
    main()
