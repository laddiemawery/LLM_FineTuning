"""Extract structured text from EPUB files."""

import json
import sys
from pathlib import Path

import ebooklib
from bs4 import BeautifulSoup
from ebooklib import epub

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from scripts.utils.text_cleaning import TextCleaner


def extract_epub(epub_path: Path, source_id: str) -> dict:
    """Extract text content from an EPUB file.

    Returns a dict with source metadata and a list of sections.
    """
    book = epub.read_epub(str(epub_path))
    sections = []

    for item in book.get_items():
        if item.get_type() != ebooklib.ITEM_DOCUMENT:
            continue

        soup = BeautifulSoup(item.get_content(), "html.parser")

        # Try to extract a title from headings
        title = ""
        for tag in ["h1", "h2", "h3"]:
            heading = soup.find(tag)
            if heading:
                title = heading.get_text(strip=True)
                break

        if not title:
            title = item.get_name()

        # Extract text
        raw_text = soup.get_text(separator="\n")
        cleaned = TextCleaner.full_clean(raw_text)

        if not TextCleaner.is_meaningful_text(cleaned):
            continue

        sections.append({
            "title": title,
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
