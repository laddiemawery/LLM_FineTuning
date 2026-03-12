"""Extract structured text from HTML files and web articles."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from scripts.utils.text_cleaning import TextCleaner


def extract_html(html_path: Path, source_id: str) -> dict:
    """Extract text from an HTML file or saved web article.

    Uses trafilatura for article extraction (handles boilerplate removal),
    falls back to BeautifulSoup for simpler HTML files.
    """
    with open(html_path, "r", encoding="utf-8", errors="replace") as f:
        content = f.read()

    sections = []

    # Try trafilatura first (best for articles)
    try:
        import trafilatura

        result = trafilatura.extract(
            content,
            include_comments=False,
            include_tables=True,
            output_format="txt",
        )
        if result and TextCleaner.is_meaningful_text(result):
            # Try to get metadata
            metadata = trafilatura.extract(content, output_format="xml")
            title = ""
            if metadata:
                from bs4 import BeautifulSoup as BS
                xml_soup = BS(metadata, "xml")
                title_tag = xml_soup.find("title")
                if title_tag:
                    title = title_tag.get_text(strip=True)

            cleaned = TextCleaner.full_clean(result)
            sections.append({
                "title": title or html_path.stem,
                "text": cleaned,
                "metadata": {"content_type": "prose", "extraction_method": "trafilatura"},
            })

            return {
                "source_id": source_id,
                "source_type": "html",
                "content_type": "prose",
                "sections": sections,
            }
    except ImportError:
        pass

    # Fallback: BeautifulSoup
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(content, "html.parser")

    # Remove script, style, nav, footer elements
    for tag in soup.find_all(["script", "style", "nav", "footer", "header", "aside"]):
        tag.decompose()

    title = ""
    title_tag = soup.find("title")
    if title_tag:
        title = title_tag.get_text(strip=True)

    # Extract main content (try article, main, then body)
    main_content = soup.find("article") or soup.find("main") or soup.find("body")
    if main_content:
        text = main_content.get_text(separator="\n")
        cleaned = TextCleaner.full_clean(text)

        if TextCleaner.is_meaningful_text(cleaned):
            sections.append({
                "title": title or html_path.stem,
                "text": cleaned,
                "metadata": {"content_type": "prose", "extraction_method": "beautifulsoup"},
            })

    return {
        "source_id": source_id,
        "source_type": "html",
        "content_type": "prose",
        "sections": sections,
    }


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Extract text from HTML files")
    parser.add_argument("html_path", type=Path, help="Path to the HTML file")
    parser.add_argument("--source-id", required=True, help="Source identifier")
    parser.add_argument("--output-dir", type=Path, help="Output directory")
    args = parser.parse_args()

    from scripts.utils.config import Config

    output_dir = args.output_dir or Config.EXTRACTED_DIR / "articles"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Extracting: {args.html_path}")
    result = extract_html(args.html_path, args.source_id)
    print(f"  Found {len(result['sections'])} sections")

    output_path = output_dir / f"{args.source_id}.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"  Saved to: {output_path}")


if __name__ == "__main__":
    main()
