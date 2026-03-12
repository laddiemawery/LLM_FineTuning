"""Orchestrator: reads registry.yaml and routes each source to the correct extractor."""

import json
import sys
from pathlib import Path

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from scripts.utils.config import Config
from scripts.utils.source_registry import SourceRegistry, Source

from .extract_epub import extract_epub
from .extract_pdf import extract_pdf
from .extract_html import extract_html
from .extract_ocr import extract_ocr, extract_ocr_batch
from .extract_spreadsheet import extract_spreadsheet, extract_all_sheets
from .extract_database import extract_database


# Map source types to (extractor_function, output_subdir)
EXTRACTORS = {
    "epub": (extract_epub, "textbooks"),
    "pdf": (extract_pdf, "textbooks"),
    "html": (extract_html, "articles"),
    "image_ocr": (extract_ocr, "training_logs"),
    "spreadsheet": (extract_spreadsheet, "training_logs"),
    "csv": (extract_spreadsheet, "training_logs"),
    "xlsx": (extract_all_sheets, "training_logs"),
    "tsv": (extract_spreadsheet, "training_logs"),
    "database": (extract_database, "training_logs"),
    "sqlite": (extract_database, "training_logs"),
    "json_db": (extract_database, "training_logs"),
}


def extract_source(source: Source, config: Config) -> dict | None:
    """Extract text from a single source using the appropriate extractor."""
    if not source.exists:
        print(f"  WARNING: Source file not found: {source.full_path}")
        return None

    extractor_info = EXTRACTORS.get(source.type)
    if not extractor_info:
        print(f"  WARNING: No extractor for type '{source.type}'")
        return None

    extract_fn, _ = extractor_info
    return extract_fn(source.full_path, source.id)


def save_result(result: dict, source: Source, config: Config):
    """Save extraction result to the appropriate output directory."""
    _, subdir = EXTRACTORS[source.type]
    output_dir = config.EXTRACTED_DIR / subdir
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / f"{source.id}.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    return output_path


def run_extraction(source_id: str | None = None):
    """Run extraction for all sources or a specific source."""
    config = Config()
    config.ensure_directories()
    registry = SourceRegistry()

    # Validate registry
    errors = registry.validate()
    if errors:
        print("Registry validation warnings:")
        for err in errors:
            print(f"  - {err}")

    # Select sources to process
    if source_id:
        source = registry.get_by_id(source_id)
        if not source:
            print(f"Error: Source '{source_id}' not found in registry")
            return
        sources = [source]
    else:
        sources = registry.get_all()

    if not sources:
        print("No sources found in registry.")
        return

    print(f"Extracting {len(sources)} source(s)...\n")

    results_summary = []
    for source in tqdm(sources, desc="Extracting"):
        print(f"\n  [{source.type}] {source.id}: {source.full_path}")
        result = extract_source(source, config)

        if result:
            output_path = save_result(result, source, config)
            n_sections = len(result.get("sections", []))
            print(f"    -> {n_sections} sections saved to {output_path}")
            results_summary.append((source.id, n_sections))
        else:
            print(f"    -> SKIPPED")
            results_summary.append((source.id, 0))

    # Summary
    print(f"\n{'='*50}")
    print("Extraction Summary:")
    total_sections = 0
    for sid, count in results_summary:
        status = f"{count} sections" if count > 0 else "SKIPPED"
        print(f"  {sid}: {status}")
        total_sections += count
    print(f"\nTotal: {total_sections} sections from {len(results_summary)} source(s)")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Extract text from all registered sources")
    parser.add_argument("--source", type=str, help="Extract a specific source by ID")
    args = parser.parse_args()

    run_extraction(source_id=args.source)


if __name__ == "__main__":
    main()
