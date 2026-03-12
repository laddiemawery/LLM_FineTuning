"""Extract and narrate structured data from Excel/CSV/TSV files."""

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from scripts.utils.text_cleaning import TextCleaner
from scripts.utils.training_log_parser import TrainingLogParser


def extract_spreadsheet(file_path: Path, source_id: str, sheet_name: int | str = 0) -> dict:
    """Extract and narrate data from a spreadsheet file.

    Supports: .xlsx, .xls, .csv, .tsv
    Converts structured training data into natural language narratives.
    """
    df = _read_file(file_path, sheet_name)

    if df.empty:
        return {
            "source_id": source_id,
            "source_type": "spreadsheet",
            "content_type": "tabular",
            "sections": [],
        }

    # Clean the dataframe
    df = df.dropna(how="all")  # Remove fully empty rows
    df = df.dropna(axis=1, how="all")  # Remove fully empty columns

    # Convert to narrated sections using the training log parser
    sections = TrainingLogParser.dataframe_to_narratives(df, group_by_date=True)

    # If narration produced nothing, fall back to raw text representation
    if not sections:
        text = _dataframe_to_text(df)
        if TextCleaner.is_meaningful_text(text, min_words=5):
            sections.append({
                "title": file_path.stem,
                "text": text,
                "metadata": {"content_type": "tabular", "fallback": True},
            })

    return {
        "source_id": source_id,
        "source_type": "spreadsheet",
        "content_type": "tabular",
        "sections": sections,
    }


def _read_file(file_path: Path, sheet_name: int | str = 0) -> pd.DataFrame:
    """Read a spreadsheet file into a DataFrame."""
    suffix = file_path.suffix.lower()

    if suffix in (".xlsx", ".xls"):
        return pd.read_excel(file_path, sheet_name=sheet_name)
    elif suffix == ".csv":
        return pd.read_csv(file_path)
    elif suffix == ".tsv":
        return pd.read_csv(file_path, sep="\t")
    else:
        raise ValueError(f"Unsupported file type: {suffix}")


def _dataframe_to_text(df: pd.DataFrame) -> str:
    """Fallback: convert DataFrame to readable text format."""
    lines = []
    lines.append(f"Data with columns: {', '.join(df.columns.tolist())}")
    lines.append(f"Total rows: {len(df)}")
    lines.append("")

    for _, row in df.head(50).iterrows():  # Limit for sanity
        parts = [f"{col}: {val}" for col, val in row.items() if pd.notna(val)]
        lines.append(" | ".join(parts))

    return "\n".join(lines)


def extract_all_sheets(file_path: Path, source_id: str) -> dict:
    """Extract from all sheets in an Excel workbook."""
    if file_path.suffix.lower() not in (".xlsx", ".xls"):
        return extract_spreadsheet(file_path, source_id)

    xl = pd.ExcelFile(file_path)
    all_sections = []

    for sheet in xl.sheet_names:
        result = extract_spreadsheet(file_path, source_id, sheet_name=sheet)
        for section in result["sections"]:
            section["metadata"]["sheet_name"] = sheet
        all_sections.extend(result["sections"])

    return {
        "source_id": source_id,
        "source_type": "spreadsheet",
        "content_type": "tabular",
        "sections": all_sections,
    }


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Extract data from spreadsheet files")
    parser.add_argument("file_path", type=Path, help="Path to spreadsheet file")
    parser.add_argument("--source-id", required=True, help="Source identifier")
    parser.add_argument("--output-dir", type=Path, help="Output directory")
    parser.add_argument("--all-sheets", action="store_true", help="Extract all sheets")
    args = parser.parse_args()

    from scripts.utils.config import Config

    output_dir = args.output_dir or Config.EXTRACTED_DIR / "training_logs"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Extracting: {args.file_path}")
    if args.all_sheets:
        result = extract_all_sheets(args.file_path, args.source_id)
    else:
        result = extract_spreadsheet(args.file_path, args.source_id)
    print(f"  Found {len(result['sections'])} sections")

    output_path = output_dir / f"{args.source_id}.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"  Saved to: {output_path}")


if __name__ == "__main__":
    main()
