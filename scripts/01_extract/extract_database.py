"""Extract and narrate data from database files (SQLite, JSON exports)."""

import json
import sqlite3
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from scripts.utils.text_cleaning import TextCleaner
from scripts.utils.training_log_parser import TrainingLogParser


def extract_database(db_path: Path, source_id: str) -> dict:
    """Extract data from a database file.

    Supports: .sqlite, .db, .json (structured exports)
    """
    suffix = db_path.suffix.lower()

    if suffix in (".sqlite", ".db", ".sqlite3"):
        return _extract_sqlite(db_path, source_id)
    elif suffix == ".json":
        return _extract_json(db_path, source_id)
    else:
        raise ValueError(f"Unsupported database type: {suffix}")


def _extract_sqlite(db_path: Path, source_id: str) -> dict:
    """Extract all tables from a SQLite database."""
    conn = sqlite3.connect(str(db_path))
    sections = []

    # Get all table names
    cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = [row[0] for row in cursor.fetchall()]

    for table in tables:
        df = pd.read_sql_query(f"SELECT * FROM [{table}]", conn)
        if df.empty:
            continue

        # Try to narrate as training data
        narrated = TrainingLogParser.dataframe_to_narratives(df, group_by_date=True)
        if narrated:
            for section in narrated:
                section["metadata"]["table_name"] = table
            sections.extend(narrated)
        else:
            # Fallback to text description
            text = _describe_table(table, df)
            if TextCleaner.is_meaningful_text(text, min_words=5):
                sections.append({
                    "title": f"Table: {table}",
                    "text": text,
                    "metadata": {
                        "content_type": "tabular",
                        "table_name": table,
                        "row_count": len(df),
                    },
                })

    conn.close()

    return {
        "source_id": source_id,
        "source_type": "database",
        "content_type": "tabular",
        "sections": sections,
    }


def _extract_json(json_path: Path, source_id: str) -> dict:
    """Extract data from a JSON file (array of records or nested structure)."""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    sections = []

    if isinstance(data, list):
        # Array of records
        df = pd.DataFrame(data)
        narrated = TrainingLogParser.dataframe_to_narratives(df, group_by_date=True)
        if narrated:
            sections.extend(narrated)
        else:
            text = _describe_records(data)
            if TextCleaner.is_meaningful_text(text, min_words=5):
                sections.append({
                    "title": json_path.stem,
                    "text": text,
                    "metadata": {"content_type": "tabular", "record_count": len(data)},
                })

    elif isinstance(data, dict):
        # Nested structure - try to find arrays within
        for key, value in data.items():
            if isinstance(value, list) and value and isinstance(value[0], dict):
                df = pd.DataFrame(value)
                narrated = TrainingLogParser.dataframe_to_narratives(df, group_by_date=True)
                if narrated:
                    for section in narrated:
                        section["metadata"]["json_key"] = key
                    sections.extend(narrated)
            elif isinstance(value, str) and TextCleaner.is_meaningful_text(value):
                sections.append({
                    "title": key,
                    "text": TextCleaner.full_clean(value),
                    "metadata": {"content_type": "prose", "json_key": key},
                })

    return {
        "source_id": source_id,
        "source_type": "database",
        "content_type": "mixed",
        "sections": sections,
    }


def _describe_table(table_name: str, df: pd.DataFrame) -> str:
    """Create a text description of a database table."""
    lines = [
        f"Database table '{table_name}' with {len(df)} rows and columns: {', '.join(df.columns.tolist())}.",
        "",
    ]
    for _, row in df.head(20).iterrows():
        parts = [f"{col}: {val}" for col, val in row.items() if pd.notna(val)]
        lines.append(" | ".join(parts))
    return "\n".join(lines)


def _describe_records(records: list[dict]) -> str:
    """Create a text description of a list of records."""
    if not records:
        return ""
    keys = list(records[0].keys())
    lines = [f"Data with {len(records)} records. Fields: {', '.join(keys)}.", ""]
    for record in records[:20]:
        parts = [f"{k}: {v}" for k, v in record.items() if v is not None]
        lines.append(" | ".join(parts))
    return "\n".join(lines)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Extract data from database files")
    parser.add_argument("db_path", type=Path, help="Path to database file")
    parser.add_argument("--source-id", required=True, help="Source identifier")
    parser.add_argument("--output-dir", type=Path, help="Output directory")
    args = parser.parse_args()

    from scripts.utils.config import Config

    output_dir = args.output_dir or Config.EXTRACTED_DIR / "training_logs"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Extracting: {args.db_path}")
    result = extract_database(args.db_path, args.source_id)
    print(f"  Found {len(result['sections'])} sections")

    output_path = output_dir / f"{args.source_id}.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"  Saved to: {output_path}")


if __name__ == "__main__":
    main()
