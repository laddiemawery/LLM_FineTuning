"""Split extracted text into semantic chunks for dataset generation."""

import json
import sys
from pathlib import Path

import tiktoken
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.utils.config import Config


def count_tokens(text: str, encoding_name: str = "cl100k_base") -> int:
    """Count tokens in text using tiktoken."""
    enc = tiktoken.get_encoding(encoding_name)
    return len(enc.encode(text))


def chunk_prose(
    text: str,
    max_tokens: int = 800,
    overlap_tokens: int = 100,
    encoding_name: str = "cl100k_base",
) -> list[str]:
    """Chunk prose text by sentences, respecting token limits with overlap."""
    import re

    sentences = re.split(r"(?<=[.!?])\s+", text)
    enc = tiktoken.get_encoding(encoding_name)

    chunks = []
    current_chunk = []
    current_tokens = 0

    for sentence in sentences:
        sent_tokens = len(enc.encode(sentence))

        if current_tokens + sent_tokens > max_tokens and current_chunk:
            chunks.append(" ".join(current_chunk))

            # Create overlap by keeping last few sentences
            overlap_chunk = []
            overlap_count = 0
            for s in reversed(current_chunk):
                s_tokens = len(enc.encode(s))
                if overlap_count + s_tokens > overlap_tokens:
                    break
                overlap_chunk.insert(0, s)
                overlap_count += s_tokens

            current_chunk = overlap_chunk
            current_tokens = overlap_count

        current_chunk.append(sentence)
        current_tokens += sent_tokens

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks


def chunk_tabular(sections: list[dict], max_sections_per_chunk: int = 5) -> list[str]:
    """Chunk tabular/training log data by grouping sessions together."""
    chunks = []
    for i in range(0, len(sections), max_sections_per_chunk):
        group = sections[i : i + max_sections_per_chunk]
        chunk_text = "\n\n".join(s["text"] for s in group)
        chunks.append(chunk_text)
    return chunks


def process_extracted_file(file_path: Path, max_tokens: int = 800) -> list[dict]:
    """Process a single extracted JSON file into chunks."""
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    source_id = data["source_id"]
    content_type = data.get("content_type", "prose")
    sections = data.get("sections", [])

    all_chunks = []

    if content_type == "tabular":
        # Group tabular sections into chunks
        raw_chunks = chunk_tabular(sections)
        for i, chunk_text in enumerate(raw_chunks):
            all_chunks.append({
                "chunk_id": f"{source_id}_chunk_{i:04d}",
                "source_id": source_id,
                "content_type": "tabular",
                "text": chunk_text,
                "token_count": count_tokens(chunk_text),
                "topics": [],
            })
    else:
        # Chunk each prose section
        chunk_idx = 0
        for section in sections:
            text = section.get("text", "")
            if not text.strip():
                continue

            section_chunks = chunk_prose(text, max_tokens=max_tokens)
            for chunk_text in section_chunks:
                all_chunks.append({
                    "chunk_id": f"{source_id}_chunk_{chunk_idx:04d}",
                    "source_id": source_id,
                    "content_type": section.get("metadata", {}).get("content_type", "prose"),
                    "section_title": section.get("title", ""),
                    "text": chunk_text,
                    "token_count": count_tokens(chunk_text),
                    "topics": [],
                })
                chunk_idx += 1

    return all_chunks


def run_chunking(max_tokens: int = 800):
    """Process all extracted files into chunks."""
    config = Config()
    config.ensure_directories()

    extracted_dirs = [
        config.EXTRACTED_DIR / "textbooks",
        config.EXTRACTED_DIR / "articles",
        config.EXTRACTED_DIR / "training_logs",
    ]

    all_chunks = []

    for ext_dir in extracted_dirs:
        if not ext_dir.exists():
            continue
        json_files = sorted(ext_dir.glob("*.json"))
        for json_file in tqdm(json_files, desc=f"Chunking {ext_dir.name}"):
            chunks = process_extracted_file(json_file, max_tokens=max_tokens)
            all_chunks.extend(chunks)
            print(f"  {json_file.stem}: {len(chunks)} chunks")

    # Save all chunks to a single JSONL file
    output_path = config.CHUNKS_DIR / "all_chunks.jsonl"
    with open(output_path, "w", encoding="utf-8") as f:
        for chunk in all_chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    print(f"\nTotal: {len(all_chunks)} chunks saved to {output_path}")

    # Print stats
    total_tokens = sum(c["token_count"] for c in all_chunks)
    avg_tokens = total_tokens / len(all_chunks) if all_chunks else 0
    print(f"Total tokens: {total_tokens:,}")
    print(f"Average tokens per chunk: {avg_tokens:.0f}")

    return all_chunks


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Chunk extracted text for generation")
    parser.add_argument("--max-tokens", type=int, default=800, help="Max tokens per chunk")
    args = parser.parse_args()

    run_chunking(max_tokens=args.max_tokens)


if __name__ == "__main__":
    main()
