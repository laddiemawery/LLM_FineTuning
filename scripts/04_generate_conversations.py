"""Generate multi-turn conversation datasets from text chunks using the Claude API."""

import argparse
import json
import sys
from pathlib import Path

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.utils.config import Config
from scripts.utils.llm_client import LLMClient


def load_chunks(chunks_path: Path) -> list[dict]:
    """Load all chunks from the JSONL file."""
    chunks = []
    with open(chunks_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))
    return chunks


def get_existing_chunk_ids(output_path: Path) -> set[str]:
    """Read already-generated chunk IDs from the output file."""
    ids = set()
    if output_path.exists():
        with open(output_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        record = json.loads(line)
                        ids.add(record.get("chunk_id", ""))
                    except json.JSONDecodeError:
                        continue
    return ids


def generate_conversations(
    chunk: dict,
    llm: LLMClient,
    config: Config,
) -> list[dict]:
    """Generate conversation records for a single chunk."""
    content_type = chunk.get("content_type", "prose")
    text = chunk["text"]
    n = config.get_generation_count("conversations")

    system_prompt = config.get_prompt("conversation_system")

    if content_type == "tabular":
        instruction_template = config.get_prompt("training_log_conversation_instruction")
    else:
        instruction_template = config.get_prompt("conversation_instruction")

    user_prompt = instruction_template.format(n=n, text=text)

    raw_results = llm.generate_json(system_prompt, user_prompt)

    records = []
    for item in raw_results:
        messages = item.get("messages", [])
        if not messages:
            continue
        records.append({
            "messages": messages,
            "source_id": chunk["source_id"],
            "chunk_id": chunk["chunk_id"],
            "content_type": content_type,
        })

    return records


def main():
    parser = argparse.ArgumentParser(
        description="Generate multi-turn conversations from text chunks"
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        help="Only process N chunks (for testing)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip chunks that already have output",
    )
    args = parser.parse_args()

    config = Config()
    config.ensure_directories()

    chunks_path = config.CHUNKS_DIR / "all_chunks.jsonl"
    if not chunks_path.exists():
        print(f"Error: chunks file not found at {chunks_path}")
        print("Run 02_chunk_text.py first.")
        sys.exit(1)

    output_path = config.GENERATED_DIR / "conversations" / "conversations.jsonl"

    chunks = load_chunks(chunks_path)
    print(f"Loaded {len(chunks)} chunks from {chunks_path}")

    # Filter already-processed chunks when resuming
    if args.resume:
        existing_ids = get_existing_chunk_ids(output_path)
        chunks = [c for c in chunks if c["chunk_id"] not in existing_ids]
        print(f"Resuming: {len(existing_ids)} already done, {len(chunks)} remaining")

    # Limit to sample size if requested
    if args.sample is not None:
        chunks = chunks[: args.sample]
        print(f"Sampling {len(chunks)} chunks")

    if not chunks:
        print("No chunks to process.")
        return

    llm = LLMClient(config)

    total_generated = 0
    errors = 0

    with open(output_path, "a", encoding="utf-8") as out_f:
        for chunk in tqdm(chunks, desc="Generating conversations"):
            try:
                records = generate_conversations(chunk, llm, config)
                for record in records:
                    out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                total_generated += len(records)
            except Exception as e:
                errors += 1
                print(f"\nError on chunk {chunk['chunk_id']}: {e}")

    print(f"\nDone. Generated {total_generated} conversations, {errors} errors.")
    print(f"Output: {output_path}")


if __name__ == "__main__":
    main()
