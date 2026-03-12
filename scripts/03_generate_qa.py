"""Generate instruction/response QA pairs from text chunks using the Claude API."""

import argparse
import json
import sys
import time
from pathlib import Path

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.utils.config import Config
from scripts.utils.llm_client import LLMClient


def load_chunks(config: Config) -> list[dict]:
    """Load all chunks from the JSONL file."""
    chunks_path = config.CHUNKS_DIR / "all_chunks.jsonl"
    if not chunks_path.exists():
        print(f"Error: Chunks file not found at {chunks_path}")
        print("Run 02_chunk_text.py first to create chunks.")
        sys.exit(1)

    chunks = []
    with open(chunks_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))
    return chunks


def get_existing_chunk_ids(output_dir: Path) -> set[str]:
    """Scan existing output files to find already-processed chunk IDs."""
    existing = set()
    for jsonl_file in output_dir.glob("*.jsonl"):
        with open(jsonl_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        record = json.loads(line)
                        if "chunk_id" in record:
                            existing.add(record["chunk_id"])
                    except json.JSONDecodeError:
                        continue
    return existing


def generate_qa_for_chunk(
    chunk: dict,
    llm: LLMClient,
    config: Config,
) -> list[dict]:
    """Generate QA pairs for a single chunk."""
    content_type = chunk.get("content_type", "prose")
    text = chunk["text"]
    n = config.get_generation_count("qa_pairs")

    system_prompt = config.get_prompt("qa_system")

    if content_type == "tabular":
        instruction_template = config.get_prompt("training_log_qa_instruction")
    else:
        instruction_template = config.get_prompt("qa_instruction")

    user_prompt = instruction_template.format(n=n, text=text)

    try:
        raw_pairs = llm.generate_json(system_prompt, user_prompt)
    except Exception as e:
        print(f"\n  Error generating QA for {chunk['chunk_id']}: {e}")
        return []

    # Normalize each pair into the output schema
    qa_pairs = []
    for pair in raw_pairs:
        if "instruction" not in pair or "response" not in pair:
            continue
        qa_pairs.append({
            "instruction": pair["instruction"],
            "response": pair["response"],
            "source_id": chunk["source_id"],
            "chunk_id": chunk["chunk_id"],
            "content_type": content_type,
        })

    return qa_pairs


def run_generation(sample: int | None = None, resume: bool = False):
    """Main generation loop."""
    config = Config()
    config.ensure_directories()

    output_dir = config.GENERATED_DIR / "qa_pairs"
    output_dir.mkdir(parents=True, exist_ok=True)

    chunks = load_chunks(config)
    print(f"Loaded {len(chunks)} chunks")

    # Filter out already-processed chunks if resuming
    if resume:
        existing_ids = get_existing_chunk_ids(output_dir)
        before = len(chunks)
        chunks = [c for c in chunks if c["chunk_id"] not in existing_ids]
        print(f"Resume mode: skipping {before - len(chunks)} already-processed chunks")

    # Subsample for testing
    if sample is not None:
        chunks = chunks[:sample]
        print(f"Sample mode: processing {len(chunks)} chunks")

    if not chunks:
        print("No chunks to process.")
        return

    llm = LLMClient(config)
    output_path = output_dir / "qa_pairs.jsonl"

    total_pairs = 0
    open_mode = "a" if resume else "w"

    with open(output_path, open_mode, encoding="utf-8") as f:
        for chunk in tqdm(chunks, desc="Generating QA pairs"):
            pairs = generate_qa_for_chunk(chunk, llm, config)
            for pair in pairs:
                f.write(json.dumps(pair, ensure_ascii=False) + "\n")
            total_pairs += len(pairs)
            f.flush()

            # Small delay for rate limiting
            time.sleep(0.5)

    print(f"\nGeneration complete: {total_pairs} QA pairs from {len(chunks)} chunks")
    print(f"Output saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate QA pairs from text chunks using Claude"
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

    run_generation(sample=args.sample, resume=args.resume)


if __name__ == "__main__":
    main()
