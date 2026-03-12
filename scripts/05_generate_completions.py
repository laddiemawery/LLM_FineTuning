"""Generate text completion datasets from text chunks using the Claude API."""

import argparse
import json
import sys
import time
from pathlib import Path

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.utils.config import Config
from scripts.utils.llm_client import LLMClient


def load_chunks(chunks_path: Path) -> list[dict]:
    """Load chunks from a JSONL file."""
    chunks = []
    with open(chunks_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))
    return chunks


def load_existing_chunk_ids(output_dir: Path) -> set[str]:
    """Collect chunk_ids that already have generated output."""
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


def generate_completions_for_chunk(
    llm: LLMClient,
    chunk: dict,
    system_prompt: str,
    instruction_template: str,
    n: int,
) -> list[dict]:
    """Generate prompt-completion pairs for a single chunk."""
    user_prompt = instruction_template.format(n=n, text=chunk["text"])

    try:
        pairs = llm.generate_json(system_prompt, user_prompt)
    except Exception as e:
        print(f"\n  Error generating for {chunk['chunk_id']}: {e}")
        return []

    results = []
    for pair in pairs:
        if "prompt" in pair and "completion" in pair:
            results.append({
                "prompt": pair["prompt"],
                "completion": pair["completion"],
                "source_id": chunk.get("source_id", ""),
                "chunk_id": chunk.get("chunk_id", ""),
                "content_type": chunk.get("content_type", ""),
            })

    return results


def run_generation(sample: int | None = None, resume: bool = False):
    """Generate completion datasets from all chunks."""
    config = Config()
    config.ensure_directories()

    output_dir = config.GENERATED_DIR / "completions"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load chunks
    chunks_path = config.CHUNKS_DIR / "all_chunks.jsonl"
    if not chunks_path.exists():
        print(f"Chunks file not found: {chunks_path}")
        print("Run 02_chunk_text.py first.")
        sys.exit(1)

    chunks = load_chunks(chunks_path)
    print(f"Loaded {len(chunks)} chunks from {chunks_path}")

    # Filter out already-processed chunks when resuming
    if resume:
        existing_ids = load_existing_chunk_ids(output_dir)
        chunks = [c for c in chunks if c.get("chunk_id") not in existing_ids]
        print(f"Resuming: {len(existing_ids)} chunks already processed, {len(chunks)} remaining")

    # Apply sample limit
    if sample is not None:
        chunks = chunks[:sample]
        print(f"Sampling {len(chunks)} chunks")

    if not chunks:
        print("No chunks to process.")
        return

    # Load prompts and generation count
    system_prompt = config.get_prompt("completion_system")
    instruction_template = config.get_prompt("completion_instruction")
    n = config.get_generation_count("completions")

    if not system_prompt or not instruction_template:
        print("Error: completion_system or completion_instruction prompt not found in config.")
        sys.exit(1)

    print(f"Generating {n} completions per chunk for {len(chunks)} chunks")

    # Initialize LLM client
    llm = LLMClient(config)

    # Output file
    output_path = output_dir / "completions.jsonl"
    total_generated = 0

    with open(output_path, "a", encoding="utf-8") as out_f:
        for chunk in tqdm(chunks, desc="Generating completions"):
            results = generate_completions_for_chunk(
                llm, chunk, system_prompt, instruction_template, n
            )

            for record in results:
                out_f.write(json.dumps(record, ensure_ascii=False) + "\n")

            total_generated += len(results)
            out_f.flush()

            # Small delay for rate limiting
            time.sleep(0.5)

    print(f"\nGeneration complete: {total_generated} completion pairs saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate text completion datasets from chunks"
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
