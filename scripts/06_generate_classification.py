"""Generate text classification datasets from text chunks using the Claude API."""

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
        print("Run 02_chunk_text.py first to generate chunks.")
        sys.exit(1)

    chunks = []
    with open(chunks_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))
    return chunks


def get_existing_chunk_ids(output_dir: Path) -> set[str]:
    """Collect chunk_ids that already have generated output."""
    existing = set()
    if not output_dir.exists():
        return existing

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


def generate_classification_for_chunk(
    client: LLMClient,
    chunk: dict,
    system_prompt: str,
    instruction_template: str,
    num_examples: int,
    topics: list[str],
) -> list[dict]:
    """Generate classification examples for a single chunk."""
    topics_str = ", ".join(topics)
    user_prompt = instruction_template.format(
        n=num_examples,
        text=chunk["text"],
        topics=topics_str,
    )

    try:
        results = client.generate_json(system_prompt, user_prompt)
    except Exception as e:
        print(f"\nError generating for chunk {chunk['chunk_id']}: {e}")
        return []

    # Normalize output format
    examples = []
    for item in results:
        if "text" in item and "label" in item:
            examples.append({
                "text": item["text"],
                "label": item["label"],
                "source_id": chunk["source_id"],
                "chunk_id": chunk["chunk_id"],
                "content_type": chunk.get("content_type", "prose"),
            })

    return examples


def run_generation(sample: int | None = None, resume: bool = False):
    """Generate classification data from all chunks."""
    config = Config()
    config.ensure_directories()

    output_dir = config.GENERATED_DIR / "classification"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "classification_examples.jsonl"

    # Load configuration
    system_prompt = config.get_prompt("classification_system")
    instruction_template = config.get_prompt("classification_instruction")
    topics = config.topics
    num_examples = config.get_generation_count("classifications")

    if not system_prompt:
        print("Error: 'classification_system' prompt not found in generation_config.yaml")
        sys.exit(1)
    if not instruction_template:
        print("Error: 'classification_instruction' prompt not found in generation_config.yaml")
        sys.exit(1)
    if not topics:
        print("Warning: No topics found in generation_config.yaml, proceeding without topics.")

    # Load chunks
    chunks = load_chunks(config)
    print(f"Loaded {len(chunks)} chunks")

    # Apply sample limit
    if sample is not None:
        chunks = chunks[:sample]
        print(f"Sampling first {sample} chunks")

    # Resume support: skip already-processed chunks
    if resume:
        existing_ids = get_existing_chunk_ids(output_dir)
        original_count = len(chunks)
        chunks = [c for c in chunks if c["chunk_id"] not in existing_ids]
        skipped = original_count - len(chunks)
        if skipped > 0:
            print(f"Resuming: skipped {skipped} already-processed chunks")

    if not chunks:
        print("No chunks to process.")
        return

    print(f"Processing {len(chunks)} chunks, generating {num_examples} examples each")

    # Initialize LLM client
    client = LLMClient(config)

    # Open output file in append mode for resume support
    mode = "a" if resume else "w"
    total_generated = 0

    with open(output_path, mode, encoding="utf-8") as out_f:
        for chunk in tqdm(chunks, desc="Generating classifications"):
            examples = generate_classification_for_chunk(
                client=client,
                chunk=chunk,
                system_prompt=system_prompt,
                instruction_template=instruction_template,
                num_examples=num_examples,
                topics=topics,
            )

            for example in examples:
                out_f.write(json.dumps(example, ensure_ascii=False) + "\n")

            total_generated += len(examples)

            # Rate limiting
            time.sleep(0.5)

    print(f"\nGeneration complete!")
    print(f"Total examples generated: {total_generated}")
    print(f"Output saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate text classification datasets from text chunks"
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
