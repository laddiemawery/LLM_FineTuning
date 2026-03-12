"""Prepare final training data: merge, format for Llama 3, and split."""

import json
import random
import sys
from pathlib import Path

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.utils.config import Config

# Llama 3 chat template tokens
BOS = "<|begin_of_text|>"
HEADER_START = "<|start_header_id|>"
HEADER_END = "<|end_header_id|>"
EOT = "<|eot_id|>"

SYSTEM_MESSAGE = (
    "You are a knowledgeable health and fitness professional with expertise in "
    "exercise science, strength and conditioning, nutrition, physical therapy, "
    "occupational therapy, and recovery. Provide accurate, evidence-based responses."
)


def format_qa_as_chat(item: dict) -> dict:
    """Convert a QA pair to Llama 3 chat format."""
    text = (
        f"{BOS}"
        f"{HEADER_START}system{HEADER_END}\n\n{SYSTEM_MESSAGE}{EOT}"
        f"{HEADER_START}user{HEADER_END}\n\n{item['instruction']}{EOT}"
        f"{HEADER_START}assistant{HEADER_END}\n\n{item['response']}{EOT}"
    )
    return {"text": text, "format": "qa", "source_id": item.get("source_id", "")}


def format_conversation_as_chat(item: dict) -> dict:
    """Convert a multi-turn conversation to Llama 3 chat format."""
    text = f"{BOS}{HEADER_START}system{HEADER_END}\n\n{SYSTEM_MESSAGE}{EOT}"
    for msg in item["messages"]:
        role = msg["role"]
        content = msg["content"]
        text += f"{HEADER_START}{role}{HEADER_END}\n\n{content}{EOT}"
    return {"text": text, "format": "conversation", "source_id": item.get("source_id", "")}


def format_completion_as_chat(item: dict) -> dict:
    """Convert a completion pair to Llama 3 chat format."""
    text = (
        f"{BOS}"
        f"{HEADER_START}system{HEADER_END}\n\n{SYSTEM_MESSAGE}{EOT}"
        f"{HEADER_START}user{HEADER_END}\n\nContinue the following: {item['prompt']}{EOT}"
        f"{HEADER_START}assistant{HEADER_END}\n\n{item['completion']}{EOT}"
    )
    return {"text": text, "format": "completion", "source_id": item.get("source_id", "")}


def format_classification_as_chat(item: dict) -> dict:
    """Convert a classification example to Llama 3 chat format."""
    text = (
        f"{BOS}"
        f"{HEADER_START}system{HEADER_END}\n\n{SYSTEM_MESSAGE}{EOT}"
        f"{HEADER_START}user{HEADER_END}\n\n"
        f"Classify the following health/fitness text into the most appropriate category: "
        f"\"{item['text']}\"{EOT}"
        f"{HEADER_START}assistant{HEADER_END}\n\n"
        f"This text falls under the category of: {item['label']}{EOT}"
    )
    return {"text": text, "format": "classification", "source_id": item.get("source_id", "")}


FORMATTERS = {
    "qa_pairs": format_qa_as_chat,
    "conversations": format_conversation_as_chat,
    "completions": format_completion_as_chat,
    "classification": format_classification_as_chat,
}


def load_validated_data(config: Config) -> list[dict]:
    """Load all validated datasets and format for training."""
    all_examples = []

    for dtype, formatter in FORMATTERS.items():
        filepath = config.VALIDATED_DIR / f"{dtype}.jsonl"
        if not filepath.exists():
            print(f"  [{dtype}] No validated data found, skipping.")
            continue

        count = 0
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                formatted = formatter(item)
                all_examples.append(formatted)
                count += 1
        print(f"  [{dtype}] Loaded {count} examples")

    return all_examples


def split_data(
    examples: list[dict],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 42,
) -> tuple[list[dict], list[dict], list[dict]]:
    """Stratified split by format type."""
    random.seed(seed)

    # Group by format
    by_format = {}
    for ex in examples:
        fmt = ex.get("format", "unknown")
        by_format.setdefault(fmt, []).append(ex)

    train, val, test = [], [], []

    for fmt, items in by_format.items():
        random.shuffle(items)
        n = len(items)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)

        train.extend(items[:n_train])
        val.extend(items[n_train : n_train + n_val])
        test.extend(items[n_train + n_val :])

    # Shuffle each split
    random.shuffle(train)
    random.shuffle(val)
    random.shuffle(test)

    return train, val, test


def save_split(examples: list[dict], output_path: Path):
    """Save examples as JSONL."""
    with open(output_path, "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")


def run_preparation():
    """Merge validated data, format, and split."""
    config = Config()
    config.ensure_directories()

    split_ratios = config.training_config.get("data", {}).get(
        "train_val_test_split", [0.8, 0.1, 0.1]
    )

    print("Loading validated datasets...")
    all_examples = load_validated_data(config)

    if not all_examples:
        print("No validated data found. Run validation first.")
        return

    print(f"\nTotal examples: {len(all_examples)}")

    # Count by format
    format_counts = {}
    for ex in all_examples:
        fmt = ex.get("format", "unknown")
        format_counts[fmt] = format_counts.get(fmt, 0) + 1
    for fmt, count in sorted(format_counts.items()):
        print(f"  {fmt}: {count}")

    # Split
    print(f"\nSplitting ({split_ratios[0]}/{split_ratios[1]}/{split_ratios[2]})...")
    train, val, test = split_data(all_examples, *split_ratios)

    # Save
    save_split(train, config.TRAINING_DIR / "train.jsonl")
    save_split(val, config.TRAINING_DIR / "val.jsonl")
    save_split(test, config.TRAINING_DIR / "test.jsonl")

    print(f"\nSaved:")
    print(f"  Train: {len(train)} examples -> {config.TRAINING_DIR / 'train.jsonl'}")
    print(f"  Val:   {len(val)} examples -> {config.TRAINING_DIR / 'val.jsonl'}")
    print(f"  Test:  {len(test)} examples -> {config.TRAINING_DIR / 'test.jsonl'}")


def main():
    run_preparation()


if __name__ == "__main__":
    main()
