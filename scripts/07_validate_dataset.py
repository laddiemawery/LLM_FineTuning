"""Validate, deduplicate, and quality-check generated datasets."""

import json
import sys
from pathlib import Path

from rapidfuzz import fuzz
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.utils.config import Config


class DatasetValidator:
    """Validate and clean generated training datasets."""

    def __init__(self, dedup_threshold: float = 85.0, min_response_words: int = 10):
        self.dedup_threshold = dedup_threshold
        self.min_response_words = min_response_words
        self.stats = {
            "total": 0,
            "duplicates_removed": 0,
            "low_quality_removed": 0,
            "format_errors_removed": 0,
            "passed": 0,
        }

    def validate_qa(self, item: dict) -> bool:
        """Validate a QA pair."""
        instruction = item.get("instruction", "").strip()
        response = item.get("response", "").strip()

        if not instruction or not response:
            return False
        if len(instruction.split()) < 3:
            return False
        if len(response.split()) < self.min_response_words:
            return False
        if instruction == response:
            return False
        return True

    def validate_conversation(self, item: dict) -> bool:
        """Validate a multi-turn conversation."""
        messages = item.get("messages", [])
        if len(messages) < 2:
            return False
        # Check alternating roles
        for i, msg in enumerate(messages):
            if "role" not in msg or "content" not in msg:
                return False
            if not msg["content"].strip():
                return False
            expected_role = "user" if i % 2 == 0 else "assistant"
            if msg["role"] != expected_role:
                return False
        return True

    def validate_completion(self, item: dict) -> bool:
        """Validate a prompt-completion pair."""
        prompt = item.get("prompt", "").strip()
        completion = item.get("completion", "").strip()
        if not prompt or not completion:
            return False
        if len(completion.split()) < self.min_response_words:
            return False
        return True

    def validate_classification(self, item: dict) -> bool:
        """Validate a classification example."""
        text = item.get("text", "").strip()
        label = item.get("label", "").strip()
        if not text or not label:
            return False
        if len(text.split()) < 5:
            return False
        return True

    def get_dedup_key(self, item: dict, dataset_type: str) -> str:
        """Extract the primary text for deduplication."""
        if dataset_type == "qa_pairs":
            return item.get("instruction", "")
        elif dataset_type == "conversations":
            messages = item.get("messages", [])
            return messages[0]["content"] if messages else ""
        elif dataset_type == "completions":
            return item.get("prompt", "")
        elif dataset_type == "classification":
            return item.get("text", "")
        return ""

    def deduplicate(self, items: list[dict], dataset_type: str) -> list[dict]:
        """Remove near-duplicate entries using fuzzy matching."""
        if not items:
            return []

        unique = []
        seen_keys = []

        for item in tqdm(items, desc="Deduplicating", leave=False):
            key = self.get_dedup_key(item, dataset_type)
            if not key:
                continue

            is_dup = False
            for seen_key in seen_keys:
                if fuzz.ratio(key, seen_key) > self.dedup_threshold:
                    is_dup = True
                    self.stats["duplicates_removed"] += 1
                    break

            if not is_dup:
                unique.append(item)
                seen_keys.append(key)

        return unique

    def validate_dataset(self, input_dir: Path, dataset_type: str) -> list[dict]:
        """Load, validate, and deduplicate a dataset."""
        validators = {
            "qa_pairs": self.validate_qa,
            "conversations": self.validate_conversation,
            "completions": self.validate_completion,
            "classification": self.validate_classification,
        }
        validate_fn = validators.get(dataset_type)
        if not validate_fn:
            raise ValueError(f"Unknown dataset type: {dataset_type}")

        # Load all JSONL files from the directory
        items = []
        for jsonl_file in sorted(input_dir.glob("*.jsonl")):
            with open(jsonl_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        item = json.loads(line)
                        items.append(item)
                    except json.JSONDecodeError:
                        self.stats["format_errors_removed"] += 1

        self.stats["total"] = len(items)

        # Validate
        valid_items = []
        for item in items:
            if validate_fn(item):
                valid_items.append(item)
            else:
                self.stats["low_quality_removed"] += 1

        # Deduplicate
        unique_items = self.deduplicate(valid_items, dataset_type)
        self.stats["passed"] = len(unique_items)

        return unique_items


def run_validation():
    """Validate all generated datasets."""
    config = Config()
    config.ensure_directories()

    dataset_types = ["qa_pairs", "conversations", "completions", "classification"]
    all_validated = {}

    for dtype in dataset_types:
        input_dir = config.GENERATED_DIR / dtype
        if not input_dir.exists() or not list(input_dir.glob("*.jsonl")):
            print(f"\n[{dtype}] No data found, skipping.")
            continue

        print(f"\n[{dtype}] Validating...")
        validator = DatasetValidator()
        validated = validator.validate_dataset(input_dir, dtype)
        all_validated[dtype] = validated

        # Save validated data
        output_path = config.VALIDATED_DIR / f"{dtype}.jsonl"
        with open(output_path, "w", encoding="utf-8") as f:
            for item in validated:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

        print(f"  Total: {validator.stats['total']}")
        print(f"  Format errors: {validator.stats['format_errors_removed']}")
        print(f"  Low quality: {validator.stats['low_quality_removed']}")
        print(f"  Duplicates: {validator.stats['duplicates_removed']}")
        print(f"  Passed: {validator.stats['passed']}")
        print(f"  Saved to: {output_path}")

    # Overall summary
    total = sum(len(v) for v in all_validated.values())
    print(f"\n{'='*50}")
    print(f"Total validated examples: {total}")
    for dtype, items in all_validated.items():
        print(f"  {dtype}: {len(items)}")


def main():
    run_validation()


if __name__ == "__main__":
    main()
