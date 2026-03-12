"""Prompt engineering toolkit for testing generation prompts before full runs."""

import argparse
import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.utils.config import Config
from scripts.utils.llm_client import LLMClient

# Claude Sonnet pricing (per million tokens)
INPUT_COST_PER_M = 3.00
OUTPUT_COST_PER_M = 15.00

# Mapping from subcommand type names to config prompt keys and generation count keys
TYPE_MAP = {
    "qa": {
        "system_prompt": "qa_system",
        "instruction_prompt": "qa_instruction",
        "generation_count_key": "qa_pairs",
    },
    "conversation": {
        "system_prompt": "conversation_system",
        "instruction_prompt": "conversation_instruction",
        "generation_count_key": "conversations",
    },
    "completion": {
        "system_prompt": "completion_system",
        "instruction_prompt": "completion_instruction",
        "generation_count_key": "completions",
    },
    "classification": {
        "system_prompt": "classification_system",
        "instruction_prompt": "classification_instruction",
        "generation_count_key": "classifications",
    },
}


def _load_random_chunks(n: int, config: Config) -> list[dict]:
    """Load n random chunks from data/chunks/all_chunks.jsonl."""
    chunks_path = config.CHUNKS_DIR / "all_chunks.jsonl"
    if not chunks_path.exists():
        print(f"Error: Chunks file not found at {chunks_path}")
        print("Run 02_chunk_text.py first to create chunks.")
        sys.exit(1)

    all_chunks = []
    with open(chunks_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                all_chunks.append(json.loads(line))

    if not all_chunks:
        print("Error: No chunks found in all_chunks.jsonl")
        sys.exit(1)

    n = min(n, len(all_chunks))
    return random.sample(all_chunks, n)


def _load_all_chunks(config: Config) -> list[dict]:
    """Load all chunks from data/chunks/all_chunks.jsonl."""
    chunks_path = config.CHUNKS_DIR / "all_chunks.jsonl"
    if not chunks_path.exists():
        print(f"Error: Chunks file not found at {chunks_path}")
        print("Run 02_chunk_text.py first to create chunks.")
        sys.exit(1)

    all_chunks = []
    with open(chunks_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                all_chunks.append(json.loads(line))
    return all_chunks


def _build_prompts(chunk: dict, gen_type: str, config: Config, custom_instruction: str | None = None) -> tuple[str, str]:
    """Build the system and user prompts for a given chunk and generation type."""
    type_info = TYPE_MAP[gen_type]
    system_prompt = config.get_prompt(type_info["system_prompt"])
    n = config.get_generation_count(type_info["generation_count_key"])
    text = chunk["text"]

    if custom_instruction:
        instruction_template = custom_instruction
    else:
        content_type = chunk.get("content_type", "prose")
        if gen_type == "qa" and content_type == "tabular":
            instruction_template = config.get_prompt("training_log_qa_instruction")
        elif gen_type == "conversation" and content_type == "tabular":
            instruction_template = config.get_prompt("training_log_conversation_instruction")
        else:
            instruction_template = config.get_prompt(type_info["instruction_prompt"])

    # Build format kwargs based on what placeholders exist in the template
    format_kwargs = {"n": n, "text": text}
    if gen_type == "classification":
        format_kwargs["topics"] = ", ".join(config.topics)

    try:
        user_prompt = instruction_template.format(**format_kwargs)
    except KeyError:
        # If the custom prompt doesn't use all placeholders, try with just text
        user_prompt = instruction_template.format(text=text, n=n)

    return system_prompt, user_prompt


def _run_on_chunk(chunk: dict, gen_type: str, config: Config, llm: LLMClient, custom_instruction: str | None = None) -> dict:
    """Run generation on a single chunk and return results with metadata."""
    system_prompt, user_prompt = _build_prompts(chunk, gen_type, config, custom_instruction)

    start_time = time.time()
    try:
        raw_response = llm.generate(system_prompt, user_prompt)
        elapsed = time.time() - start_time
    except Exception as e:
        return {
            "chunk_id": chunk.get("chunk_id", "unknown"),
            "chunk_preview": chunk["text"][:100],
            "error": str(e),
            "examples": [],
            "raw_response": "",
            "elapsed": 0,
        }

    # Try to parse JSON examples from the response
    try:
        text = raw_response.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            lines = [line for line in lines if not line.strip().startswith("```")]
            text = "\n".join(lines)

        parsed = json.loads(text)
        examples = parsed if isinstance(parsed, list) else [parsed]
    except json.JSONDecodeError:
        # Try extracting individual JSON objects
        examples = []
        depth = 0
        start = None
        for i, ch in enumerate(raw_response):
            if ch == "{":
                if depth == 0:
                    start = i
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0 and start is not None:
                    try:
                        obj = json.loads(raw_response[start:i + 1])
                        examples.append(obj)
                    except json.JSONDecodeError:
                        pass
                    start = None

    return {
        "chunk_id": chunk.get("chunk_id", "unknown"),
        "chunk_preview": chunk["text"][:100],
        "examples": examples,
        "raw_response": raw_response,
        "elapsed": elapsed,
        "input_tokens_est": len(system_prompt + user_prompt) // 4,  # rough estimate
        "output_tokens_est": len(raw_response) // 4,
    }


def _format_results(results: list[dict], label: str = "") -> None:
    """Display formatted results from generation runs."""
    header = f"Results{f' ({label})' if label else ''}"
    print(f"\n{'=' * 70}")
    print(f"  {header}")
    print(f"{'=' * 70}")

    total_examples = 0
    total_response_len = 0

    for i, result in enumerate(results):
        print(f"\n--- Chunk {i + 1}: {result['chunk_id']} ---")
        print(f"  Preview: {result['chunk_preview']}...")

        if "error" in result:
            print(f"  ERROR: {result['error']}")
            continue

        num_examples = len(result["examples"])
        total_examples += num_examples
        response_len = len(result.get("raw_response", ""))
        total_response_len += response_len

        print(f"  Examples generated: {num_examples}")
        print(f"  Response length: {response_len:,} chars")
        print(f"  Time: {result['elapsed']:.1f}s")

        # Show first example as a preview
        if result["examples"]:
            first = result["examples"][0]
            print(f"  Sample output:")
            for key, value in first.items():
                val_str = str(value)
                if len(val_str) > 120:
                    val_str = val_str[:120] + "..."
                print(f"    {key}: {val_str}")

    # Summary stats
    n = len(results)
    valid = [r for r in results if "error" not in r]
    print(f"\n--- Summary ---")
    print(f"  Chunks processed: {n}")
    print(f"  Successful: {len(valid)}")
    print(f"  Total examples: {total_examples}")
    if valid:
        print(f"  Avg examples/chunk: {total_examples / len(valid):.1f}")
        print(f"  Avg response length: {total_response_len / len(valid):,.0f} chars")


# ─── Subcommands ─────────────────────────────────────────────────────────────


def cmd_test(args):
    """Test a prompt on sample chunks."""
    config = Config()
    llm = LLMClient(config)
    chunks = _load_random_chunks(args.sample, config)

    print(f"Testing '{args.type}' generation on {len(chunks)} random chunk(s)...")

    results = []
    for chunk in chunks:
        result = _run_on_chunk(chunk, args.type, config, llm)
        results.append(result)
        if len(chunks) > 1:
            time.sleep(0.5)

    _format_results(results)


def cmd_compare(args):
    """A/B test two prompts on the same chunks."""
    config = Config()
    llm = LLMClient(config)
    chunks = _load_random_chunks(args.sample, config)

    print(f"A/B comparing '{args.type}' generation on {len(chunks)} random chunk(s)...")
    print(f"  Prompt A: default (from config)")
    print(f"  Prompt B: custom")

    # Run prompt A (default)
    print("\nRunning Prompt A (default)...")
    results_a = []
    for chunk in chunks:
        result = _run_on_chunk(chunk, args.type, config, llm)
        results_a.append(result)
        time.sleep(0.5)

    # Run prompt B (custom)
    print("Running Prompt B (custom)...")
    results_b = []
    for chunk in chunks:
        result = _run_on_chunk(chunk, args.type, config, llm, custom_instruction=args.prompt_b)
        results_b.append(result)
        time.sleep(0.5)

    # Display side-by-side
    _format_results(results_a, label="Prompt A - Default")
    _format_results(results_b, label="Prompt B - Custom")

    # Comparison summary
    examples_a = sum(len(r["examples"]) for r in results_a if "error" not in r)
    examples_b = sum(len(r["examples"]) for r in results_b if "error" not in r)
    valid_a = [r for r in results_a if "error" not in r]
    valid_b = [r for r in results_b if "error" not in r]
    avg_len_a = sum(len(r["raw_response"]) for r in valid_a) / max(len(valid_a), 1)
    avg_len_b = sum(len(r["raw_response"]) for r in valid_b) / max(len(valid_b), 1)

    print(f"\n{'=' * 70}")
    print(f"  A/B Comparison Summary")
    print(f"{'=' * 70}")
    print(f"  {'Metric':<30} {'Prompt A':>15} {'Prompt B':>15}")
    print(f"  {'-' * 60}")
    print(f"  {'Total examples':<30} {examples_a:>15} {examples_b:>15}")
    print(f"  {'Avg response length':<30} {avg_len_a:>15,.0f} {avg_len_b:>15,.0f}")
    print(f"  {'Errors':<30} {len(results_a) - len(valid_a):>15} {len(results_b) - len(valid_b):>15}")


def cmd_estimate(args):
    """Estimate full-run API cost based on sample results."""
    config = Config()
    llm = LLMClient(config)

    # Count total chunks
    all_chunks = _load_all_chunks(config)
    total_chunks = len(all_chunks)

    # Run a small sample to estimate token usage
    sample_size = min(3, total_chunks)
    sample_chunks = random.sample(all_chunks, sample_size)

    print(f"Estimating cost for '{args.type}' generation across {total_chunks} chunks...")
    print(f"Running {sample_size}-chunk sample to measure token usage...\n")

    input_tokens = []
    output_tokens = []
    elapsed_times = []

    for chunk in sample_chunks:
        result = _run_on_chunk(chunk, args.type, config, llm)
        if "error" not in result:
            input_tokens.append(result["input_tokens_est"])
            output_tokens.append(result["output_tokens_est"])
            elapsed_times.append(result["elapsed"])
        time.sleep(0.5)

    if not input_tokens:
        print("Error: All sample runs failed. Cannot estimate cost.")
        sys.exit(1)

    avg_input = sum(input_tokens) / len(input_tokens)
    avg_output = sum(output_tokens) / len(output_tokens)
    avg_time = sum(elapsed_times) / len(elapsed_times)

    # Calculate estimates for full run
    total_input_tokens = avg_input * total_chunks
    total_output_tokens = avg_output * total_chunks
    input_cost = (total_input_tokens / 1_000_000) * INPUT_COST_PER_M
    output_cost = (total_output_tokens / 1_000_000) * OUTPUT_COST_PER_M
    total_cost = input_cost + output_cost

    # Time estimate includes 0.5s rate limiting delay per chunk
    total_time_seconds = (avg_time + 0.5) * total_chunks
    total_time_minutes = total_time_seconds / 60

    print(f"{'=' * 50}")
    print(f"  Full-Run Cost Estimate: {args.type}")
    print(f"{'=' * 50}")
    print(f"  Total chunks:          {total_chunks:,}")
    print(f"  Total API calls:       {total_chunks:,}")
    print()
    print(f"  Avg input tokens/call: {avg_input:,.0f}")
    print(f"  Avg output tokens/call:{avg_output:,.0f}")
    print()
    print(f"  Est. total input:      {total_input_tokens:,.0f} tokens")
    print(f"  Est. total output:     {total_output_tokens:,.0f} tokens")
    print()
    print(f"  Input cost:            ${input_cost:,.2f}  (@ ${INPUT_COST_PER_M}/M tokens)")
    print(f"  Output cost:           ${output_cost:,.2f}  (@ ${OUTPUT_COST_PER_M}/M tokens)")
    print(f"  ──────────────────────────────────")
    print(f"  Estimated total cost:  ${total_cost:,.2f}")
    print()
    print(f"  Estimated time:        {total_time_minutes:,.1f} minutes ({total_time_seconds / 3600:.1f} hours)")
    print(f"  (based on {avg_time:.1f}s/call + 0.5s rate limit delay)")


def cmd_preview(args):
    """Preview prompts without calling the API."""
    config = Config()
    chunks = _load_random_chunks(args.sample, config)

    for i, chunk in enumerate(chunks):
        system_prompt, user_prompt = _build_prompts(chunk, args.type, config)

        print(f"\n{'=' * 70}")
        print(f"  Chunk {i + 1}: {chunk.get('chunk_id', 'unknown')}")
        print(f"{'=' * 70}")
        print(f"\n--- SYSTEM PROMPT ---")
        print(system_prompt.strip())
        print(f"\n--- USER PROMPT ---")
        print(user_prompt.strip())
        print()


# ─── CLI ─────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Prompt engineering toolkit for testing generation prompts",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
examples:
  python scripts/prompt_toolkit.py test --type qa --sample 3
  python scripts/prompt_toolkit.py compare --type qa --sample 3 --prompt-b "Your custom prompt here"
  python scripts/prompt_toolkit.py estimate --type qa
  python scripts/prompt_toolkit.py preview --type qa --sample 1
""",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # ── test ──
    test_parser = subparsers.add_parser("test", help="Test a prompt on sample chunks")
    test_parser.add_argument(
        "--type",
        required=True,
        choices=["qa", "conversation", "completion", "classification"],
        help="Generation type to test",
    )
    test_parser.add_argument(
        "--sample",
        type=int,
        default=3,
        help="Number of random chunks to sample (default: 3)",
    )
    test_parser.set_defaults(func=cmd_test)

    # ── compare ──
    compare_parser = subparsers.add_parser("compare", help="A/B test two prompts on the same chunks")
    compare_parser.add_argument(
        "--type",
        required=True,
        choices=["qa", "conversation", "completion", "classification"],
        help="Generation type to test",
    )
    compare_parser.add_argument(
        "--sample",
        type=int,
        default=3,
        help="Number of random chunks to sample (default: 3)",
    )
    compare_parser.add_argument(
        "--prompt-b",
        required=True,
        help="Custom instruction prompt to compare against the default",
    )
    compare_parser.set_defaults(func=cmd_compare)

    # ── estimate ──
    estimate_parser = subparsers.add_parser("estimate", help="Estimate full-run API cost")
    estimate_parser.add_argument(
        "--type",
        required=True,
        choices=["qa", "conversation", "completion", "classification"],
        help="Generation type to estimate",
    )
    estimate_parser.set_defaults(func=cmd_estimate)

    # ── preview ──
    preview_parser = subparsers.add_parser("preview", help="Preview prompts without calling the API")
    preview_parser.add_argument(
        "--type",
        required=True,
        choices=["qa", "conversation", "completion", "classification"],
        help="Generation type to preview",
    )
    preview_parser.add_argument(
        "--sample",
        type=int,
        default=1,
        help="Number of random chunks to preview (default: 1)",
    )
    preview_parser.set_defaults(func=cmd_preview)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
