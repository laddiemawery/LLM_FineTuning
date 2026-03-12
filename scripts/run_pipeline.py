"""End-to-end pipeline orchestrator for the LLM fine-tuning data pipeline."""

import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.utils.config import Config


STEPS = {
    1: ("Extract", "scripts/01_extract/extract_runner.py"),
    2: ("Chunk", "scripts/02_chunk_text.py"),
    3: ("Generate QA", "scripts/03_generate_qa.py"),
    4: ("Generate Conversations", "scripts/04_generate_conversations.py"),
    5: ("Generate Completions", "scripts/05_generate_completions.py"),
    6: ("Generate Classification", "scripts/06_generate_classification.py"),
    7: ("Validate", "scripts/07_validate_dataset.py"),
    8: ("Prepare Training", "scripts/08_prepare_training.py"),
}


def run_step(step_num: int, source: str | None = None, sample: int | None = None):
    """Run a single pipeline step."""
    name, script_path = STEPS[step_num]
    config = Config()
    full_path = config.PROJECT_ROOT / script_path

    cmd = [sys.executable, str(full_path)]

    # Add source flag for extraction step
    if step_num == 1 and source:
        cmd.extend(["--source", source])

    # Add sample flag for generation steps
    if step_num in (3, 4, 5, 6) and sample:
        cmd.extend(["--sample", str(sample)])

    print(f"\n{'='*60}")
    print(f"Step {step_num}: {name}")
    print(f"Running: {' '.join(cmd)}")
    print(f"{'='*60}\n")

    result = subprocess.run(cmd, cwd=str(config.PROJECT_ROOT))
    if result.returncode != 0:
        print(f"\nERROR: Step {step_num} ({name}) failed with exit code {result.returncode}")
        return False
    return True


def run_pipeline(
    steps: list[int] | None = None,
    source: str | None = None,
    sample: int | None = None,
):
    """Run the full pipeline or selected steps."""
    config = Config()
    config.ensure_directories()

    if steps is None:
        steps = sorted(STEPS.keys())

    print(f"Running pipeline steps: {steps}")
    if source:
        print(f"Source filter: {source}")
    if sample:
        print(f"Sample size: {sample}")

    for step in steps:
        if step not in STEPS:
            print(f"WARNING: Unknown step {step}, skipping.")
            continue
        success = run_step(step, source=source, sample=sample)
        if not success:
            print(f"\nPipeline halted at step {step}.")
            return False

    print(f"\n{'='*60}")
    print("Pipeline completed successfully!")
    print(f"{'='*60}")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Run the LLM fine-tuning data pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/run_pipeline.py                        # Run full pipeline
  python scripts/run_pipeline.py --steps 1,2            # Extract + chunk only
  python scripts/run_pipeline.py --steps 1 --source cscs_exam_prep  # Single source
  python scripts/run_pipeline.py --steps 3,4,5,6 --sample 5        # Generate small sample
  python scripts/run_pipeline.py --steps 7,8            # Validate + prepare only
        """,
    )
    parser.add_argument(
        "--steps",
        type=str,
        help="Comma-separated step numbers to run (default: all)",
    )
    parser.add_argument(
        "--source",
        type=str,
        help="Process a specific source by ID (for extraction step)",
    )
    parser.add_argument(
        "--sample",
        type=int,
        help="Number of chunks to sample (for generation steps)",
    )
    parser.add_argument(
        "--list-steps",
        action="store_true",
        help="List all available pipeline steps",
    )
    args = parser.parse_args()

    if args.list_steps:
        print("Available pipeline steps:")
        for num, (name, script) in sorted(STEPS.items()):
            print(f"  {num}. {name} ({script})")
        return

    steps = None
    if args.steps:
        steps = [int(s.strip()) for s in args.steps.split(",")]

    run_pipeline(steps=steps, source=args.source, sample=args.sample)


if __name__ == "__main__":
    main()
