"""Evaluate a fine-tuned model on the test set and run sample inferences."""

import json
import sys
from pathlib import Path

import torch
from datasets import load_dataset
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.utils.config import Config


def load_model(model_path: str, base_model_name: str | None = None):
    """Load the fine-tuned model (base + LoRA adapter)."""
    model_path = Path(model_path)

    # Check if this is a PEFT adapter or full model
    is_peft = (model_path / "adapter_config.json").exists()

    if is_peft:
        if not base_model_name:
            cfg = Config().training_config
            base_model_name = cfg["model"]["name"]

        print(f"Loading base model: {base_model_name}")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
        )
        model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=torch.bfloat16,
        )
        print(f"Loading LoRA adapter: {model_path}")
        model = PeftModel.from_pretrained(model, str(model_path))
    else:
        print(f"Loading full model: {model_path}")
        model = AutoModelForCausalLM.from_pretrained(
            str(model_path),
            device_map="auto",
            torch_dtype=torch.bfloat16,
        )

    tokenizer = AutoTokenizer.from_pretrained(str(model_path))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model.eval()
    return model, tokenizer


def evaluate_perplexity(model, tokenizer, test_path: str, max_samples: int = 100):
    """Compute perplexity on the test set."""
    dataset = load_dataset("json", data_files=test_path, split="train")
    if len(dataset) > max_samples:
        dataset = dataset.select(range(max_samples))

    total_loss = 0.0
    total_tokens = 0

    print(f"Evaluating perplexity on {len(dataset)} samples...")

    for i, example in enumerate(dataset):
        text = example["text"]
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=4096)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs, labels=inputs["input_ids"])
            loss = outputs.loss

        n_tokens = inputs["input_ids"].shape[1]
        total_loss += loss.item() * n_tokens
        total_tokens += n_tokens

        if (i + 1) % 20 == 0:
            running_ppl = torch.exp(torch.tensor(total_loss / total_tokens)).item()
            print(f"  [{i+1}/{len(dataset)}] Running perplexity: {running_ppl:.2f}")

    avg_loss = total_loss / total_tokens
    perplexity = torch.exp(torch.tensor(avg_loss)).item()
    return perplexity


def run_inference(model, tokenizer, prompt: str, max_new_tokens: int = 512) -> str:
    """Run inference on a single prompt."""
    BOS = "<|begin_of_text|>"
    HEADER_START = "<|start_header_id|>"
    HEADER_END = "<|end_header_id|>"
    EOT = "<|eot_id|>"

    system_msg = (
        "You are a knowledgeable health and fitness professional with expertise in "
        "exercise science, strength and conditioning, nutrition, physical therapy, "
        "occupational therapy, and recovery. Provide accurate, evidence-based responses."
    )

    formatted = (
        f"{BOS}"
        f"{HEADER_START}system{HEADER_END}\n\n{system_msg}{EOT}"
        f"{HEADER_START}user{HEADER_END}\n\n{prompt}{EOT}"
        f"{HEADER_START}assistant{HEADER_END}\n\n"
    )

    inputs = tokenizer(formatted, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
        )

    response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return response.strip()


SAMPLE_PROMPTS = [
    "What are the key principles of progressive overload in strength training?",
    "Design a 4-day upper/lower split for an intermediate lifter focused on hypertrophy.",
    "What nutritional strategies help with post-workout recovery?",
    "Explain the difference between concentric and eccentric muscle contractions.",
    "A client reports knee pain during squats. What assessments and modifications would you recommend?",
    "What are the physiological adaptations to cardiovascular endurance training?",
    "How should a strength and conditioning program be periodized for a college football player?",
    "What is the role of proprioception in injury prevention and rehabilitation?",
]


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate fine-tuned model")
    parser.add_argument("model_path", type=str, help="Path to fine-tuned model")
    parser.add_argument("--base-model", type=str, help="Base model name (for PEFT models)")
    parser.add_argument("--test-file", type=str, help="Path to test JSONL file")
    parser.add_argument("--max-samples", type=int, default=100, help="Max test samples")
    parser.add_argument("--interactive", action="store_true", help="Interactive prompt mode")
    args = parser.parse_args()

    config = Config()
    model, tokenizer = load_model(args.model_path, args.base_model)

    # Perplexity evaluation
    test_file = args.test_file or str(config.TRAINING_DIR / "test.jsonl")
    if Path(test_file).exists():
        perplexity = evaluate_perplexity(model, tokenizer, test_file, args.max_samples)
        print(f"\nTest Perplexity: {perplexity:.2f}")
    else:
        print(f"\nTest file not found: {test_file}")

    # Sample inferences
    print(f"\n{'='*60}")
    print("Sample Inferences:")
    print(f"{'='*60}")

    for prompt in SAMPLE_PROMPTS:
        print(f"\nQ: {prompt}")
        response = run_inference(model, tokenizer, prompt)
        print(f"A: {response}\n")
        print("-" * 40)

    # Interactive mode
    if args.interactive:
        print(f"\n{'='*60}")
        print("Interactive Mode (type 'quit' to exit)")
        print(f"{'='*60}")
        while True:
            prompt = input("\nYour question: ").strip()
            if prompt.lower() in ("quit", "exit", "q"):
                break
            response = run_inference(model, tokenizer, prompt)
            print(f"\nResponse: {response}")


if __name__ == "__main__":
    main()
