"""Fine-tune Llama 3 with PEFT/LoRA on the prepared health & fitness dataset."""

import sys
from pathlib import Path

import torch
import yaml
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.utils.config import Config


def load_training_config() -> dict:
    config = Config()
    return config.training_config


def setup_quantization(cfg: dict) -> BitsAndBytesConfig:
    """Configure 4-bit quantization for QLoRA."""
    model_cfg = cfg["model"]
    return BitsAndBytesConfig(
        load_in_4bit=model_cfg.get("load_in_4bit", True),
        bnb_4bit_compute_dtype=getattr(torch, model_cfg.get("bnb_4bit_compute_dtype", "bfloat16")),
        bnb_4bit_quant_type=model_cfg.get("bnb_4bit_quant_type", "nf4"),
        bnb_4bit_use_double_quant=True,
    )


def setup_lora(cfg: dict) -> LoraConfig:
    """Configure LoRA adapter."""
    lora_cfg = cfg["lora"]
    return LoraConfig(
        r=lora_cfg.get("r", 16),
        lora_alpha=lora_cfg.get("lora_alpha", 32),
        lora_dropout=lora_cfg.get("lora_dropout", 0.05),
        target_modules=lora_cfg.get("target_modules", ["q_proj", "v_proj"]),
        bias=lora_cfg.get("bias", "none"),
        task_type=lora_cfg.get("task_type", "CAUSAL_LM"),
    )


def tokenize_dataset(dataset, tokenizer, max_length: int = 4096):
    """Tokenize the dataset for training."""
    def tokenize_fn(examples):
        tokenized = tokenizer(
            examples["text"],
            truncation=True,
            max_length=max_length,
            padding=False,
        )
        tokenized["labels"] = tokenized["input_ids"].copy()
        return tokenized

    return dataset.map(
        tokenize_fn,
        batched=True,
        remove_columns=dataset.column_names,
        desc="Tokenizing",
    )


def train():
    """Main training function."""
    cfg = load_training_config()
    model_name = cfg["model"]["name"]
    max_seq_length = cfg["model"].get("max_seq_length", 4096)
    train_cfg = cfg["training"]
    data_cfg = cfg["data"]

    print(f"Model: {model_name}")
    print(f"Max sequence length: {max_seq_length}")

    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # Load model with quantization
    print("Loading model with 4-bit quantization...")
    bnb_config = setup_quantization(cfg)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    model = prepare_model_for_kbit_training(model)

    # Apply LoRA
    print("Applying LoRA adapter...")
    lora_config = setup_lora(cfg)
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Load datasets
    print("Loading datasets...")
    config = Config()
    train_path = str(config.PROJECT_ROOT / data_cfg["train_file"])
    val_path = str(config.PROJECT_ROOT / data_cfg["val_file"])

    train_dataset = load_dataset("json", data_files=train_path, split="train")
    val_dataset = load_dataset("json", data_files=val_path, split="train")

    print(f"Train examples: {len(train_dataset)}")
    print(f"Val examples: {len(val_dataset)}")

    # Tokenize
    print("Tokenizing...")
    train_dataset = tokenize_dataset(train_dataset, tokenizer, max_seq_length)
    val_dataset = tokenize_dataset(val_dataset, tokenizer, max_seq_length)

    # Training arguments
    output_dir = str(config.PROJECT_ROOT / train_cfg.get("output_dir", "./outputs"))
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=train_cfg.get("num_train_epochs", 3),
        per_device_train_batch_size=train_cfg.get("per_device_train_batch_size", 4),
        per_device_eval_batch_size=train_cfg.get("per_device_eval_batch_size", 4),
        gradient_accumulation_steps=train_cfg.get("gradient_accumulation_steps", 4),
        learning_rate=train_cfg.get("learning_rate", 2e-4),
        weight_decay=train_cfg.get("weight_decay", 0.01),
        warmup_ratio=train_cfg.get("warmup_ratio", 0.03),
        lr_scheduler_type=train_cfg.get("lr_scheduler_type", "cosine"),
        logging_steps=train_cfg.get("logging_steps", 10),
        save_strategy=train_cfg.get("save_strategy", "steps"),
        save_steps=train_cfg.get("save_steps", 100),
        eval_strategy=train_cfg.get("eval_strategy", "steps"),
        eval_steps=train_cfg.get("eval_steps", 100),
        save_total_limit=train_cfg.get("save_total_limit", 3),
        fp16=train_cfg.get("fp16", False),
        bf16=train_cfg.get("bf16", True),
        max_grad_norm=train_cfg.get("max_grad_norm", 0.3),
        group_by_length=train_cfg.get("group_by_length", True),
        report_to=train_cfg.get("report_to", "none"),
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
    )

    # Data collator
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
    )

    # Train
    print("\nStarting training...")
    trainer.train()

    # Save final model
    final_path = Path(output_dir) / "final_model"
    print(f"\nSaving final model to {final_path}...")
    trainer.save_model(str(final_path))
    tokenizer.save_pretrained(str(final_path))

    print("Training complete!")


def main():
    train()


if __name__ == "__main__":
    main()
