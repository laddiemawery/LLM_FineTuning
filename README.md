# Health & Fitness LLM Fine-Tuning Pipeline

A complete pipeline for fine-tuning Llama 3 on health and fitness domain knowledge — covering exercise science, strength & conditioning, nutrition, physical therapy, occupational therapy, and recovery.

## Overview

This project takes raw source materials (textbooks, research papers, training logs, etc.), extracts and normalizes the text, generates diverse training datasets via LLM, and prepares the data for fine-tuning with HuggingFace + PEFT/LoRA.

```
Sources → Extract → Chunk → Generate (LLM) → Validate → Format → Train
```

## Supported Source Types

| Type | Formats | Extractor |
|------|---------|-----------|
| Textbooks | EPUB, PDF | `extract_epub.py`, `extract_pdf.py` |
| Articles & Papers | PDF, HTML | `extract_pdf.py`, `extract_html.py` |
| Scanned / Handwritten Logs | JPG, PNG, TIFF | `extract_ocr.py` (Tesseract / EasyOCR) |
| Spreadsheet Logs | XLSX, CSV, TSV | `extract_spreadsheet.py` |
| Database Exports | SQLite, JSON | `extract_database.py` |

## Generated Dataset Formats

| Format | Description | Use Case |
|--------|-------------|----------|
| QA Pairs | Instruction + response | Factual recall, application, reasoning |
| Conversations | Multi-turn user/assistant | Coaching and consultation scenarios |
| Completions | Prompt + completion | Knowledge continuation |
| Classification | Text + label | Topic categorization |

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

For OCR support (scanned documents / handwritten logs), also install [Tesseract](https://github.com/tesseract-ocr/tesseract):
- **Windows:** Download installer from [UB Mannheim](https://github.com/UB-Mannheim/tesseract/wiki)
- **macOS:** `brew install tesseract`
- **Linux:** `sudo apt install tesseract-ocr`

### 2. Add Sources

Place your source files in the appropriate subdirectory:

```
sources/
├── textbooks/          # EPUB, PDF textbooks
├── articles/           # Research papers, articles
└── training_logs/
    ├── images/         # Photos of handwritten logs
    ├── spreadsheets/   # Excel, CSV files
    ├── databases/      # SQLite, JSON exports
    └── other/
```

Then register each source in `sources/registry.yaml`:

```yaml
sources:
  - id: my_textbook           # Unique identifier
    type: epub                 # epub | pdf | html | image_ocr | spreadsheet | csv | xlsx | database | sqlite | json_db
    path: textbooks/my_book.epub  # Relative to sources/
    topics:                    # Tag with relevant domains
      - strength_conditioning
      - nutrition
    priority: high             # high | normal | low
    notes: "Optional description"
```

### 3. Set Up API Key

The dataset generation steps use the Claude API:

```bash
export ANTHROPIC_API_KEY="your-api-key-here"
```

### 4. Run the Pipeline

**Full pipeline:**
```bash
python scripts/run_pipeline.py
```

**Step by step:**
```bash
# Extract text from all registered sources
python scripts/run_pipeline.py --steps 1

# Chunk extracted text
python scripts/run_pipeline.py --steps 2

# Generate datasets (use --sample for testing)
python scripts/run_pipeline.py --steps 3,4,5,6 --sample 5

# Validate and prepare training data
python scripts/run_pipeline.py --steps 7,8
```

**Single source only:**
```bash
python scripts/run_pipeline.py --steps 1 --source cscs_exam_prep
```

**List available steps:**
```bash
python scripts/run_pipeline.py --list-steps
```

### 5. Train the Model

```bash
python training/train.py
```

Training uses QLoRA (4-bit quantization + LoRA) by default. Configuration is in `configs/training_config.yaml`.

### 6. Evaluate

```bash
# Run evaluation with sample prompts
python training/evaluate.py outputs/final_model

# Interactive mode
python training/evaluate.py outputs/final_model --interactive
```

## Pipeline Steps

| Step | Script | Description |
|------|--------|-------------|
| 1 | `scripts/01_extract/extract_runner.py` | Extract text from all sources via registry |
| 2 | `scripts/02_chunk_text.py` | Split into semantic chunks (~800 tokens) |
| 3 | `scripts/03_generate_qa.py` | Generate instruction/response QA pairs |
| 4 | `scripts/04_generate_conversations.py` | Generate multi-turn coaching dialogues |
| 5 | `scripts/05_generate_completions.py` | Generate prompt/completion pairs |
| 6 | `scripts/06_generate_classification.py` | Generate topic classification examples |
| 7 | `scripts/07_validate_dataset.py` | Deduplicate and quality-check datasets |
| 8 | `scripts/08_prepare_training.py` | Format for Llama 3 + train/val/test split |

## Project Structure

```
LLM_FineTuning/
├── configs/
│   ├── generation_config.yaml    # LLM prompts, generation settings
│   └── training_config.yaml      # Model, LoRA, and training hyperparameters
├── data/                         # Generated data (gitignored)
│   ├── extracted/                # Raw extracted text
│   ├── chunks/                   # Chunked text
│   ├── generated/                # LLM-generated datasets
│   ├── validated/                # Quality-checked datasets
│   └── training/                 # Final train/val/test splits
├── scripts/
│   ├── 01_extract/               # Source-specific extractors
│   ├── 02-08_*.py                # Pipeline steps
│   ├── run_pipeline.py           # Pipeline orchestrator
│   └── utils/                    # Shared utilities
├── sources/                      # Raw source materials (gitignored)
│   ├── registry.yaml             # Source manifest
│   ├── textbooks/
│   ├── articles/
│   └── training_logs/
├── training/
│   ├── train.py                  # Fine-tuning with HF + PEFT/LoRA
│   └── evaluate.py               # Model evaluation + inference
└── requirements.txt
```

## Configuration

### Generation Config (`configs/generation_config.yaml`)

- **LLM settings:** model, temperature, max tokens
- **Generation counts:** examples per chunk per format
- **Prompt templates:** customizable system and user prompts for each dataset type
- **Domain topics:** classification labels for the health/fitness domain

### Training Config (`configs/training_config.yaml`)

- **Model:** base model name, sequence length, quantization
- **LoRA:** rank, alpha, dropout, target modules
- **Training:** epochs, batch size, learning rate, scheduler, etc.

## Adding New Source Types

1. Create a new extractor in `scripts/01_extract/extract_<type>.py`
2. It must return the standard format:
   ```python
   {
       "source_id": "...",
       "source_type": "...",
       "content_type": "prose|tabular|mixed",
       "sections": [{"title": "...", "text": "...", "metadata": {...}}]
   }
   ```
3. Register the type in `scripts/utils/source_registry.py` (`EXTRACTOR_MAP`)
4. Import and add it to `scripts/01_extract/extract_runner.py` (`EXTRACTORS`)

## Training Log Handling

Structured data (spreadsheets, databases) is automatically converted into natural language narratives. For example:

**Raw CSV row:**
```
date,exercise,sets,reps,weight,rpe
2024-01-15,squat,5,5,315,8
```

**Narrated output:**
> On January 15, 2024, the client performed 5 sets of 5 reps of squat at 315 lbs with an RPE of 8, indicating high difficulty.

This narration is handled by `scripts/utils/training_log_parser.py` and can be customized.
