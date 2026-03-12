"""Central configuration for the LLM fine-tuning pipeline."""

import os
from pathlib import Path

import yaml


class Config:
    """Loads and provides access to all project configuration."""

    # Project root is two levels up from this file (scripts/utils/config.py -> project root)
    PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

    # Directory paths
    SOURCES_DIR = PROJECT_ROOT / "sources"
    SCRIPTS_DIR = PROJECT_ROOT / "scripts"
    DATA_DIR = PROJECT_ROOT / "data"
    CONFIGS_DIR = PROJECT_ROOT / "configs"

    # Data subdirectories
    EXTRACTED_DIR = DATA_DIR / "extracted"
    CHUNKS_DIR = DATA_DIR / "chunks"
    GENERATED_DIR = DATA_DIR / "generated"
    VALIDATED_DIR = DATA_DIR / "validated"
    TRAINING_DIR = DATA_DIR / "training"

    # Source subdirectories
    TEXTBOOKS_DIR = SOURCES_DIR / "textbooks"
    ARTICLES_DIR = SOURCES_DIR / "articles"
    TRAINING_LOGS_DIR = SOURCES_DIR / "training_logs"

    # Registry
    REGISTRY_PATH = SOURCES_DIR / "registry.yaml"

    def __init__(self):
        self.generation_config = self._load_yaml(self.CONFIGS_DIR / "generation_config.yaml")
        self.training_config = self._load_yaml(self.CONFIGS_DIR / "training_config.yaml")

    @staticmethod
    def _load_yaml(path: Path) -> dict:
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f)
        return {}

    @property
    def anthropic_api_key(self) -> str:
        key = os.environ.get("ANTHROPIC_API_KEY", "")
        if not key:
            raise ValueError(
                "ANTHROPIC_API_KEY environment variable is not set. "
                "Set it before running generation scripts."
            )
        return key

    @property
    def llm_model(self) -> str:
        return self.generation_config.get("llm", {}).get("model", "claude-sonnet-4-20250514")

    @property
    def llm_max_tokens(self) -> int:
        return self.generation_config.get("llm", {}).get("max_tokens", 2048)

    @property
    def llm_temperature(self) -> float:
        return self.generation_config.get("llm", {}).get("temperature", 0.7)

    def get_prompt(self, prompt_name: str) -> str:
        return self.generation_config.get("prompts", {}).get(prompt_name, "")

    @property
    def topics(self) -> list[str]:
        return self.generation_config.get("topics", [])

    def get_generation_count(self, format_type: str) -> int:
        return self.generation_config.get("generation", {}).get(
            f"{format_type}_per_chunk", 3
        )

    def ensure_directories(self):
        """Create all required data directories if they don't exist."""
        dirs = [
            self.EXTRACTED_DIR / "textbooks",
            self.EXTRACTED_DIR / "articles",
            self.EXTRACTED_DIR / "training_logs",
            self.CHUNKS_DIR,
            self.GENERATED_DIR / "qa_pairs",
            self.GENERATED_DIR / "conversations",
            self.GENERATED_DIR / "completions",
            self.GENERATED_DIR / "classification",
            self.VALIDATED_DIR,
            self.TRAINING_DIR,
        ]
        for d in dirs:
            d.mkdir(parents=True, exist_ok=True)
