# Lazy imports to avoid requiring all dependencies for every script.
# Usage: from scripts.utils.config import Config (direct import preferred)


def __getattr__(name):
    if name == "Config":
        from .config import Config
        return Config
    elif name == "LLMClient":
        from .llm_client import LLMClient
        return LLMClient
    elif name == "TextCleaner":
        from .text_cleaning import TextCleaner
        return TextCleaner
    elif name == "SourceRegistry":
        from .source_registry import SourceRegistry
        return SourceRegistry
    elif name == "TrainingLogParser":
        from .training_log_parser import TrainingLogParser
        return TrainingLogParser
    raise AttributeError(f"module 'scripts.utils' has no attribute {name}")
