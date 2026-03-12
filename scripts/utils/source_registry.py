"""Parse and manage the source registry (sources/registry.yaml)."""

from dataclasses import dataclass, field
from pathlib import Path

import yaml

from .config import Config


@dataclass
class Source:
    """Represents a single source entry in the registry."""
    id: str
    type: str
    path: str
    title: str = ""
    topics: list[str] = field(default_factory=list)
    priority: str = "normal"
    notes: str = ""
    format_notes: str = ""

    @property
    def full_path(self) -> Path:
        """Return the full filesystem path relative to sources/."""
        return Config.SOURCES_DIR / self.path

    @property
    def exists(self) -> bool:
        return self.full_path.exists()


# Map source types to extractor module names
EXTRACTOR_MAP = {
    "epub": "extract_epub",
    "pdf": "extract_pdf",
    "html": "extract_html",
    "image_ocr": "extract_ocr",
    "spreadsheet": "extract_spreadsheet",
    "csv": "extract_spreadsheet",
    "xlsx": "extract_spreadsheet",
    "tsv": "extract_spreadsheet",
    "database": "extract_database",
    "sqlite": "extract_database",
    "json_db": "extract_database",
}


class SourceRegistry:
    """Load and query the source registry."""

    def __init__(self, registry_path: Path | None = None):
        self.path = registry_path or Config.REGISTRY_PATH
        self.sources: list[Source] = []
        self._load()

    def _load(self):
        if not self.path.exists():
            return
        with open(self.path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        for entry in data.get("sources", []):
            self.sources.append(Source(**entry))

    def get_all(self) -> list[Source]:
        return self.sources

    def get_by_id(self, source_id: str) -> Source | None:
        for s in self.sources:
            if s.id == source_id:
                return s
        return None

    def get_by_type(self, source_type: str) -> list[Source]:
        return [s for s in self.sources if s.type == source_type]

    def get_by_topic(self, topic: str) -> list[Source]:
        return [s for s in self.sources if topic in s.topics]

    def get_extractor_name(self, source: Source) -> str | None:
        return EXTRACTOR_MAP.get(source.type)

    def validate(self) -> list[str]:
        """Check all registered sources exist on disk. Returns list of errors."""
        errors = []
        for s in self.sources:
            if not s.exists:
                errors.append(f"Source '{s.id}' not found at: {s.full_path}")
            if s.type not in EXTRACTOR_MAP:
                errors.append(f"Source '{s.id}' has unknown type: {s.type}")
        return errors
