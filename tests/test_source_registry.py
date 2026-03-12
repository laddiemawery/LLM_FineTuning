"""Tests for SourceRegistry utility."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.utils.source_registry import SourceRegistry, Source, EXTRACTOR_MAP


class TestLoadRegistry:
    """Tests for SourceRegistry loading from YAML."""

    def test_load_registry(self, sample_registry_file):
        registry = SourceRegistry(registry_path=sample_registry_file)
        assert len(registry.sources) == 3
        assert all(isinstance(s, Source) for s in registry.sources)

    def test_load_nonexistent_file(self, tmp_path):
        missing = tmp_path / "does_not_exist.yaml"
        registry = SourceRegistry(registry_path=missing)
        assert len(registry.sources) == 0


class TestGetById:
    """Tests for SourceRegistry.get_by_id."""

    def test_found(self, sample_registry_file):
        registry = SourceRegistry(registry_path=sample_registry_file)
        source = registry.get_by_id("textbook_strength")
        assert source is not None
        assert source.id == "textbook_strength"
        assert source.type == "pdf"
        assert source.title == "Strength Training Fundamentals"

    def test_not_found(self, sample_registry_file):
        registry = SourceRegistry(registry_path=sample_registry_file)
        source = registry.get_by_id("nonexistent_source")
        assert source is None


class TestGetByType:
    """Tests for SourceRegistry.get_by_type."""

    def test_single_match(self, sample_registry_file):
        registry = SourceRegistry(registry_path=sample_registry_file)
        pdfs = registry.get_by_type("pdf")
        assert len(pdfs) == 1
        assert pdfs[0].id == "textbook_strength"

    def test_no_match(self, sample_registry_file):
        registry = SourceRegistry(registry_path=sample_registry_file)
        epubs = registry.get_by_type("epub")
        assert len(epubs) == 0

    def test_spreadsheet_match(self, sample_registry_file):
        registry = SourceRegistry(registry_path=sample_registry_file)
        sheets = registry.get_by_type("spreadsheet")
        assert len(sheets) == 1
        assert sheets[0].id == "log_january"


class TestGetByTopic:
    """Tests for SourceRegistry.get_by_topic."""

    def test_single_topic_match(self, sample_registry_file):
        registry = SourceRegistry(registry_path=sample_registry_file)
        results = registry.get_by_topic("nutrition")
        assert len(results) == 1
        assert results[0].id == "article_nutrition"

    def test_shared_topic(self, sample_registry_file):
        registry = SourceRegistry(registry_path=sample_registry_file)
        results = registry.get_by_topic("hypertrophy")
        assert len(results) == 1
        assert results[0].id == "textbook_strength"

    def test_no_topic_match(self, sample_registry_file):
        registry = SourceRegistry(registry_path=sample_registry_file)
        results = registry.get_by_topic("cardiology")
        assert len(results) == 0


class TestValidate:
    """Tests for SourceRegistry.validate."""

    def test_missing_file_reported(self, sample_registry_file):
        registry = SourceRegistry(registry_path=sample_registry_file)
        errors = registry.validate()
        # All source files are missing because they are relative to Config.SOURCES_DIR
        missing_errors = [e for e in errors if "not found" in e]
        assert len(missing_errors) >= 1

    def test_unknown_type_reported(self, tmp_path):
        """A source with an unrecognised type should produce a validation error."""
        import yaml

        registry_data = {
            "sources": [
                {
                    "id": "weird_source",
                    "type": "floppy_disk",
                    "path": "data/something.flp",
                    "title": "Mystery Data",
                    "topics": [],
                }
            ]
        }
        path = tmp_path / "registry.yaml"
        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(registry_data, f)

        registry = SourceRegistry(registry_path=path)
        errors = registry.validate()
        type_errors = [e for e in errors if "unknown type" in e]
        assert len(type_errors) == 1
        assert "floppy_disk" in type_errors[0]


class TestExtractorMap:
    """Tests for the EXTRACTOR_MAP constant."""

    def test_known_types_have_extractors(self):
        expected_types = {"epub", "pdf", "html", "image_ocr", "spreadsheet", "csv",
                          "xlsx", "tsv", "database", "sqlite", "json_db"}
        assert expected_types == set(EXTRACTOR_MAP.keys())

    def test_get_extractor_name(self, sample_registry_file):
        registry = SourceRegistry(registry_path=sample_registry_file)
        pdf_source = registry.get_by_id("textbook_strength")
        assert registry.get_extractor_name(pdf_source) == "extract_pdf"

        html_source = registry.get_by_id("article_nutrition")
        assert registry.get_extractor_name(html_source) == "extract_html"

        sheet_source = registry.get_by_id("log_january")
        assert registry.get_extractor_name(sheet_source) == "extract_spreadsheet"
