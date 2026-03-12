"""Tests for chunking functions in 02_chunk_text.py."""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Load the chunking module directly from the script file
_chunk_module_path = Path(__file__).resolve().parent.parent / "scripts" / "02_chunk_text.py"
_spec = importlib.util.spec_from_file_location("chunk_text", _chunk_module_path)
_chunk = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_chunk)

count_tokens = _chunk.count_tokens
chunk_prose = _chunk.chunk_prose
chunk_tabular = _chunk.chunk_tabular
process_extracted_file = _chunk.process_extracted_file


class TestCountTokens:
    """Tests for count_tokens."""

    def test_returns_positive_int(self):
        result = count_tokens("Progressive overload is essential for muscle growth.")
        assert isinstance(result, int)
        assert result > 0

    def test_empty_string_returns_zero(self):
        result = count_tokens("")
        assert result == 0

    def test_longer_text_more_tokens(self):
        short = count_tokens("Squat.")
        long = count_tokens(
            "The barbell back squat is a compound movement that targets the "
            "quadriceps, hamstrings, glutes, and core musculature."
        )
        assert long > short


class TestChunkProseBasic:
    """Tests for chunk_prose splitting behaviour."""

    def test_splits_long_text(self, sample_prose_text):
        # Use a small token limit to force splitting
        chunks = chunk_prose(sample_prose_text, max_tokens=50, overlap_tokens=10)
        assert len(chunks) > 1
        # Each chunk should be non-empty
        for chunk in chunks:
            assert len(chunk.strip()) > 0

    def test_all_content_preserved(self, sample_prose_text):
        chunks = chunk_prose(sample_prose_text, max_tokens=50, overlap_tokens=10)
        # First and last sentences should appear somewhere in the chunks
        assert "Progressive overload" in chunks[0]
        assert "long-term progress" in chunks[-1]


class TestChunkProseShort:
    """Tests for chunk_prose with text shorter than max_tokens."""

    def test_no_split_needed(self):
        short_text = "Bench press builds upper body strength. It targets the pectorals."
        chunks = chunk_prose(short_text, max_tokens=800, overlap_tokens=100)
        assert len(chunks) == 1
        assert chunks[0] == short_text


class TestChunkProseOverlap:
    """Tests for overlap between consecutive chunks."""

    def test_overlap_content_shared(self, sample_prose_text):
        chunks = chunk_prose(sample_prose_text, max_tokens=80, overlap_tokens=30)
        if len(chunks) >= 2:
            # The end of chunk 0 should share some words with the start of chunk 1
            words_end_of_first = set(chunks[0].split()[-15:])
            words_start_of_second = set(chunks[1].split()[:15])
            overlap = words_end_of_first & words_start_of_second
            assert len(overlap) > 0, (
                "Expected overlapping content between consecutive chunks"
            )


class TestChunkTabular:
    """Tests for chunk_tabular."""

    def test_groups_sections(self):
        sections = [
            {"text": f"Session {i}: Squat 3x5 at {200 + i * 10} lbs."}
            for i in range(12)
        ]
        chunks = chunk_tabular(sections, max_sections_per_chunk=5)
        # 12 sections / 5 per chunk = 3 chunks (5, 5, 2)
        assert len(chunks) == 3

    def test_single_section_per_chunk(self):
        sections = [{"text": "Only one session here."}]
        chunks = chunk_tabular(sections, max_sections_per_chunk=5)
        assert len(chunks) == 1
        assert "Only one session" in chunks[0]

    def test_chunk_text_joins_sections(self):
        sections = [
            {"text": "Session A: Squat day."},
            {"text": "Session B: Bench day."},
        ]
        chunks = chunk_tabular(sections, max_sections_per_chunk=5)
        assert len(chunks) == 1
        assert "Session A" in chunks[0]
        assert "Session B" in chunks[0]


class TestProcessExtractedFile:
    """Tests for process_extracted_file end-to-end."""

    def test_prose_file(self, tmp_path, sample_extracted_json):
        file_path = tmp_path / "test_textbook_001.json"
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(sample_extracted_json, f)

        result = process_extracted_file(file_path, max_tokens=800)
        assert len(result) >= 1
        for chunk in result:
            assert "chunk_id" in chunk
            assert chunk["source_id"] == "test_textbook_001"
            assert "text" in chunk
            assert "token_count" in chunk
            assert chunk["token_count"] > 0

    def test_tabular_file(self, tmp_path):
        tabular_data = {
            "source_id": "training_log_jan",
            "content_type": "tabular",
            "sections": [
                {"text": f"Session {i}: Deadlift 3x5.", "metadata": {"content_type": "tabular"}}
                for i in range(8)
            ],
        }
        file_path = tmp_path / "training_log_jan.json"
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(tabular_data, f)

        result = process_extracted_file(file_path)
        assert len(result) >= 1
        for chunk in result:
            assert chunk["content_type"] == "tabular"
            assert chunk["source_id"] == "training_log_jan"

    def test_empty_sections_skipped(self, tmp_path):
        data = {
            "source_id": "sparse_doc",
            "content_type": "prose",
            "sections": [
                {"title": "Empty", "text": "", "metadata": {"content_type": "prose"}},
                {"title": "Also Empty", "text": "   ", "metadata": {"content_type": "prose"}},
                {
                    "title": "Real Content",
                    "text": "Resistance training is effective for building strength and muscle mass.",
                    "metadata": {"content_type": "prose"},
                },
            ],
        }
        file_path = tmp_path / "sparse_doc.json"
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f)

        result = process_extracted_file(file_path)
        # Only the non-empty section should produce chunks
        assert len(result) == 1
        assert "Resistance training" in result[0]["text"]
