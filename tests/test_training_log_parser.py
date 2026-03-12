"""Tests for TrainingLogParser utility."""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.utils.training_log_parser import TrainingLogParser


class TestNormalizeColumns:
    """Tests for TrainingLogParser.normalize_columns."""

    def test_standard_aliases_mapped(self):
        df = pd.DataFrame(
            {
                "workout_date": ["2025-01-06"],
                "exercise_name": ["Squat"],
                "num_sets": [4],
                "repetitions": [8],
                "load": [225],
                "intensity": [7.5],
            }
        )
        result = TrainingLogParser.normalize_columns(df)
        assert "date" in result.columns
        assert "exercise" in result.columns
        assert "sets" in result.columns
        assert "reps" in result.columns
        assert "weight" in result.columns
        assert "rpe" in result.columns

    def test_already_standard_names_unchanged(self):
        df = pd.DataFrame(
            {
                "date": ["2025-01-06"],
                "exercise": ["Bench Press"],
                "sets": [3],
                "reps": [10],
            }
        )
        result = TrainingLogParser.normalize_columns(df)
        assert list(result.columns) == ["date", "exercise", "sets", "reps"]

    def test_unrecognised_columns_preserved(self):
        df = pd.DataFrame(
            {
                "date": ["2025-01-06"],
                "exercise": ["Squat"],
                "custom_field": ["value"],
            }
        )
        result = TrainingLogParser.normalize_columns(df)
        assert "custom_field" in result.columns


class TestNarrateRow:
    """Tests for TrainingLogParser.narrate_row."""

    def test_full_row_with_all_fields(self):
        row = {
            "date": "2025-01-06",
            "exercise": "Barbell Back Squat",
            "sets": 4,
            "reps": 6,
            "weight": 315,
            "rpe": 8.5,
        }
        result = TrainingLogParser.narrate_row(row)
        assert "January 06, 2025" in result
        assert "Barbell Back Squat" in result
        assert "4 sets" in result
        assert "6 reps" in result
        assert "315 lbs" in result
        assert "8.5" in result
        assert "high difficulty" in result
        assert result.endswith(".")

    def test_minimal_row_exercise_and_sets_only(self):
        row = {
            "exercise": "Pull-ups",
            "sets": 3,
            "reps": None,
            "weight": None,
            "rpe": None,
        }
        result = TrainingLogParser.narrate_row(row)
        assert "Pull-ups" in result
        assert "3 sets" in result
        # Should not contain lbs or RPE info
        assert "lbs" not in result
        assert "RPE" not in result

    def test_exercise_only(self):
        row = {
            "exercise": "Foam Rolling",
            "sets": None,
            "reps": None,
            "weight": None,
            "rpe": None,
        }
        result = TrainingLogParser.narrate_row(row)
        assert "Foam Rolling" in result
        assert "performed" in result.lower()

    def test_sentence_capitalized(self):
        row = {"exercise": "Deadlift", "sets": 1, "reps": 5, "weight": 405, "rpe": 9.5}
        result = TrainingLogParser.narrate_row(row)
        assert result[0].isupper()

    def test_notes_included(self):
        row = {
            "exercise": "Squat",
            "sets": 3,
            "reps": 8,
            "weight": 225,
            "rpe": None,
            "notes": "Felt strong today",
        }
        result = TrainingLogParser.narrate_row(row)
        assert "Felt strong today" in result


class TestNarrateSession:
    """Tests for TrainingLogParser.narrate_session."""

    def test_multiple_rows_with_date(self):
        rows = [
            {"exercise": "Squat", "sets": 4, "reps": 6, "weight": 315, "rpe": 8.0},
            {"exercise": "Leg Press", "sets": 3, "reps": 12, "weight": 450, "rpe": 6.5},
        ]
        result = TrainingLogParser.narrate_session(rows, session_date="January 06, 2025")
        assert "January 06, 2025" in result
        assert "Squat" in result
        assert "Leg Press" in result
        # Summary should mention exercise count and total sets
        assert "2 exercises" in result
        assert "7 total sets" in result

    def test_session_without_date(self):
        rows = [
            {"exercise": "Bench Press", "sets": 4, "reps": 5, "weight": 245, "rpe": 9.0},
        ]
        result = TrainingLogParser.narrate_session(rows)
        assert result.startswith("Training session:")
        assert "Bench Press" in result

    def test_empty_session_returns_empty_string(self):
        result = TrainingLogParser.narrate_session([])
        assert result == ""


class TestDataframeToNarratives:
    """Tests for TrainingLogParser.dataframe_to_narratives."""

    def test_groups_by_date(self, sample_training_log_df):
        sections = TrainingLogParser.dataframe_to_narratives(sample_training_log_df)
        # Should have 2 sections (2 distinct dates)
        assert len(sections) == 2
        # Each section should have required keys
        for section in sections:
            assert "title" in section
            assert "text" in section
            assert "metadata" in section
            assert section["metadata"]["content_type"] == "tabular"

    def test_section_titles_contain_dates(self, sample_training_log_df):
        sections = TrainingLogParser.dataframe_to_narratives(sample_training_log_df)
        titles = [s["title"] for s in sections]
        assert any("January 06" in t for t in titles)
        assert any("January 08" in t for t in titles)

    def test_no_date_grouping(self, sample_training_log_df):
        sections = TrainingLogParser.dataframe_to_narratives(
            sample_training_log_df, group_by_date=False
        )
        # Should produce one section per row
        assert len(sections) == 5


class TestRpeDescriptions:
    """Tests for RPE-to-description mapping in narrate_row."""

    @pytest.mark.parametrize(
        "rpe_value,expected_description",
        [
            (4.0, "low difficulty"),
            (5.0, "low difficulty"),
            (6.0, "moderate difficulty"),
            (7.0, "moderate difficulty"),
            (8.0, "high difficulty"),
            (8.5, "high difficulty"),
            (9.0, "near-maximal effort"),
            (10.0, "near-maximal effort"),
        ],
    )
    def test_rpe_description(self, rpe_value, expected_description):
        row = {"exercise": "Squat", "sets": 3, "reps": 5, "weight": 300, "rpe": rpe_value}
        result = TrainingLogParser.narrate_row(row)
        assert expected_description in result
