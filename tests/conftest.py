"""Shared fixtures for the LLM fine-tuning pipeline test suite."""

import json
import sys
from pathlib import Path

import pandas as pd
import pytest
import yaml

# Ensure project root is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


@pytest.fixture
def tmp_data_dir(tmp_path):
    """Provide a temporary directory for test outputs."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    return data_dir


@pytest.fixture
def sample_prose_text():
    """Return a realistic health/fitness paragraph (~200 words) about progressive overload."""
    return (
        "Progressive overload is the cornerstone of effective resistance training. "
        "It refers to the gradual increase in stress placed upon the musculoskeletal "
        "system during exercise. Without progressively challenging the body, adaptation "
        "stalls and strength gains plateau. There are several ways to implement "
        "progressive overload in a training programme. The most straightforward method "
        "is to increase the weight lifted while maintaining proper form. For example, "
        "if a trainee can complete three sets of eight repetitions on the bench press "
        "at 185 pounds with good technique, they might progress to 190 pounds the "
        "following week. Another approach involves adding repetitions or sets at the "
        "same load, thereby increasing total training volume. A trainee who performs "
        "three sets of eight could aim for three sets of ten before raising the weight. "
        "Manipulating rest periods is another variable: reducing rest between sets from "
        "three minutes to two minutes increases metabolic stress and overall workout "
        "density. Tempo manipulation, such as slowing the eccentric phase of a lift to "
        "four seconds, can also heighten mechanical tension. Coaches should periodise "
        "these variables to avoid overtraining and reduce injury risk. Deload weeks, "
        "typically scheduled every four to six weeks, allow recovery while preserving "
        "neuromuscular adaptations. Tracking training volume across mesocycles is "
        "essential for long-term progress."
    )


@pytest.fixture
def sample_training_log_df():
    """Return a pandas DataFrame with 5 rows of training log data."""
    return pd.DataFrame(
        {
            "date": [
                "2025-01-06",
                "2025-01-06",
                "2025-01-06",
                "2025-01-08",
                "2025-01-08",
            ],
            "exercise": [
                "Barbell Back Squat",
                "Romanian Deadlift",
                "Leg Press",
                "Bench Press",
                "Overhead Press",
            ],
            "sets": [4, 3, 3, 4, 3],
            "reps": [6, 10, 12, 5, 8],
            "weight": [315, 225, 450, 245, 135],
            "rpe": [8.5, 7.0, 6.5, 9.0, 7.5],
        }
    )


@pytest.fixture
def sample_extracted_json():
    """Return a dict matching the extraction output format with 3 sections."""
    return {
        "source_id": "test_textbook_001",
        "content_type": "prose",
        "sections": [
            {
                "title": "Hypertrophy Mechanisms",
                "text": (
                    "Skeletal muscle hypertrophy occurs through increases in the cross-sectional "
                    "area of existing muscle fibres. The primary drivers include mechanical "
                    "tension, metabolic stress, and muscle damage. Mechanical tension is "
                    "generated when muscles produce force against external resistance. Higher "
                    "loads produce greater mechanical tension, which activates mechanosensors "
                    "on the sarcolemma. These sensors trigger intracellular signalling cascades "
                    "involving the mTOR pathway, ultimately promoting muscle protein synthesis."
                ),
                "metadata": {"content_type": "prose"},
            },
            {
                "title": "Nutrition for Recovery",
                "text": (
                    "Post-exercise nutrition plays a critical role in recovery and adaptation. "
                    "Consuming adequate protein within the post-workout window supports muscle "
                    "protein synthesis. Research suggests that 20 to 40 grams of high-quality "
                    "protein, such as whey or casein, is sufficient to maximise the anabolic "
                    "response in most individuals. Carbohydrate intake replenishes muscle "
                    "glycogen stores depleted during training."
                ),
                "metadata": {"content_type": "prose"},
            },
            {
                "title": "Sleep and Recovery",
                "text": (
                    "Sleep is a fundamental pillar of athletic recovery. During deep sleep "
                    "stages, the body releases growth hormone, which facilitates tissue repair "
                    "and muscle growth. Adults should aim for seven to nine hours of quality "
                    "sleep per night. Poor sleep has been shown to impair reaction time, reduce "
                    "strength output, and increase perceived exertion during exercise."
                ),
                "metadata": {"content_type": "prose"},
            },
        ],
    }


@pytest.fixture
def sample_chunks_file(tmp_path):
    """Write sample chunks to a temp JSONL file and return the path."""
    chunks = [
        {
            "chunk_id": "test_001_chunk_0000",
            "source_id": "test_001",
            "content_type": "prose",
            "text": (
                "Resistance training is a form of physical activity designed to improve "
                "muscular fitness by exercising a specific muscle or muscle group against "
                "external resistance."
            ),
            "token_count": 30,
            "topics": ["resistance training"],
        },
        {
            "chunk_id": "test_001_chunk_0001",
            "source_id": "test_001",
            "content_type": "prose",
            "text": (
                "Periodisation divides a training programme into distinct phases, each "
                "with specific goals. Common models include linear periodisation, "
                "undulating periodisation, and block periodisation."
            ),
            "token_count": 35,
            "topics": ["periodisation"],
        },
    ]
    file_path = tmp_path / "all_chunks.jsonl"
    with open(file_path, "w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")
    return file_path


@pytest.fixture
def sample_registry_file(tmp_path):
    """Write a minimal registry.yaml to a temp dir and return the path."""
    registry_data = {
        "sources": [
            {
                "id": "textbook_strength",
                "type": "pdf",
                "path": "textbooks/strength_training.pdf",
                "title": "Strength Training Fundamentals",
                "topics": ["strength", "hypertrophy"],
                "priority": "high",
            },
            {
                "id": "log_january",
                "type": "spreadsheet",
                "path": "training_logs/jan_2025.xlsx",
                "title": "January 2025 Training Log",
                "topics": ["training_logs"],
                "priority": "normal",
            },
            {
                "id": "article_nutrition",
                "type": "html",
                "path": "articles/nutrition_guide.html",
                "title": "Complete Nutrition Guide",
                "topics": ["nutrition", "recovery"],
                "priority": "normal",
            },
        ]
    }
    file_path = tmp_path / "registry.yaml"
    with open(file_path, "w", encoding="utf-8") as f:
        yaml.dump(registry_data, f)
    return file_path
