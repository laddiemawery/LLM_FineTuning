"""Tests for DatasetValidator in 07_validate_dataset.py."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Load the validation module from the script file
import importlib.util

_validate_module_path = (
    Path(__file__).resolve().parent.parent / "scripts" / "07_validate_dataset.py"
)


def _load_validate_module():
    spec = importlib.util.spec_from_file_location("validate_dataset", _validate_module_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_val = _load_validate_module()
DatasetValidator = _val.DatasetValidator


class TestValidateQa:
    """Tests for DatasetValidator.validate_qa."""

    def test_valid_qa_pair(self):
        v = DatasetValidator()
        item = {
            "instruction": "What is the recommended rep range for hypertrophy training?",
            "response": (
                "For hypertrophy, research supports a moderate rep range of 6 to 12 "
                "repetitions per set. This range provides sufficient mechanical tension "
                "and metabolic stress to stimulate muscle protein synthesis and promote "
                "increases in cross-sectional area of muscle fibres."
            ),
        }
        assert v.validate_qa(item) is True

    def test_empty_instruction_rejected(self):
        v = DatasetValidator()
        item = {
            "instruction": "",
            "response": "Some valid response with enough words to pass the minimum.",
        }
        assert v.validate_qa(item) is False

    def test_empty_response_rejected(self):
        v = DatasetValidator()
        item = {
            "instruction": "What is progressive overload?",
            "response": "",
        }
        assert v.validate_qa(item) is False

    def test_whitespace_only_rejected(self):
        v = DatasetValidator()
        item = {
            "instruction": "   ",
            "response": "   ",
        }
        assert v.validate_qa(item) is False

    def test_too_short_instruction(self):
        v = DatasetValidator()
        item = {
            "instruction": "Squat?",
            "response": (
                "The squat is a compound lower body exercise that targets the "
                "quadriceps, hamstrings, glutes, and core stabilisers."
            ),
        }
        # "Squat?" is only 1 word, needs at least 3
        assert v.validate_qa(item) is False

    def test_too_short_response(self):
        v = DatasetValidator(min_response_words=10)
        item = {
            "instruction": "What is the best exercise for legs?",
            "response": "Squats are good.",
        }
        # 3 words < 10 minimum
        assert v.validate_qa(item) is False

    def test_identical_instruction_response_rejected(self):
        v = DatasetValidator()
        text = "What is the recommended rep range for hypertrophy training in beginners?"
        item = {"instruction": text, "response": text}
        assert v.validate_qa(item) is False


class TestValidateConversation:
    """Tests for DatasetValidator.validate_conversation."""

    def test_valid_conversation(self):
        v = DatasetValidator()
        item = {
            "messages": [
                {"role": "user", "content": "How should I warm up before squatting heavy?"},
                {
                    "role": "assistant",
                    "content": (
                        "Start with 5 minutes of light cardio such as cycling or rowing "
                        "to raise your core temperature. Then perform dynamic stretches "
                        "targeting the hips, ankles, and thoracic spine. Follow with "
                        "progressively heavier warm-up sets."
                    ),
                },
            ]
        }
        assert v.validate_conversation(item) is True

    def test_wrong_role_order(self):
        v = DatasetValidator()
        item = {
            "messages": [
                {"role": "assistant", "content": "Hello, how can I help?"},
                {"role": "user", "content": "Tell me about deadlifts."},
            ]
        }
        # First message should be "user", not "assistant"
        assert v.validate_conversation(item) is False

    def test_missing_role_key(self):
        v = DatasetValidator()
        item = {
            "messages": [
                {"content": "What is RPE?"},
                {"role": "assistant", "content": "RPE stands for Rate of Perceived Exertion."},
            ]
        }
        assert v.validate_conversation(item) is False

    def test_empty_content_rejected(self):
        v = DatasetValidator()
        item = {
            "messages": [
                {"role": "user", "content": ""},
                {"role": "assistant", "content": "Response here."},
            ]
        }
        assert v.validate_conversation(item) is False

    def test_single_message_rejected(self):
        v = DatasetValidator()
        item = {
            "messages": [
                {"role": "user", "content": "How do I bench press?"},
            ]
        }
        assert v.validate_conversation(item) is False

    def test_four_turn_conversation(self):
        v = DatasetValidator()
        item = {
            "messages": [
                {"role": "user", "content": "What is a good starting weight for squats?"},
                {"role": "assistant", "content": "For most beginners, starting with just the barbell at 45 pounds is recommended."},
                {"role": "user", "content": "How quickly should I add weight?"},
                {"role": "assistant", "content": "A common approach is to add 5 pounds per session as long as form stays solid."},
            ]
        }
        assert v.validate_conversation(item) is True


class TestValidateCompletion:
    """Tests for DatasetValidator.validate_completion."""

    def test_valid_completion(self):
        v = DatasetValidator()
        item = {
            "prompt": "Explain the benefits of compound exercises.",
            "completion": (
                "Compound exercises recruit multiple muscle groups simultaneously, making "
                "them highly efficient for building overall strength and muscle mass. "
                "Examples include squats, deadlifts, bench press, and overhead press."
            ),
        }
        assert v.validate_completion(item) is True

    def test_empty_prompt_rejected(self):
        v = DatasetValidator()
        item = {
            "prompt": "",
            "completion": "Some completion text with enough words to pass validation here.",
        }
        assert v.validate_completion(item) is False

    def test_too_short_completion(self):
        v = DatasetValidator(min_response_words=10)
        item = {
            "prompt": "Describe progressive overload.",
            "completion": "Lift more.",
        }
        assert v.validate_completion(item) is False


class TestValidateClassification:
    """Tests for DatasetValidator.validate_classification."""

    def test_valid_classification(self):
        v = DatasetValidator()
        item = {
            "text": (
                "The client performed 4 sets of 6 reps on the barbell back squat "
                "at 315 pounds with an RPE of 8.5."
            ),
            "label": "strength_training",
        }
        assert v.validate_classification(item) is True

    def test_empty_text_rejected(self):
        v = DatasetValidator()
        item = {"text": "", "label": "nutrition"}
        assert v.validate_classification(item) is False

    def test_empty_label_rejected(self):
        v = DatasetValidator()
        item = {
            "text": "The athlete consumed 30 grams of protein post-workout.",
            "label": "",
        }
        assert v.validate_classification(item) is False

    def test_too_short_text_rejected(self):
        v = DatasetValidator()
        item = {"text": "Squat day", "label": "training"}
        # Only 2 words, need at least 5
        assert v.validate_classification(item) is False


class TestDeduplicate:
    """Tests for DatasetValidator.deduplicate."""

    def test_near_duplicates_removed(self):
        v = DatasetValidator(dedup_threshold=85.0)
        items = [
            {
                "instruction": "What is the recommended rep range for hypertrophy?",
                "response": "Six to twelve reps per set is generally recommended.",
            },
            {
                "instruction": "What is the recommended repetition range for hypertrophy?",
                "response": "Most research points to 6-12 reps per set.",
            },
            {
                "instruction": "How does creatine supplementation improve performance?",
                "response": "Creatine increases phosphocreatine stores in muscle cells.",
            },
        ]
        result = v.deduplicate(items, dataset_type="qa_pairs")
        # The first two are near-duplicates; one should be removed
        assert len(result) == 2
        assert v.stats["duplicates_removed"] == 1

    def test_distinct_items_all_kept(self):
        v = DatasetValidator(dedup_threshold=85.0)
        items = [
            {"instruction": "What is progressive overload?", "response": "A training principle."},
            {"instruction": "How does sleep affect recovery?", "response": "Sleep is critical."},
            {"instruction": "What role does nutrition play?", "response": "Nutrition fuels growth."},
        ]
        result = v.deduplicate(items, dataset_type="qa_pairs")
        assert len(result) == 3
        assert v.stats["duplicates_removed"] == 0

    def test_empty_list_returns_empty(self):
        v = DatasetValidator()
        result = v.deduplicate([], dataset_type="qa_pairs")
        assert result == []

    def test_conversation_dedup_uses_first_message(self):
        v = DatasetValidator(dedup_threshold=85.0)
        items = [
            {
                "messages": [
                    {"role": "user", "content": "How should I warm up before squatting?"},
                    {"role": "assistant", "content": "Start with dynamic stretches."},
                ]
            },
            {
                "messages": [
                    {"role": "user", "content": "How should I warm up before squatting heavy?"},
                    {"role": "assistant", "content": "Begin with mobility work."},
                ]
            },
        ]
        result = v.deduplicate(items, dataset_type="conversations")
        # These are near-duplicates based on the first user message
        assert len(result) == 1
