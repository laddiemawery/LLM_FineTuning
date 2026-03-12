"""Tests for TextCleaner utility."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.utils.text_cleaning import TextCleaner


class TestNormalizeUnicode:
    """Tests for TextCleaner.normalize_unicode."""

    def test_smart_single_quotes(self):
        text = "\u2018hello\u2019"
        result = TextCleaner.normalize_unicode(text)
        assert result == "'hello'"

    def test_smart_double_quotes(self):
        text = "\u201chello\u201d"
        result = TextCleaner.normalize_unicode(text)
        assert result == '"hello"'

    def test_en_dash(self):
        text = "pages 10\u201320"
        result = TextCleaner.normalize_unicode(text)
        assert result == "pages 10-20"

    def test_em_dash(self):
        text = "training\u2014especially squats\u2014is important"
        result = TextCleaner.normalize_unicode(text)
        assert result == "training--especially squats--is important"

    def test_ellipsis(self):
        text = "and so on\u2026"
        result = TextCleaner.normalize_unicode(text)
        assert result == "and so on..."

    def test_non_breaking_space(self):
        text = "100\u00a0kg"
        result = TextCleaner.normalize_unicode(text)
        assert result == "100 kg"

    def test_zero_width_space_removed(self):
        text = "dead\u200blift"
        result = TextCleaner.normalize_unicode(text)
        assert result == "deadlift"

    def test_plain_ascii_unchanged(self):
        text = "Bench press 225 lbs for 5 reps."
        result = TextCleaner.normalize_unicode(text)
        assert result == text


class TestCleanWhitespace:
    """Tests for TextCleaner.clean_whitespace."""

    def test_multiple_spaces_collapsed(self):
        text = "squat   with   good   form"
        result = TextCleaner.clean_whitespace(text)
        assert result == "squat with good form"

    def test_tabs_collapsed(self):
        text = "exercise\t\tsets\t\treps"
        result = TextCleaner.clean_whitespace(text)
        assert result == "exercise sets reps"

    def test_excessive_newlines_collapsed(self):
        text = "Section one.\n\n\n\n\nSection two."
        result = TextCleaner.clean_whitespace(text)
        assert result == "Section one.\n\nSection two."

    def test_double_newline_preserved(self):
        text = "Paragraph one.\n\nParagraph two."
        result = TextCleaner.clean_whitespace(text)
        assert result == "Paragraph one.\n\nParagraph two."

    def test_leading_trailing_whitespace_stripped(self):
        text = "   progressive overload   "
        result = TextCleaner.clean_whitespace(text)
        assert result == "progressive overload"


class TestRemovePageArtifacts:
    """Tests for TextCleaner.remove_page_artifacts."""

    def test_standalone_page_number_removed(self):
        text = "Some content.\n42\nMore content."
        result = TextCleaner.remove_page_artifacts(text)
        assert "42" not in result
        assert "Some content." in result
        assert "More content." in result

    def test_number_in_sentence_preserved(self):
        text = "Perform 42 reps of bodyweight squats."
        result = TextCleaner.remove_page_artifacts(text)
        assert "42" in result

    def test_chapter_header_removed(self):
        text = "Previous section.\nChapter 3\nNew section starts."
        result = TextCleaner.remove_page_artifacts(text)
        assert "Chapter 3" not in result

    def test_chapter_header_uppercase_removed(self):
        text = "Previous section.\nCHAPTER 12\nNew section starts."
        result = TextCleaner.remove_page_artifacts(text)
        assert "CHAPTER 12" not in result


class TestCleanHtmlResidue:
    """Tests for TextCleaner.clean_html_residue."""

    def test_html_tags_removed(self):
        text = "<p>Progressive overload</p> is <b>key</b>."
        result = TextCleaner.clean_html_residue(text)
        assert "<p>" not in result
        assert "</p>" not in result
        assert "<b>" not in result
        assert "Progressive overload" in result

    def test_named_entities_replaced(self):
        text = "Squat&amp;Deadlift &gt; isolation"
        result = TextCleaner.clean_html_residue(text)
        assert "&amp;" not in result
        assert "&gt;" not in result

    def test_numeric_entities_replaced(self):
        text = "Training&#8212;nutrition"
        result = TextCleaner.clean_html_residue(text)
        assert "&#8212;" not in result

    def test_plain_text_unchanged(self):
        text = "Load the bar with 135 lbs."
        result = TextCleaner.clean_html_residue(text)
        assert result == text


class TestFullClean:
    """Tests for TextCleaner.full_clean integration."""

    def test_combines_all_steps(self):
        text = (
            "  <b>\u201cProgressive overload\u201d</b>  is\t\t key.\n\n\n\n\n"
            "42\n"
            "Train&amp;recover.  "
        )
        result = TextCleaner.full_clean(text)
        # Unicode quotes normalized
        assert "\u201c" not in result
        assert "\u201d" not in result
        # HTML cleaned
        assert "<b>" not in result
        assert "&amp;" not in result
        # Whitespace normalized
        assert "\t" not in result
        # Standalone page number removed
        assert result.count("42") == 0 or "42" not in result.split("\n")
        # Content preserved
        assert "Progressive overload" in result
        assert "key" in result

    def test_references_section_removed_when_requested(self):
        text = (
            "Muscle growth requires progressive overload.\n"
            "References\n"
            "1. Smith et al. (2020). Journal of Strength.\n"
            "2. Jones (2019). Sports Science Review."
        )
        result = TextCleaner.full_clean(text, remove_references=True)
        assert "Smith" not in result
        assert "progressive overload" in result

    def test_references_kept_by_default(self):
        text = (
            "Muscle growth requires progressive overload.\n"
            "References\n"
            "1. Smith et al. (2020)."
        )
        result = TextCleaner.full_clean(text, remove_references=False)
        assert "Smith" in result


class TestIsMeaningfulText:
    """Tests for TextCleaner.is_meaningful_text."""

    def test_short_text_not_meaningful(self):
        assert TextCleaner.is_meaningful_text("Too short.") is False

    def test_numbers_only_not_meaningful(self):
        text = "100 200 300 400 500 600 700 800 900 1000 1100"
        assert TextCleaner.is_meaningful_text(text) is False

    def test_good_text_is_meaningful(self):
        text = (
            "Progressive overload is the gradual increase of stress placed on "
            "the body during resistance training to stimulate muscle growth."
        )
        assert TextCleaner.is_meaningful_text(text) is True

    def test_custom_min_words(self):
        text = "Squats build legs"
        assert TextCleaner.is_meaningful_text(text, min_words=3) is True
        assert TextCleaner.is_meaningful_text(text, min_words=5) is False

    def test_mixed_content_below_alpha_threshold(self):
        # More than half the words are pure numbers/symbols
        text = "123 456 789 012 345 678 901 234 567 890 exercise done"
        assert TextCleaner.is_meaningful_text(text) is False
