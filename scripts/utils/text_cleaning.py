"""Shared text cleaning and normalization utilities."""

import re
import unicodedata


class TextCleaner:
    """Clean and normalize extracted text from various sources."""

    @staticmethod
    def normalize_unicode(text: str) -> str:
        """Normalize unicode characters to their closest ASCII equivalent."""
        text = unicodedata.normalize("NFKD", text)
        # Replace common unicode characters
        replacements = {
            "\u2018": "'", "\u2019": "'",  # smart quotes
            "\u201c": '"', "\u201d": '"',
            "\u2013": "-", "\u2014": "--",  # dashes
            "\u2026": "...",  # ellipsis
            "\u00a0": " ",  # non-breaking space
            "\u200b": "",  # zero-width space
        }
        for old, new in replacements.items():
            text = text.replace(old, new)
        return text

    @staticmethod
    def clean_whitespace(text: str) -> str:
        """Normalize whitespace: collapse multiple spaces/newlines."""
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    @staticmethod
    def remove_page_artifacts(text: str) -> str:
        """Remove common page artifacts like headers, footers, page numbers."""
        # Remove standalone page numbers
        text = re.sub(r"^\s*\d+\s*$", "", text, flags=re.MULTILINE)
        # Remove common header/footer patterns
        text = re.sub(r"^\s*(Chapter|CHAPTER)\s+\d+\s*$", "", text, flags=re.MULTILINE)
        return text

    @staticmethod
    def remove_references_section(text: str) -> str:
        """Remove bibliography/references sections that aren't useful for training."""
        patterns = [
            r"\n(References|REFERENCES|Bibliography|BIBLIOGRAPHY)\s*\n[\s\S]*$",
            r"\n(Works Cited|WORKS CITED)\s*\n[\s\S]*$",
        ]
        for pattern in patterns:
            text = re.sub(pattern, "", text)
        return text

    @staticmethod
    def clean_html_residue(text: str) -> str:
        """Remove any residual HTML tags or entities."""
        text = re.sub(r"<[^>]+>", "", text)
        text = re.sub(r"&[a-zA-Z]+;", " ", text)
        text = re.sub(r"&#\d+;", " ", text)
        return text

    @classmethod
    def full_clean(cls, text: str, remove_references: bool = False) -> str:
        """Apply all cleaning steps in order."""
        text = cls.normalize_unicode(text)
        text = cls.clean_html_residue(text)
        text = cls.remove_page_artifacts(text)
        if remove_references:
            text = cls.remove_references_section(text)
        text = cls.clean_whitespace(text)
        return text

    @staticmethod
    def is_meaningful_text(text: str, min_words: int = 10) -> bool:
        """Check if text has enough meaningful content to be worth processing."""
        words = text.split()
        if len(words) < min_words:
            return False
        # Check that it's not just numbers or special characters
        alpha_ratio = sum(1 for w in words if any(c.isalpha() for c in w)) / len(words)
        return alpha_ratio > 0.5
