"""Metadata normalization used for matching (original fields are kept untouched)."""
import re

from unidecode import unidecode


def normalize_title(title):
    """Lowercase, strip accents/punctuation, collapse spaces — for matching only."""
    if not isinstance(title, str):
        return ""
    title = unidecode(title.lower().strip())
    title = re.sub(r"[^\w\s]", " ", title)
    return re.sub(r"\s+", " ", title).strip()


def normalize_author(author):
    """Same idea as normalize_title, for author strings."""
    if not isinstance(author, str):
        return ""
    author = re.sub(r"[^\w\s]", "", unidecode(author.lower()))
    return " ".join(author.split())


def normalize_isbn(isbn):
    """Keep only digits and X; return None if empty after cleaning."""
    if not isinstance(isbn, str):
        return None
    isbn = re.sub(r"[^0-9Xx]", "", isbn).upper()
    return isbn or None
