"""
normalize.py
------------
Text normalization: unidecode, lowercase, regex cleaning.
Step 2 in the preprocessing pipeline.

Applied AFTER clean.py (operates on *_clean columns).
"""

import re
import pandas as pd
from unidecode import unidecode


def normalize_text(text: str | None) -> str | None:
    """
    General text normalization:
    - unidecode (handles accented chars, Bosnian/Croatian letters, etc.)
    - lowercase
    - hyphen → space
    - remove punctuation (but keep /)
    - collapse whitespace

    NOTE: Does NOT touch opening_hours (handled separately).
    """
    if text is None:
        return None

    text = str(text)
    text = unidecode(text)        # đ→d, š→s, ć→c, č→c, ž→z, etc.
    text = text.lower()
    text = text.replace("-", " ")
    text = re.sub(r"[^\w\s/]", "", text)   # remove punct, keep /
    text = re.sub(r"\s+", " ", text).strip()

    return text if text else None


def normalize_name(text: str | None) -> str | None:
    """
    Name normalization - same as general but preserves 24/7 pattern.
    """
    return normalize_text(text)


def normalize_cuisine(text: str | None) -> str | None:
    """
    Cuisine: normalize and replace underscores with spaces
    (OSM uses 'fast_food' style).
    """
    normalized = normalize_text(text)
    if normalized is None:
        return None
    return normalized.replace("_", " ")


def normalize_addr(text: str | None) -> str | None:
    """
    Address normalization - same as general.
    libpostal/usaddress parsing happens in address.py (after this step).
    """
    return normalize_text(text)


def normalize_category(text: str | None) -> str | None:
    """
    Category: normalize + replace underscores with spaces.
    """
    normalized = normalize_text(text)
    if normalized is None:
        return None
    return normalized.replace("_", " ")


def apply_normalization(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies normalization to all *_clean columns.
    Returns df with new *_norm columns.

    Input columns expected (from clean.py):
        name_clean, cuisine_clean, category_clean, addr_clean
    Output columns added:
        name_norm, cuisine_norm, category_norm, addr_norm
    """
    df = df.copy()

    col_map = {
        "name_clean": ("name_norm", normalize_name),
        "cuisine_clean": ("cuisine_norm", normalize_cuisine),
        "category_clean": ("category_norm", normalize_category),
        "addr_clean": ("addr_norm", normalize_addr),
    }

    for src_col, (dst_col, fn) in col_map.items():
        if src_col in df.columns:
            df[dst_col] = df[src_col].apply(fn)

    return df
