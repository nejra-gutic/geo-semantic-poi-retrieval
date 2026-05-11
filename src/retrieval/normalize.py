"""
retrieval/normalize.py
----------------------
Text normalization for NLP/search pipeline.
Applied to both POI text fields and user queries before matching.

Steps:
  1. lowercase
  2. remove punctuation
  3. remove stopwords (custom - preserves spatial/relational words)
  4. lemmatization (spaCy)

Note:
  Spatial words like "in", "at", "on", "near" are preserved
  because they carry important meaning in GeoNLP queries.
  e.g. "coffee near burnside" -> ["coffee", "near", "burnside"]
"""

import re
import pandas as pd
import spacy

nlp = spacy.load("en_core_web_sm")

# Spatial/relational words that carry meaning in GeoNLP context
SPATIAL_WORDS = {"in", "at", "on", "near", "by", "next", "around", "between", "off"}

# Default spaCy stopwords minus spatial words
STOPWORDS = nlp.Defaults.stop_words - SPATIAL_WORDS


def normalize(text: str) -> str | None:
    """
    Normalize a single text string for NLP matching.
    Returns None for missing/unknown values.
    """
    if pd.isna(text) or str(text).strip().lower() in ("", "unknown"):
        return None

    text = str(text).lower()
    text = re.sub(r"[^a-z0-9\s/]", " ", text)    # remove punctuation
    text = re.sub(r"\s+", " ", text).strip()      # collapse whitespace

    doc = nlp(text)
    tokens = [
        token.lemma_
        for token in doc
        if token.text not in STOPWORDS and token.text.strip() != ""
    ]

    return " ".join(tokens) if tokens else None


def add_normalized(df: pd.DataFrame, col: str = "poi_text") -> pd.DataFrame:
    """
    Add a normalized text column to the dataframe.
    Default source column is poi_text.
    Output column: {col}_lemma
    """
    df = df.copy()
    if col not in df.columns:
        print(f"[retrieval.normalize] Warning - '{col}' not found, skipping")
        df[f"{col}_lemma"] = None
        return df

    out_col = f"{col}_lemma"
    df[out_col] = df[col].apply(normalize)

    filled = df[out_col].notna().sum()
    print(f"[retrieval.normalize] Normalized: {filled} / {len(df)} rows")
    return df


def run(df: pd.DataFrame) -> pd.DataFrame:
    return add_normalized(df)