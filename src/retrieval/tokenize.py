"""
retrieval/tokenize.py
---------------------
Tokenization of POI text fields and user queries.
Uses spaCy as the primary tokenizer (best tradeoff between
semantic preservation and flexibility for further processing).

Tested approaches:
  - split: too naive, preserves punctuation
  - regex: breaks meaningful forms like "McDonald's", "24/7"
  - spaCy: handles edge cases best, selected as final approach
"""

import spacy
import pandas as pd

# Load spaCy model once at module level
nlp = spacy.load("en_core_web_sm")


def tokenize(text: str) -> list[str]:
    """
    Tokenize a single text string using spaCy.
    Returns empty list for missing/unknown values.
    """
    if pd.isna(text) or str(text).strip().lower() in ("", "unknown"):
        return []
    doc = nlp(str(text))
    return [token.text for token in doc]


def tokenize_series(series: pd.Series) -> pd.Series:
    """
    Tokenize a pandas Series of text values.
    Returns a Series of token lists.
    """
    return series.apply(tokenize)


def add_tokens(df: pd.DataFrame, col: str = "poi_text") -> pd.DataFrame:
    """
    Add a tokens column to the dataframe based on a text column.
    Default source column is poi_text.
    """
    df = df.copy()
    if col not in df.columns:
        print(f"[tokenize] Warning - '{col}' not found, skipping")
        df["tokens"] = [[] for _ in range(len(df))]
        return df

    df["tokens"] = tokenize_series(df[col])
    filled = df["tokens"].apply(len).gt(0).sum()
    print(f"[tokenize] Tokenized: {filled} / {len(df)} rows")
    return df


def run(df: pd.DataFrame) -> pd.DataFrame:
    return add_tokens(df)