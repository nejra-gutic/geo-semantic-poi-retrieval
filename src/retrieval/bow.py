"""
retrieval/bow.py
----------------
Bag of Words (BOW) representation of POI text fields.
Builds vocabulary from the corpus and creates BOW vectors.

Uses scikit-learn CountVectorizer.
"""

import pandas as pd
import pickle
from pathlib import Path
from sklearn.feature_extraction.text import CountVectorizer


def build_bow(
    df: pd.DataFrame,
    col: str = "poi_text_lemma",
    max_features: int = 5000,
    min_df: int = 2,
) -> tuple:
    """
    Build BOW matrix and vocabulary from a text column.

    Args:
        df:           input dataframe
        col:          text column to vectorize
        max_features: maximum vocabulary size
        min_df:       minimum document frequency for a term

    Returns:
        (vectorizer, bow_matrix, vocabulary)
    """
    if col not in df.columns:
        print(f"[bow] Warning - '{col}' not found, falling back to poi_text")
        col = "poi_text"

    corpus = df[col].fillna("").tolist()

    vectorizer = CountVectorizer(
        max_features=max_features,
        min_df=min_df,
        ngram_range=(1, 1),
    )

    bow_matrix = vectorizer.fit_transform(corpus)
    vocabulary = vectorizer.get_feature_names_out()

    print(f"[bow] Corpus size:    {len(corpus)}")
    print(f"[bow] Vocabulary size: {len(vocabulary)}")
    print(f"[bow] BOW matrix:      {bow_matrix.shape}")
    print(f"[bow] Top 20 terms:    {vocabulary[:20].tolist()}")

    return vectorizer, bow_matrix, vocabulary


def save_vectorizer(vectorizer, path: str = "models/bow_vectorizer.pkl") -> None:
    """Save fitted CountVectorizer to disk."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(vectorizer, f)
    print(f"[bow] Vectorizer saved: {path}")


def load_vectorizer(path: str = "models/bow_vectorizer.pkl") -> CountVectorizer:
    """Load fitted CountVectorizer from disk."""
    with open(path, "rb") as f:
        vectorizer = pickle.load(f)
    print(f"[bow] Vectorizer loaded: {path}")
    return vectorizer


def run(df: pd.DataFrame) -> tuple:
    return build_bow(df)