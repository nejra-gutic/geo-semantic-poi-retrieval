"""
retrieval/tfidf.py
------------------
TF-IDF and n-gram representation of POI text fields.
Builds TF-IDF matrix for semantic text matching.

Uses scikit-learn TfidfVectorizer.
TF-IDF preferred over BOW because it downweights common terms
and highlights distinctive terms per POI.
"""

import pandas as pd
import pickle
import numpy as np
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def build_tfidf(
    df: pd.DataFrame,
    col: str = "poi_text_lemma",
    max_features: int = 5000,
    min_df: int = 2,
    ngram_range: tuple = (1, 2),
) -> tuple:
    """
    Build TF-IDF matrix from a text column.

    Args:
        df:           input dataframe
        col:          text column to vectorize
        max_features: maximum vocabulary size
        min_df:       minimum document frequency
        ngram_range:  (1,2) includes unigrams and bigrams

    Returns:
        (vectorizer, tfidf_matrix)
    """
    if col not in df.columns:
        print(f"[tfidf] Warning - '{col}' not found, falling back to poi_text")
        col = "poi_text"

    corpus = df[col].fillna("").tolist()

    vectorizer = TfidfVectorizer(
        max_features=max_features,
        min_df=min_df,
        ngram_range=ngram_range,
        token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z]+\b",
    )

    tfidf_matrix = vectorizer.fit_transform(corpus)
    vocabulary = vectorizer.get_feature_names_out()

    print(f"[tfidf] Corpus size:    {len(corpus)}")
    print(f"[tfidf] Vocabulary size: {len(vocabulary)}")
    print(f"[tfidf] TF-IDF matrix:   {tfidf_matrix.shape}")
    print(f"[tfidf] Top 20 terms:    {vocabulary[:20].tolist()}")

    return vectorizer, tfidf_matrix


def search_tfidf(
    query: str,
    vectorizer: TfidfVectorizer,
    tfidf_matrix,
    df: pd.DataFrame,
    top_k: int = 10,
) -> pd.DataFrame:
    """
    Search POI dataset using TF-IDF cosine similarity.

    Args:
        query:        user input string
        vectorizer:   fitted TfidfVectorizer
        tfidf_matrix: fitted TF-IDF matrix
        df:           cleaned POI dataframe
        top_k:        number of results to return

    Returns:
        DataFrame of top_k matching POIs with similarity scores
    """
    query_vec = vectorizer.transform([query])
    scores = cosine_similarity(query_vec, tfidf_matrix).flatten()

    top_indices = np.argsort(scores)[::-1][:top_k]
    results = df.iloc[top_indices].copy()
    results["similarity_score"] = scores[top_indices]

    print(f"[tfidf] Query: '{query}'")
    print(f"[tfidf] Top {top_k} results (similarity scores):")
    print(results[["name", "category_final", "similarity_score"]].to_string())

    return results


def save_vectorizer(vectorizer, path: str = "models/tfidf_vectorizer.pkl") -> None:
    """Save fitted TfidfVectorizer to disk."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(vectorizer, f)
    print(f"[tfidf] Vectorizer saved: {path}")


def load_vectorizer(path: str = "models/tfidf_vectorizer.pkl") -> TfidfVectorizer:
    """Load fitted TfidfVectorizer from disk."""
    with open(path, "rb") as f:
        vectorizer = pickle.load(f)
    print(f"[tfidf] Vectorizer loaded: {path}")
    return vectorizer


def run(df: pd.DataFrame) -> tuple:
    return build_tfidf(df)