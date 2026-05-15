"""
retrieval/bm25.py
-----------------
BM25 ranking as an alternative to TF-IDF.
BM25 accounts for document length and term frequency saturation,
making it generally better for search than TF-IDF.

Requires: pip install rank-bm25
"""

import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi



def build_bm25(df: pd.DataFrame, col: str = "poi_text_lemma") -> BM25Okapi:
    """Build BM25 index from a text column."""
    if col not in df.columns:
        print(f"[bm25] Warning - '{col}' not found, falling back to poi_text")
        col = "poi_text"

    corpus = df[col].fillna("").apply(lambda x: x.split()).tolist()
    bm25 = BM25Okapi(corpus)

    print(f"[bm25] Index built on {len(corpus)} documents")
    return bm25


def search_bm25(
    query: str,
    bm25: BM25Okapi,
    df: pd.DataFrame,
    top_k: int = 10,
    df_filtered: pd.DataFrame = None,
) -> pd.DataFrame:
    tokens = query.lower().split()
    
    if df_filtered is not None:
        # rebuild BM25 on filtered subset
        corpus = df_filtered["poi_text_lemma"].fillna("").apply(lambda x: x.split()).tolist()
        bm25_filtered = BM25Okapi(corpus)
        scores = bm25_filtered.get_scores(tokens)
        results = df_filtered.iloc[np.argsort(scores)[::-1][:top_k]].copy()
        results["bm25_score"] = scores[np.argsort(scores)[::-1][:top_k]]
    else:
        scores = bm25.get_scores(tokens)
        results = df.iloc[np.argsort(scores)[::-1][:top_k]].copy()
        results["bm25_score"] = scores[np.argsort(scores)[::-1][:top_k]]

    print(f"[bm25] Query: '{query}' → {len(results)} results")
    return results


def compare_bm25_tfidf(
    query: str,
    bm25: BM25Okapi,
    vectorizer,
    tfidf_matrix,
    df: pd.DataFrame,
    intent_model=None,
    intent_vectorizer=None,
    top_k: int = 5,
) -> None:
    from sklearn.metrics.pairwise import cosine_similarity
    from src.retrieval.intent_classifier import predict

    print(f"\n{'='*50}")
    print(f"Query: '{query}'")

    # Intent filter
    df_filtered = df.copy()
    if intent_model and intent_vectorizer:
        from src.retrieval.query import INTENT_TO_CATEGORY
        intent, confidence = predict(query, intent_model, intent_vectorizer)
        print(f"Intent: {intent} ({confidence}%)")
        categories = INTENT_TO_CATEGORY.get(intent)
        if categories:
            df_filtered = df[df["category_final"].isin(categories)]
            print(f"Filtered: {len(df_filtered)} POIs")

    # BM25 na filtriranom subsetu
    tokens = query.lower().split()
    corpus = df_filtered["poi_text_lemma"].fillna("").apply(lambda x: x.split()).tolist()
    bm25_filtered = BM25Okapi(corpus)
    bm25_scores = bm25_filtered.get_scores(tokens)
    bm25_top = np.argsort(bm25_scores)[::-1][:top_k]

    print(f"\nBM25 top {top_k}:")
    for i in bm25_top:
        print(f"  {df_filtered.iloc[i]['name']} | {df_filtered.iloc[i]['category_final']} | score: {round(bm25_scores[i], 3)}")

    # TF-IDF na filtriranom subsetu
    filtered_indices = [df.index.get_loc(idx) for idx in df_filtered.index]
    tfidf_subset = tfidf_matrix[filtered_indices]
    query_vec = vectorizer.transform([query])
    tfidf_scores = cosine_similarity(query_vec, tfidf_subset).flatten()
    tfidf_top = np.argsort(tfidf_scores)[::-1][:top_k]

    print(f"\nTF-IDF top {top_k}:")
    for i in tfidf_top:
        print(f"  {df_filtered.iloc[i]['name']} | {df_filtered.iloc[i]['category_final']} | score: {round(tfidf_scores[i], 3)}")


def run(df: pd.DataFrame) -> BM25Okapi:
    return build_bm25(df)