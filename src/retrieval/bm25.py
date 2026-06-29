"""
retrieval/bm25.py
-----------------
BM25 ranking as an alternative to TF-IDF.

Requires: pip install rank-bm25
"""

import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi
from src.retrieval.normalize import normalize
from src.retrieval.query import expand_query_synonyms, extract_temporal_phrase


BM25_STOPWORDS = {
    "me", "find", "place", "places", "close", "by", "nearby", "near"
}


def _make_ngrams(tokens: list[str]) -> list[str]:
    """Add bigrams and trigrams to token list."""
    bigrams = [f"{tokens[i]}_{tokens[i+1]}" for i in range(len(tokens) - 1)]
    trigrams = [
        f"{tokens[i]}_{tokens[i+1]}_{tokens[i+2]}"
        for i in range(len(tokens) - 2)
    ]
    return tokens + bigrams + trigrams



def _tokenize_text(text: str) -> list[str]:
    # poi_text_lemma je već normaliziran — samo split, bez normalize()
    tokens = [t for t in text.lower().split() if len(t) > 1 and t not in BM25_STOPWORDS]
    return _make_ngrams(tokens)


def _tokenize_query(query: str) -> list[str]:
    query_clean = extract_temporal_phrase(expand_query_synonyms(query))
    normalized = normalize(query_clean) or query_clean.lower()

    tokens = [
        t for t in normalized.split()
        if len(t) > 1 and t not in BM25_STOPWORDS
    ]

    return _make_ngrams(tokens)


def build_bm25(df: pd.DataFrame, col: str = "poi_text_lemma") -> BM25Okapi:
    if col not in df.columns:
        print(f"[bm25] Warning - '{col}' not found, falling back to poi_text")
        col = "poi_text"

    corpus = df[col].fillna("").apply(_tokenize_text).tolist()
    bm25 = BM25Okapi(corpus, k1=1.5, b=0.75)
    print(f"[bm25] Index built on {len(corpus)} documents")
    return bm25


def search_bm25(
    query: str,
    bm25: BM25Okapi,
    df: pd.DataFrame,
    top_k: int = 10,
    df_filtered: pd.DataFrame = None,
) -> pd.DataFrame:
    tokens = _tokenize_query(query)

    search_df = df_filtered.copy() if df_filtered is not None else df.copy()

    if search_df.empty:
        print(f"[bm25] Query: '{query}' → 0 results")
        return search_df.copy()

    # Score cijeli korpus jednom - NE rebuildat BM25 na filtriranom subsetu.
    # IDF mora biti računat na svih 24918 dokumenata, ne na filtriranom subsetu.
    all_scores = bm25.get_scores(tokens)

    # Izvuci skorove samo za filtrirane redove prema njihovim pozicijama u df.
    filtered_positions = [df.index.get_loc(idx) for idx in search_df.index]
    scores = all_scores[filtered_positions]

    candidate_k = min(len(scores), max(top_k * 20, top_k))
    top_idx = np.argsort(scores)[::-1][:candidate_k]

    results = search_df.iloc[top_idx].copy()
    results["bm25_score"] = scores[top_idx]

    # Deduplicate known names, but keep multiple unknown POIs.
    if "name" in results.columns:
        known = results[results["name"] != "unknown"].drop_duplicates(subset=["name"])
        unknown = results[results["name"] == "unknown"]
        results = pd.concat([known, unknown], ignore_index=False)

    results = results.head(top_k)

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
    from src.retrieval.query import INTENT_TO_CATEGORY, detect_specific_category

    print(f"\n{'=' * 50}")
    print(f"Query: '{query}'")

    df_filtered = df.copy()

    if intent_model and intent_vectorizer:
        intent, confidence = predict(query, intent_model, intent_vectorizer)
        print(f"Intent: {intent} ({confidence}%)")

        specific_categories = detect_specific_category(query)

        if specific_categories:
            categories = specific_categories
            print(f"Specific category override: {categories}")
        else:
            categories = INTENT_TO_CATEGORY.get(intent)

        if categories:
            df_filtered = df[df["category_final"].isin(categories)]
            print(f"Filtered: {len(df_filtered)} POIs")

    tokens = _tokenize_query(query)
    all_scores = bm25.get_scores(tokens)
    filtered_positions = [df.index.get_loc(idx) for idx in df_filtered.index]
    bm25_scores = all_scores[filtered_positions]

    candidate_k = min(len(bm25_scores), top_k * 10)
    bm25_top = np.argsort(bm25_scores)[::-1][:candidate_k]

    print(f"\nBM25 top {top_k}:")
    for i in bm25_top[:top_k]:
        print(
            f"  {df_filtered.iloc[i]['name']} | "
            f"{df_filtered.iloc[i]['category_final']} | "
            f"score: {round(bm25_scores[i], 3)}"
        )

    filtered_indices = [df.index.get_loc(idx) for idx in df_filtered.index]
    tfidf_subset = tfidf_matrix[filtered_indices]
    query_vec = vectorizer.transform([normalize(query) or query])
    tfidf_scores = cosine_similarity(query_vec, tfidf_subset).flatten()
    tfidf_top = np.argsort(tfidf_scores)[::-1][:top_k]

    print(f"\nTF-IDF top {top_k}:")
    for i in tfidf_top:
        print(
            f"  {df_filtered.iloc[i]['name']} | "
            f"{df_filtered.iloc[i]['category_final']} | "
            f"score: {round(tfidf_scores[i], 3)}"
        )


def tune_bm25(
    df: pd.DataFrame,
    evaluation_queries: list,
    intent_model=None,
    intent_vectorizer=None,
    col: str = "poi_text_lemma",
) -> None:
    from src.retrieval.intent_classifier import predict
    from src.retrieval.query import INTENT_TO_CATEGORY, detect_specific_category, parse_filters

    k1_values = [0.5, 1.0, 1.2, 1.5, 2.0]
    b_values  = [0.25, 0.5, 0.75, 1.0]

    corpus = df[col].fillna("").apply(_tokenize_text).tolist()
    best_score = -1
    best_params = {}

    for k1 in k1_values:
        for b in b_values:
            bm25 = BM25Okapi(corpus, k1=k1, b=b)

            precisions = []
            for item in evaluation_queries:
                q = item["query"]
                expected = item["expected_category"]

                # Intent filtering
                df_filtered = df.copy()
                if intent_model and intent_vectorizer:
                    intent, _ = predict(q, intent_model, intent_vectorizer)
                    specific_categories = detect_specific_category(q)
                    categories = specific_categories or INTENT_TO_CATEGORY.get(intent)
                    if categories:
                        df_filtered = df_filtered[df_filtered["category_final"].isin(categories)]

                # Boolean filters
                filters = parse_filters(q)
                for col_f, val in filters.items():
                    if col_f in df_filtered.columns:
                        df_filtered = df_filtered[df_filtered[col_f] == val]

                if df_filtered.empty:
                    precisions.append(0.0)
                    continue

                tokens = _tokenize_query(q)
                all_scores = bm25.get_scores(tokens)
                filtered_positions = [df.index.get_loc(idx) for idx in df_filtered.index]
                scores = all_scores[filtered_positions]

                candidate_k = min(len(scores), max(5 * 20, 5))
                top_idx = np.argsort(scores)[::-1][:candidate_k]
                results = df_filtered.iloc[top_idx].copy()

                # Dedupliciranje — isto kao u search_bm25
                if "name" in results.columns:
                    known = results[results["name"] != "unknown"].drop_duplicates(subset=["name"])
                    unknown = results[results["name"] == "unknown"]
                    results = pd.concat([known, unknown], ignore_index=False)

                results = results.head(5)
                p = (results["category_final"] == expected).sum() / 5
                precisions.append(p)

            mean_p = np.mean(precisions)
            print(f"k1={k1}, b={b} → Precision@5: {mean_p:.3f}")

            if mean_p > best_score:
                best_score = mean_p
                best_params = {"k1": k1, "b": b}

    print(f"\nBest params: {best_params} → Precision@5: {best_score:.3f}")

def run(df: pd.DataFrame) -> BM25Okapi:
    return build_bm25(df)