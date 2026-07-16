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
    bm25_index: BM25Okapi,
    df: pd.DataFrame,
    top_k: int = 10,
    intent_model=None,
    intent_vectorizer=None,
    user_lat: float = None,
    user_lon: float = None,
    check_time=None,
    apply_geo: bool = True,
    apply_temporal: bool = True,
) -> pd.DataFrame:
    """
    apply_geo: if False, skips geo re-ranking. Use False when this output
    will be fused into a Hybrid result and geo will be applied once
    after fusion.

    apply_temporal: if False, skips the temporal/open_now boost step.
    Set to False when this function's output is going to be merged into a
    Hybrid result (e.g. in app.py), so the temporal boost is applied exactly
    once -- after the merge -- instead of once here AND once again downstream.
    """
    from src.retrieval.common import (
        apply_intent_boost,
        get_query_core,
        apply_boolean_filters,
        apply_temporal_filter,
        apply_geo_reranking,
    )
    from src.retrieval.query import parse_filters, RESULT_COLS

    # 0. Guard: df mora biti u istom redoslijedu/dužini kao BM25 indeks
    if len(df) != len(bm25_index.doc_freqs):
        raise ValueError(
            f"[bm25] df length ({len(df)}) != index length ({len(bm25_index.doc_freqs)}) "
            f"— je li df filtriran/sortiran nakon build_bm25()?"
        )

    # 1. Strip temporal phrases before BM25 scoring
    query_core = get_query_core(query)
    print(f"[bm25] Original:   '{query}'")
    print(f"[bm25] Core:       '{query_core}'")

    # 2. BM25 scoring on ALL POIs (soft boost mode)
    print(f"[bm25] Searching all {len(df)} POIs (soft boost mode)")
    tokenized_query = _tokenize_query(query)
    all_scores = bm25_index.get_scores(tokenized_query)

    top_indices = np.argsort(all_scores)[::-1][:top_k * 5]
    results = df.iloc[top_indices].copy()
    results["bm25_score"] = all_scores[top_indices]

    # 3. Soft boost
    results = apply_intent_boost(
        query, df, results, score_col="bm25_score",
        intent_model=intent_model, intent_vectorizer=intent_vectorizer
    )

    # 4. Boolean filters
    filters = parse_filters(query)
    if filters:
        print(f"[bm25] Filters detected: {filters}")
    results = apply_boolean_filters(results, filters)

    # Keep only relevant columns
    existing_cols = [col for col in RESULT_COLS if col in results.columns]
    results = results[existing_cols + ["bm25_score"]]

    # 5. GEO FIRST (optional; disable when feeding a Hybrid fusion)
    if apply_geo:
        results = apply_geo_reranking(
            results, query, user_lat, user_lon, score_col="bm25_score"
        )

    # 6. TEMP LAST (skip if this call feeds into a Hybrid merge downstream)
    if apply_temporal:
        score_col = "combined_score" if "combined_score" in results.columns else "bm25_score"
        results = apply_temporal_filter(results, query, filters, check_time=check_time, score_col=score_col)

    results = results.head(top_k)
    print(f"[bm25] Results found: {len(results)}")
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