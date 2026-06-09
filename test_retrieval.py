import pandas as pd
import numpy as np

from src.utils.io import load_csv
from src.retrieval import pipeline, query, tfidf
from src.retrieval.intent_classifier import load_model, predict
from src.retrieval.query import INTENT_TO_CATEGORY, parse_filters
from src.retrieval.tfidf import compare_ngrams
from src.retrieval.bm25 import build_bm25, search_bm25, compare_bm25_tfidf, tune_bm25
from src.retrieval.embeddings import (
    load_embedding_model,
    get_or_build_embeddings,
    search_embeddings,
)


def precision_at_k(results: pd.DataFrame, expected_category: str, k: int = 5) -> float:
    top_k = results.head(k)
    relevant = top_k["category_final"] == expected_category
    return relevant.sum() / k


def hit_at_k(results: pd.DataFrame, expected_category: str, k: int = 5) -> int:
    top_k = results.head(k)
    relevant = top_k["category_final"] == expected_category
    return 1 if relevant.any() else 0


def recall_at_k(
    results: pd.DataFrame,
    expected_category: str,
    total_relevant: int,
    k: int = 5,
) -> float:
    top_k = results.head(k)
    relevant_found = (top_k["category_final"] == expected_category).sum()

    if total_relevant == 0:
        return 0.0

    return relevant_found / total_relevant


def ndcg_at_k(results: pd.DataFrame, expected_category: str, k: int = 5) -> float:
    top_k = results.head(k)

    relevance = [
        1 if cat == expected_category else 0
        for cat in top_k["category_final"]
    ]

    if not relevance or sum(relevance) == 0:
        return 0.0

    dcg = 0.0
    for i, rel in enumerate(relevance):
        dcg += rel / np.log2(i + 2)

    ideal_relevance = sorted(relevance, reverse=True)

    idcg = 0.0
    for i, rel in enumerate(ideal_relevance):
        idcg += rel / np.log2(i + 2)

    if idcg == 0:
        return 0.0

    return dcg / idcg


def print_metric_summary(name, precision_scores, hit_scores, recall_scores, ndcg_scores, k):
    print(f"\n--- {name} Average ---")
    print(f"Mean Precision@{k}: {np.mean(precision_scores):.3f}")
    print(f"Mean Hit@{k}: {np.mean(hit_scores):.3f}")
    print(f"Mean Recall@{k}: {np.mean(recall_scores):.3f}")
    print(f"Mean NDCG@{k}: {np.mean(ndcg_scores):.3f}")


def filter_by_intent_and_rules(q, df, intent_model, intent_vectorizer):
    intent, confidence = predict(q, intent_model, intent_vectorizer)

    print(f"Intent: {intent} ({confidence}%)")

    df_filtered = df.copy()

    from src.retrieval.query import (
        INTENT_TO_CATEGORY,
        detect_specific_category,
    )

    specific_categories = detect_specific_category(q)

    if specific_categories:
        categories = specific_categories
        print(f"Specific category override: {categories}")
    else:
        categories = INTENT_TO_CATEGORY.get(intent)

    if categories:
        df_filtered = df_filtered[
            df_filtered["category_final"].isin(categories)
        ]
        print(f"Filtered to categories: {categories} ({len(df_filtered)} POIs)")

    return df_filtered


def apply_boolean_filters(q, results):
    filters = parse_filters(q)

    if filters:
        print(f"Filters detected: {filters}")
        for col, val in filters.items():
            if col in results.columns:
                results = results[results[col] == val]

    return results


def calculate_metrics(results, df, expected, k):
    total_relevant = (df["category_final"] == expected).sum()

    p = precision_at_k(results, expected, k)
    h = hit_at_k(results, expected, k)
    r = recall_at_k(results, expected, total_relevant, k)
    n = ndcg_at_k(results, expected, k)

    return p, h, r, n


def evaluate_tfidf(
    evaluation_queries,
    df,
    vectorizer,
    tfidf_matrix,
    intent_model,
    intent_vectorizer,
    k: int = 5,
):
    precision_scores = []
    hit_scores = []
    recall_scores = []
    ndcg_scores = []

    print("\n\n=== TF-IDF METRICS ===")

    for item in evaluation_queries:
        q = item["query"]
        expected = item["expected_category"]

        results = query.search(
            q,
            df,
            vectorizer=vectorizer,
            tfidf_matrix=tfidf_matrix,
            intent_model=intent_model,
            intent_vectorizer=intent_vectorizer,
            top_k=k,
        )

        p, h, r, n = calculate_metrics(results, df, expected, k)

        precision_scores.append(p)
        hit_scores.append(h)
        recall_scores.append(r)
        ndcg_scores.append(n)

        print(f"\nQuery: '{q}'")
        print(f"Expected category: {expected}")
        print(f"Precision@{k}: {p:.3f}")
        print(f"Hit@{k}: {h}")
        print(f"Recall@{k}: {r:.3f}")
        print(f"NDCG@{k}: {n:.3f}")

        cols = ["name", "category_final", "similarity_score"]
        existing_cols = [c for c in cols if c in results.columns]
        print(results[existing_cols].to_string(index=False))

    print_metric_summary("TF-IDF", precision_scores, hit_scores, recall_scores, ndcg_scores, k)


def evaluate_bm25(
    evaluation_queries,
    df,
    bm25,
    intent_model,
    intent_vectorizer,
    k: int = 5,
):
    precision_scores = []
    hit_scores = []
    recall_scores = []
    ndcg_scores = []

    print("\n\n=== BM25 METRICS ===")

    for item in evaluation_queries:
        q = item["query"]
        expected = item["expected_category"]

        print(f"\nQuery: '{q}'")
        df_filtered = filter_by_intent_and_rules(
            q,
            df,
            intent_model,
            intent_vectorizer,
        )

        # PRVO boolean filter
        df_filtered = apply_boolean_filters(q, df_filtered)

        # ONDA BM25 search
        results = search_bm25(
            q,
            bm25,
            df,
            top_k=k,
            df_filtered=df_filtered,
        )

        results = results.head(k)

        p, h, r, n = calculate_metrics(results, df, expected, k)

        precision_scores.append(p)
        hit_scores.append(h)
        recall_scores.append(r)
        ndcg_scores.append(n)

        print(f"Expected category: {expected}")
        print(f"Precision@{k}: {p:.3f}")
        print(f"Hit@{k}: {h}")
        print(f"Recall@{k}: {r:.3f}")
        print(f"NDCG@{k}: {n:.3f}")

        cols = ["name", "category_final", "bm25_score"]
        existing_cols = [c for c in cols if c in results.columns]
        print(results[existing_cols].to_string(index=False))

    print_metric_summary("BM25", precision_scores, hit_scores, recall_scores, ndcg_scores, k)

def evaluate_tfidf_no_filter(
    evaluation_queries,
    df,
    vectorizer,
    tfidf_matrix,
    k: int = 5,
):
    precision_scores = []
    hit_scores = []
    recall_scores = []
    ndcg_scores = []

    print("\n\n=== TF-IDF METRICS (no intent filtering) ===")

    for item in evaluation_queries:
        q = item["query"]
        expected = item["expected_category"]

        print(f"\nQuery: '{q}'")

        from src.retrieval.tfidf import search_tfidf
        results = search_tfidf(q, vectorizer, tfidf_matrix, df, top_k=k)

        p, h, r, n = calculate_metrics(results, df, expected, k)

        precision_scores.append(p)
        hit_scores.append(h)
        recall_scores.append(r)
        ndcg_scores.append(n)

        print(f"Expected category: {expected}")
        print(f"Precision@{k}: {p:.3f}")
        print(f"Hit@{k}: {h}")

        cols = ["name", "category_final", "similarity_score"]
        existing_cols = [c for c in cols if c in results.columns]
        print(results[existing_cols].to_string(index=False))

    print_metric_summary("TF-IDF (no filter)", precision_scores, hit_scores, recall_scores, ndcg_scores, k)

def evaluate_bm25_no_filter(
    evaluation_queries,
    df,
    bm25,
    k: int = 5,
):
    precision_scores = []
    hit_scores = []
    recall_scores = []
    ndcg_scores = []

    print("\n\n=== BM25 METRICS (no intent filtering) ===")

    for item in evaluation_queries:
        q = item["query"]
        expected = item["expected_category"]

        print(f"\nQuery: '{q}'")

        # BM25 search na cijelom df — bez intent filteringa
        results = search_bm25(
            q,
            bm25,
            df,
            top_k=k,
        )

        results = results.head(k)

        p, h, r, n = calculate_metrics(results, df, expected, k)

        precision_scores.append(p)
        hit_scores.append(h)
        recall_scores.append(r)
        ndcg_scores.append(n)

        print(f"Expected category: {expected}")
        print(f"Precision@{k}: {p:.3f}")
        print(f"Hit@{k}: {h}")

        cols = ["name", "category_final", "bm25_score"]
        existing_cols = [c for c in cols if c in results.columns]
        print(results[existing_cols].to_string(index=False))

    print_metric_summary("BM25 (no filter)", precision_scores, hit_scores, recall_scores, ndcg_scores, k)

def evaluate_embeddings(
    evaluation_queries,
    df,
    embedding_model,
    poi_embeddings,
    intent_model,
    intent_vectorizer,
    k: int = 5,
):
    precision_scores = []
    hit_scores = []
    recall_scores = []
    ndcg_scores = []

    print("\n\n=== EMBEDDINGS METRICS ===")

    for item in evaluation_queries:
        q = item["query"]
        expected = item["expected_category"]

        print(f"\nQuery: '{q}'")
        df_filtered = filter_by_intent_and_rules(q, df, intent_model, intent_vectorizer)

        results = search_embeddings(
            q,
            embedding_model,
            poi_embeddings,
            df,
            top_k=k * 5,
            df_filtered=df_filtered,
        )

        results = apply_boolean_filters(q, results)
        results = results.head(k)

        p, h, r, n = calculate_metrics(results, df, expected, k)

        precision_scores.append(p)
        hit_scores.append(h)
        recall_scores.append(r)
        ndcg_scores.append(n)

        print(f"Expected category: {expected}")
        print(f"Precision@{k}: {p:.3f}")
        print(f"Hit@{k}: {h}")
        print(f"Recall@{k}: {r:.3f}")
        print(f"NDCG@{k}: {n:.3f}")

        cols = ["name", "category_final", "embedding_score"]
        existing_cols = [c for c in cols if c in results.columns]
        print(results[existing_cols].to_string(index=False))

    print_metric_summary("EMBEDDINGS", precision_scores, hit_scores, recall_scores, ndcg_scores, k)


def main():
    intent_model, intent_vectorizer = load_model("models/intent_classifier.pkl")

    df = load_csv("data/processed/cleaned_pois.csv")
    df = pipeline.run(df)

    vectorizer, tfidf_matrix = tfidf.run(df)
    bm25 = build_bm25(df)

    print("\n=== TOKENIZATION DEBUG ===")

    tfidf.debug_tokenization(
        "where can i get espresso",
        df,
        row_idx=0,
    )

    tfidf.debug_tokenization(
        "accessible shop near me",
        df,
        row_idx=10,
    )

    tfidf.debug_tokenization(
        "need a cashpoint",
        df,
        row_idx=20,
    )

    tfidf.debug_tokenization(
        "place to lock my bike",
        df,
        row_idx=30,
    )

    embedding_model = load_embedding_model()
    poi_embeddings = get_or_build_embeddings(
        df,
        embedding_model,
        col="poi_text_lemma",
        path="models/poi_embeddings.npy",
    )

    test_queries = [
        "coffee near burnside",
        "mexican restaurant",
        "wheelchair accessible cafe",
        "24/7 pharmacy",
        "italian restaurant takeaway",
    ]

    print("\n\n=== TF-IDF SEARCH RESULTS ===")

    for q in test_queries:
        print("\n" + "=" * 50)

        results = query.search(
            q,
            df,
            vectorizer=vectorizer,
            tfidf_matrix=tfidf_matrix,
            intent_model=intent_model,
            intent_vectorizer=intent_vectorizer,
            top_k=5,
        )

        cols = ["name", "category_final", "addr:street", "similarity_score"]
        existing_cols = [c for c in cols if c in results.columns]
        print(results[existing_cols].to_string(index=False))

    print("\n\n=== N-GRAM COMPARISON ===")
    compare_ngrams(df)

    comparison_queries = [
        "coffee near burnside",
        "mexican restaurant",
        "wheelchair accessible cafe",
    ]

    print("\n\n=== BM25 vs TF-IDF COMPARISON ===")

    for q in comparison_queries:
        compare_bm25_tfidf(
            q,
            bm25,
            vectorizer,
            tfidf_matrix,
            df,
            intent_model=intent_model,
            intent_vectorizer=intent_vectorizer,
        )

    evaluation_queries = [
    {"query": "where can i get espresso", "expected_category": "cafe"},
    {"query": "tacos near me", "expected_category": "restaurant"},
    {"query": "find me a dentist", "expected_category": "dentist"},
    {"query": "need a cashpoint", "expected_category": "atm"},
    {"query": "emergency room nearby", "expected_category": "hospital"},
    {"query": "where to park my car", "expected_category": "parking"},
    {"query": "place to lock my bike", "expected_category": "bicycle_parking"},
    {"query": "gluten free restaurant", "expected_category": "restaurant"},
    {"query": "place for a haircut", "expected_category": "hairdresser"},
    {"query": "grab a coffee and work", "expected_category": "cafe"},
]

    print("\n\n=== BM25 TUNING ===")
    tune_bm25(df, evaluation_queries, intent_model, intent_vectorizer)

    for k in [5, 20, 50]:
        print(f"\n{'='*60}")
        print(f"EVALUACIJA SA k={k}")
        print(f"{'='*60}")
        
        evaluate_tfidf(
            evaluation_queries, df, vectorizer, tfidf_matrix,
            intent_model, intent_vectorizer, k=k,
        )
        evaluate_bm25(
            evaluation_queries, df, bm25,
            intent_model, intent_vectorizer, k=k,
        )
        evaluate_embeddings(
            evaluation_queries, df, embedding_model, poi_embeddings,
            intent_model, intent_vectorizer, k=k,
        )
        evaluate_tfidf_no_filter(evaluation_queries, df, vectorizer, tfidf_matrix, k=k)
        evaluate_bm25_no_filter(evaluation_queries, df, bm25, k=k)



if __name__ == "__main__":
    main()