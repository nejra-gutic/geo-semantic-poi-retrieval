"""
eval.py
-------
Standalone evaluation script for TF-IDF, BM25, Embeddings, Hybrid, and RRF retrieval.

Uses per-query relevance labels from:
    data/relevance_labels.csv

Run:
    python3 eval.py
"""

import numpy as np
import pandas as pd

from sklearn.preprocessing import MinMaxScaler

from src.utils.io import load_csv
from src.retrieval import pipeline, tfidf
from src.retrieval.intent_classifier import load_model, predict
from src.retrieval.bm25 import build_bm25, search_bm25
from src.retrieval.query import (
    search,
    INTENT_TO_CATEGORY,
    detect_specific_category,
)
from src.retrieval.embeddings import (
    load_embedding_model,
    get_or_build_embeddings,
    search_embeddings,
)


DATA_PATH = "data/processed/cleaned_pois.csv"
RELEVANCE_PATH = "data/relevance_labels_test.csv"
INTENT_MODEL_PATH = "models/intent_classifier.pkl"
EMBEDDINGS_PATH = "models/poi_embeddings.npy"

K_VALUES = [5, 10, 20, 50]


def load_relevance_labels(path: str) -> dict:
    df = pd.read_csv(path)

    labels = {}
    for _, row in df.iterrows():
        poi_ids = set(
            int(x.strip())
            for x in str(row["relevant_poi_ids"]).split(",")
            if x.strip()
        )
        labels[row["query"].strip()] = poi_ids

    return labels


def precision_at_k(retrieved_ids: list, relevant_ids: set, k: int) -> float:
    top_k = retrieved_ids[:k]
    hits = sum(1 for pid in top_k if pid in relevant_ids)
    return hits / k if k > 0 else 0.0


def recall_at_k(retrieved_ids: list, relevant_ids: set, k: int) -> float:
    if not relevant_ids:
        return 0.0

    top_k = retrieved_ids[:k]
    hits = sum(1 for pid in top_k if pid in relevant_ids)
    return hits / len(relevant_ids)


def hit_at_k(retrieved_ids: list, relevant_ids: set, k: int) -> int:
    top_k = retrieved_ids[:k]
    return 1 if any(pid in relevant_ids for pid in top_k) else 0


def ndcg_at_k(retrieved_ids: list, relevant_ids: set, k: int) -> float:
    top_k = retrieved_ids[:k]
    relevance = [1 if pid in relevant_ids else 0 for pid in top_k]

    if sum(relevance) == 0:
        return 0.0

    dcg = sum(rel / np.log2(i + 2) for i, rel in enumerate(relevance))

    ideal_relevance = sorted(relevance, reverse=True)
    idcg = sum(rel / np.log2(i + 2) for i, rel in enumerate(ideal_relevance))

    return dcg / idcg if idcg > 0 else 0.0


def compute_metrics(retrieved_ids: list, relevant_ids: set, k: int) -> dict:
    return {
        "precision": precision_at_k(retrieved_ids, relevant_ids, k),
        "recall": recall_at_k(retrieved_ids, relevant_ids, k),
        "hit": hit_at_k(retrieved_ids, relevant_ids, k),
        "ndcg": ndcg_at_k(retrieved_ids, relevant_ids, k),
    }


def ensure_poi_id(results: pd.DataFrame) -> pd.DataFrame:
    results = results.copy()

    if "poi_id" not in results.columns:
        if "index" in results.columns:
            results["poi_id"] = results["index"]
        else:
            results["poi_id"] = results.index

    results["poi_id"] = results["poi_id"].astype(int)
    return results


def filter_by_intent(
    query: str,
    df: pd.DataFrame,
    intent_model,
    intent_vectorizer,
) -> pd.DataFrame:
    intent, confidence = predict(query, intent_model, intent_vectorizer)

    specific = detect_specific_category(query)
    categories = specific or INTENT_TO_CATEGORY.get(intent)

    if categories:
        return df[df["category_final"].isin(categories)]

    return df


def evaluate(
    name: str,
    queries: list,
    relevance_labels: dict,
    get_results_fn,
    k_values: list,
):
    print(f"\n{'=' * 60}")
    print(f"  {name}")
    print(f"{'=' * 60}")

    all_metrics = {
        k: {"precision": [], "recall": [], "hit": [], "ndcg": []}
        for k in k_values
    }

    for query in queries:
        relevant_ids = relevance_labels.get(query, set())

        if not relevant_ids:
            print(f"  [SKIP] No relevance labels for: '{query}'")
            continue

        results = get_results_fn(query)

        if results.empty:
            retrieved_ids = []
        else:
            results = ensure_poi_id(results)
            retrieved_ids = results["poi_id"].tolist()

        print(f"\n  Query: '{query}'")
        print(f"  Relevant POIs: {len(relevant_ids)} | Retrieved: {len(retrieved_ids)}")

        for k in k_values:
            m = compute_metrics(retrieved_ids, relevant_ids, k)

            for metric, value in m.items():
                all_metrics[k][metric].append(value)

            print(
                f"    @{k:2d}  "
                f"P={m['precision']:.3f}  "
                f"R={m['recall']:.3f}  "
                f"Hit={m['hit']}  "
                f"NDCG={m['ndcg']:.3f}"
            )

    print(f"\n  {'─' * 50}")
    print("  AVERAGES")

    for k in k_values:
        m = all_metrics[k]

        if len(m["precision"]) == 0:
            print(f"    @{k:2d}  No evaluated queries")
            continue

        print(
            f"    @{k:2d}  "
            f"P={np.mean(m['precision']):.3f}  "
            f"R={np.mean(m['recall']):.3f}  "
            f"Hit={np.mean(m['hit']):.3f}  "
            f"NDCG={np.mean(m['ndcg']):.3f}"
        )


def main():
    print("[eval] Loading data...")

    df = load_csv(DATA_PATH)
    df = df.rename(columns={"Unnamed: 0": "poi_id"})

    if "poi_id" not in df.columns:
        df["poi_id"] = df.index

    original_poi_ids = df["poi_id"].copy()

    df = pipeline.run(df)

    df["poi_id"] = original_poi_ids.values

    print("[eval] Loading models...")

    intent_model, intent_vectorizer = load_model(INTENT_MODEL_PATH)
    vectorizer, tfidf_matrix = tfidf.run(df)
    bm25 = build_bm25(df)

    print("[eval] Loading embeddings...")

    embedding_model = load_embedding_model()
    poi_embeddings = get_or_build_embeddings(
        df,
        embedding_model,
        col="poi_text",
        path=EMBEDDINGS_PATH,
    )

    print("[eval] Loading relevance labels...")

    relevance_labels = load_relevance_labels(RELEVANCE_PATH)
    queries = list(relevance_labels.keys())

    print(f"[eval] {len(queries)} queries loaded")

    def tfidf_with_filter(query):
        results = search(
            query,
            df,
            vectorizer=vectorizer,
            tfidf_matrix=tfidf_matrix,
            intent_model=intent_model,
            intent_vectorizer=intent_vectorizer,
            top_k=max(K_VALUES),
        )
        return ensure_poi_id(results)

    def tfidf_no_filter(query):
        from src.retrieval.tfidf import search_tfidf

        results = search_tfidf(
            query,
            vectorizer,
            tfidf_matrix,
            df,
            top_k=max(K_VALUES),
        )
        return ensure_poi_id(results)

    def bm25_with_filter(query):
        df_filtered = filter_by_intent(
            query,
            df,
            intent_model,
            intent_vectorizer,
        )

        results = search_bm25(
            query,
            bm25,
            df,
            top_k=max(K_VALUES),
            df_filtered=df_filtered,
        )
        return ensure_poi_id(results)

    def bm25_no_filter(query):
        results = search_bm25(
            query,
            bm25,
            df,
            top_k=max(K_VALUES),
        )
        return ensure_poi_id(results)

    def embeddings_no_filter(query):
        results = search_embeddings(
            query,
            embedding_model,
            poi_embeddings,
            df,
            top_k=max(K_VALUES),
        )
        return ensure_poi_id(results)

    def hybrid_bm25_embeddings(query):
        bm25_results = search_bm25(
            query,
            bm25,
            df,
            top_k=200,
        )

        emb_results = search_embeddings(
            query,
            embedding_model,
            poi_embeddings,
            df,
            top_k=200,
        )

        bm25_results = ensure_poi_id(bm25_results)
        emb_results = ensure_poi_id(emb_results)

        if "bm25_score" not in bm25_results.columns:
            score_cols = [c for c in bm25_results.columns if "score" in c.lower()]
            if score_cols:
                bm25_results = bm25_results.rename(columns={score_cols[0]: "bm25_score"})
            else:
                bm25_results["bm25_score"] = 1.0

        bm25_scores = bm25_results[["poi_id", "bm25_score"]].copy()
        emb_scores = emb_results[["poi_id", "embedding_score"]].copy()

        hybrid = pd.merge(
            bm25_scores,
            emb_scores,
            on="poi_id",
            how="outer",
        ).fillna(0)

        if hybrid["bm25_score"].max() > hybrid["bm25_score"].min():
            hybrid["bm25_norm"] = MinMaxScaler().fit_transform(
                hybrid[["bm25_score"]]
            )
        else:
            hybrid["bm25_norm"] = 0

        if hybrid["embedding_score"].max() > hybrid["embedding_score"].min():
            hybrid["emb_norm"] = MinMaxScaler().fit_transform(
                hybrid[["embedding_score"]]
            )
        else:
            hybrid["emb_norm"] = 0

        hybrid["hybrid_score"] = (
            0.2 * hybrid["bm25_norm"]
            + 0.8 * hybrid["emb_norm"]
        )

        results = hybrid.sort_values(
            "hybrid_score",
            ascending=False,
        ).head(max(K_VALUES))

        print(f"[hybrid] Query: '{query}' → {len(results)} results")

        return results

    def rrf_bm25_embeddings(query):
        bm25_results = search_bm25(
            query,
            bm25,
            df,
            top_k=200,
        )

        emb_results = search_embeddings(
            query,
            embedding_model,
            poi_embeddings,
            df,
            top_k=200,
        )

        bm25_results = ensure_poi_id(bm25_results)
        emb_results = ensure_poi_id(emb_results)

        rrf_k = 60
        scores = {}

        for rank, poi_id in enumerate(
            bm25_results["poi_id"].tolist(),
            start=1,
        ):
            scores[poi_id] = scores.get(poi_id, 0) + 1 / (rrf_k + rank)

        for rank, poi_id in enumerate(
            emb_results["poi_id"].tolist(),
            start=1,
        ):
            scores[poi_id] = scores.get(poi_id, 0) + 1 / (rrf_k + rank)

        rrf_df = pd.DataFrame({
            "poi_id": list(scores.keys()),
            "rrf_score": list(scores.values()),
        })

        results = rrf_df.sort_values(
            "rrf_score",
            ascending=False,
        ).head(max(K_VALUES))

        print(f"[rrf] Query: '{query}' → {len(results)} results")

        return results
    
    def hybrid_geo(query):
        bm25_results = search_bm25(query, bm25, df, top_k=200)
        emb_results = search_embeddings(query, embedding_model, poi_embeddings, df, top_k=200)

        bm25_results = ensure_poi_id(bm25_results)
        emb_results = ensure_poi_id(emb_results)

        if "bm25_score" not in bm25_results.columns:
            score_cols = [c for c in bm25_results.columns if "score" in c.lower()]
            if score_cols:
                bm25_results = bm25_results.rename(columns={score_cols[0]: "bm25_score"})
            else:
                bm25_results["bm25_score"] = 1.0

        bm25_scores = bm25_results[["poi_id", "bm25_score"]].copy()
        emb_scores = emb_results[["poi_id", "embedding_score"]].copy()

        hybrid = pd.merge(bm25_scores, emb_scores, on="poi_id", how="outer").fillna(0)

        if hybrid["bm25_score"].max() > hybrid["bm25_score"].min():
            hybrid["bm25_norm"] = MinMaxScaler().fit_transform(hybrid[["bm25_score"]])
        else:
            hybrid["bm25_norm"] = 0

        if hybrid["embedding_score"].max() > hybrid["embedding_score"].min():
            hybrid["emb_norm"] = MinMaxScaler().fit_transform(hybrid[["embedding_score"]])
        else:
            hybrid["emb_norm"] = 0

        hybrid["hybrid_score"] = 0.1 * hybrid["bm25_norm"] + 0.9 * hybrid["emb_norm"]

        results = hybrid.sort_values("hybrid_score", ascending=False).head(max(K_VALUES))
        results = ensure_poi_id(results)

        # Merge latitude/longitude za geo
        results = results.merge(df[["poi_id", "latitude", "longitude"]], on="poi_id", how="left")

        from src.retrieval.geo import combine_with_geo, PORTLAND_CENTER
        results = combine_with_geo(
            results,
            PORTLAND_CENTER[0],
            PORTLAND_CENTER[1],
            score_col="hybrid_score",
        )

        print(f"[hybrid_geo] Query: '{query}' → {len(results)} results")
        return results

    evaluate(
        "TF-IDF (with intent filter)",
        queries,
        relevance_labels,
        tfidf_with_filter,
        K_VALUES,
    )

    evaluate(
        "TF-IDF (no filter)",
        queries,
        relevance_labels,
        tfidf_no_filter,
        K_VALUES,
    )

    evaluate(
        "BM25 (with intent filter)",
        queries,
        relevance_labels,
        bm25_with_filter,
        K_VALUES,
    )

    evaluate(
        "BM25 (no filter)",
        queries,
        relevance_labels,
        bm25_no_filter,
        K_VALUES,
    )

    evaluate(
        "Embeddings (no filter)",
        queries,
        relevance_labels,
        embeddings_no_filter,
        K_VALUES,
    )

    evaluate(
        "Hybrid BM25 + Embeddings",
        queries,
        relevance_labels,
        hybrid_bm25_embeddings,
        K_VALUES,
    )

    evaluate(
        "RRF BM25 + Embeddings",
        queries,
        relevance_labels,
        rrf_bm25_embeddings,
        K_VALUES,
    )

    evaluate(
        "Hybrid BM25 + Embeddings + Geo",
        queries,
        relevance_labels,
        hybrid_geo,
        K_VALUES,
    )


if __name__ == "__main__":
    main()