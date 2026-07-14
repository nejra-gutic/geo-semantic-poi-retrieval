"""
tune_hybrid_weights.py
----------------------
Grid search za hybrid BM25 + Embeddings težine.
Evaluira različite kombinacije w_bm25 i w_emb na relevance_labels_validation.csv.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from src.utils.io import load_csv
from src.retrieval import pipeline, tfidf
from src.retrieval.intent_classifier import load_model
from src.retrieval.bm25 import build_bm25, search_bm25
from src.retrieval.embeddings import load_embedding_model, get_or_build_embeddings, search_embeddings

RELEVANCE_PATH = "data/relevance_labels_validation_combined.csv"

# === UČITAJ PODATKE ===
df = load_csv("data/processed/cleaned_pois.csv")
df["poi_id"] = df.index
original_poi_ids = df["poi_id"].copy()
df = pipeline.run(df)
df["poi_id"] = original_poi_ids.values

intent_model, intent_vectorizer = load_model("models/intent_classifier.pkl")
bm25 = build_bm25(df)
embedding_model = load_embedding_model()
poi_embeddings = get_or_build_embeddings(df, embedding_model, col="poi_text", path="models/poi_embeddings.npy")

labels_df = pd.read_csv(RELEVANCE_PATH)
queries = labels_df["query"].str.strip().tolist()
relevance_labels = {}
for _, row in labels_df.iterrows():
    q = row["query"].strip()
    ids = set(int(x.strip()) for x in str(row["relevant_poi_ids"]).split(",") if x.strip())
    relevance_labels[q] = ids


def ndcg_at_k(retrieved_ids, relevant_ids, k):
    top_k = retrieved_ids[:k]
    relevance = [1 if pid in relevant_ids else 0 for pid in top_k]

    dcg = sum(rel / np.log2(i + 2) for i, rel in enumerate(relevance))

    # IDCG mora odražavati broj STVARNO relevantnih POI-ja (do k), a ne samo
    # onih koji su slučajno pronađeni -- inače query s jednim pogotkom na
    # prvom mjestu dobija NDCG 1.0 čak i kad postoji mnogo relevantnih POI-ja.
    ideal_hits = min(len(relevant_ids), k)
    idcg = sum(1 / np.log2(i + 2) for i in range(ideal_hits))

    return dcg / idcg if idcg > 0 else 0.0


def evaluate_weights(w_bm25, w_emb):
    ndcg_5_scores = []

    for query in queries:
        relevant_ids = relevance_labels.get(query, set())
        if not relevant_ids:
            continue

        # apply_temporal=False: tuning mjeri isključivo BM25+embeddings
        # kombinaciju, bez vremenski promjenjivog open_now boosta.
        bm25_results = search_bm25(
            query, bm25, df,
            top_k=200,
            intent_model=intent_model,
            intent_vectorizer=intent_vectorizer,
            apply_temporal=False,
        )
        emb_results = search_embeddings(
            query, embedding_model, poi_embeddings, df,
            top_k=200,
            intent_model=intent_model,
            intent_vectorizer=intent_vectorizer,
            apply_temporal=False,
        )

        # search_bm25()/search_embeddings() već vraćaju ispravan poi_id --
        # NE prepisivati sa .index, jer kod embeddingsa index rezultata ne
        # mora odgovarati originalnom poi_id-u (spajalo bi na pogrešna mjesta).
        bm25_results = bm25_results.copy()
        emb_results = emb_results.copy()

        score_col = [c for c in bm25_results.columns if "score" in c.lower()]
        if score_col:
            bm25_results = bm25_results.rename(columns={score_col[0]: "bm25_score"})
        else:
            bm25_results["bm25_score"] = 1.0

        bm25_scores = bm25_results[["poi_id", "bm25_score"]]
        emb_scores = emb_results[["poi_id", "embedding_score"]]

        hybrid = pd.merge(bm25_scores, emb_scores, on="poi_id", how="outer").fillna(0)

        if hybrid["bm25_score"].max() > hybrid["bm25_score"].min():
            hybrid["bm25_norm"] = MinMaxScaler().fit_transform(hybrid[["bm25_score"]])
        else:
            hybrid["bm25_norm"] = 0

        if hybrid["embedding_score"].max() > hybrid["embedding_score"].min():
            hybrid["emb_norm"] = MinMaxScaler().fit_transform(hybrid[["embedding_score"]])
        else:
            hybrid["emb_norm"] = 0

        hybrid["hybrid_score"] = w_bm25 * hybrid["bm25_norm"] + w_emb * hybrid["emb_norm"]
        results = hybrid.sort_values("hybrid_score", ascending=False).head(50)
        retrieved_ids = results["poi_id"].tolist()

        ndcg_5_scores.append(ndcg_at_k(retrieved_ids, relevant_ids, 5))

    return np.mean(ndcg_5_scores)


# === GRID SEARCH ===
weights = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
best_score = -1
best_params = {}
results_table = []

print(f"{'w_bm25':>8} {'w_emb':>8} {'NDCG@5':>10}")
print("-" * 30)

for w_bm25 in weights:
    w_emb = round(1.0 - w_bm25, 1)
    score = evaluate_weights(w_bm25, w_emb)
    results_table.append({"w_bm25": w_bm25, "w_emb": w_emb, "ndcg_5": score})
    print(f"{w_bm25:>8.1f} {w_emb:>8.1f} {score:>10.4f}")

    if score > best_score:
        best_score = score
        best_params = {"w_bm25": w_bm25, "w_emb": w_emb}

print(f"\nBest: w_bm25={best_params['w_bm25']}, w_emb={best_params['w_emb']} → NDCG@5: {best_score:.4f}")