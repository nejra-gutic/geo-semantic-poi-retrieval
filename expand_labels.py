"""
expand_labels.py
----------------
Pooling pristup za proširenje relevance_labels_core.csv.
Za svaki query skupi top-K rezultate od svih metoda,
pronađi POI-e koji nisu u labeled setu, ispiši ih za ručnu provjeru.
"""

import pandas as pd
import numpy as np
from src.utils.io import load_csv
from src.retrieval import pipeline, tfidf
from src.retrieval.intent_classifier import load_model
from src.retrieval.bm25 import build_bm25, search_bm25
from src.retrieval.embeddings import load_embedding_model, get_or_build_embeddings, search_embeddings
from src.retrieval.query import search, INTENT_TO_CATEGORY, detect_specific_category

POOL_K = 20  # koliko rezultata uzeti od svake metode

df = load_csv("data/processed/cleaned_pois.csv")
df["poi_id"] = df.index
original_poi_ids = df["poi_id"].copy()
df = pipeline.run(df)
df["poi_id"] = original_poi_ids.values

intent_model, intent_vectorizer = load_model("models/intent_classifier.pkl")
vectorizer, tfidf_matrix = tfidf.run(df)
bm25 = build_bm25(df)
embedding_model = load_embedding_model()
poi_embeddings = get_or_build_embeddings(df, embedding_model, col="poi_text", path="models/poi_embeddings.npy")

labels_df = pd.read_csv("data/relevance_labels_core.csv")

for _, row in labels_df.iterrows():
    query = row["query"].strip()
    current_ids = set(int(x.strip()) for x in str(row["relevant_poi_ids"]).split(",") if x.strip())

    pool = set()

    # TF-IDF
    r1 = search(query, df, vectorizer, tfidf_matrix, intent_model, intent_vectorizer, top_k=POOL_K)
    if not r1.empty:
        pool.update(r1.index.tolist())

    # BM25
    r2 = search_bm25(query, bm25, df, top_k=POOL_K)
    if not r2.empty:
        pool.update(r2.index.tolist())

    # Embeddings
    r3 = search_embeddings(query, embedding_model, poi_embeddings, df, top_k=POOL_K)
    if not r3.empty:
        pool.update(r3.index.tolist())

    # Novi kandidati
    new_candidates = pool - current_ids

    if new_candidates:
        print(f"\n{'='*60}")
        print(f"Query: '{query}'")
        print(f"Trenutno labeled: {len(current_ids)} POI-a")
        print(f"Novi kandidati za provjeru ({len(new_candidates)}):")
        cands = df[df.index.isin(new_candidates)][["name", "category_final"]].copy()
        cands["poi_id"] = cands.index
        print(cands[["poi_id", "name", "category_final"]].to_string(index=False))