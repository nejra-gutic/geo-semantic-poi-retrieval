import pandas as pd

from src.utils.io import load_csv
from src.retrieval import pipeline
from src.retrieval.intent_classifier import load_model
from src.retrieval.bm25 import build_bm25, search_bm25
from src.retrieval.embeddings import load_embedding_model, get_or_build_embeddings, search_embeddings
from src.retrieval.query import parse_filters
from src.retrieval.common import (
    apply_intent_filter,
    apply_boolean_filters,
    apply_temporal_filter,
    apply_geo_reranking,
)

df = load_csv("data/processed/cleaned_pois.csv")
df = pipeline.run(df)

intent_model, intent_vectorizer = load_model("models/intent_classifier.pkl")
bm25 = build_bm25(df)

embedding_model = load_embedding_model()
embeddings = get_or_build_embeddings(df, embedding_model, col="poi_text", path="models/poi_embeddings.npy")

queries = [
    "coffee open now near me",
    "pharmacy open late",
    "wheelchair accessible cafe",
    "restaurant open tonight",
    "bank near me",
]

for q in queries:
    print("\n" + "=" * 80)
    print("QUERY:", q)

    filters = parse_filters(q)
    print("FILTERS:", filters)

    df_filtered = apply_intent_filter(q, df, intent_model, intent_vectorizer)

    bm25_results = search_bm25(q, bm25, df, top_k=10, df_filtered=df_filtered)
    bm25_results = apply_boolean_filters(bm25_results, filters)
    bm25_results = apply_temporal_filter(bm25_results, q, filters, score_col="bm25_score")
    bm25_results = apply_geo_reranking(bm25_results, q, None, None, score_col="bm25_score")

    print("\nBM25:")
    print(bm25_results[["name", "category_final", "opening_hours", "is_open_now", "bm25_score"]].head(5).to_string())

    emb_results = search_embeddings(q, embedding_model, embeddings, df, top_k=10, df_filtered=df_filtered)
    emb_results = apply_boolean_filters(emb_results, filters)
    emb_results = apply_temporal_filter(emb_results, q, filters, score_col="embedding_score")
    emb_results = apply_geo_reranking(emb_results, q, None, None, score_col="embedding_score")

    print("\nEmbeddings:")
    print(emb_results[["name", "category_final", "opening_hours", "is_open_now", "embedding_score"]].head(5).to_string())