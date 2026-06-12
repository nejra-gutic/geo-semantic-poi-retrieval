"""
app.py
------
Streamlit UI for Geo-Semantic POI Search.

Run:
    streamlit run app.py
"""

import streamlit as st
import folium
from streamlit_folium import st_folium
import pandas as pd

st.set_page_config(
    page_title="Portland POI Search",
    page_icon="📍",
    layout="wide",
)

PORTLAND_CENTER = [45.5051, -122.6750]

# === LOAD MODELS (cached) ===
@st.cache_resource
def load_models():
    from src.utils.io import load_csv
    from src.retrieval import pipeline, tfidf
    from src.retrieval.intent_classifier import load_model
    from src.retrieval.bm25 import build_bm25
    from src.retrieval.embeddings import load_embedding_model, get_or_build_embeddings

    df = load_csv("data/processed/cleaned_pois.csv")
    df["poi_id"] = df.index
    original_poi_ids = df["poi_id"].copy()
    df = pipeline.run(df)
    df["poi_id"] = original_poi_ids.values

    intent_model, intent_vectorizer = load_model("models/intent_classifier.pkl")
    vectorizer, tfidf_matrix = tfidf.run(df)
    bm25 = build_bm25(df)
    embedding_model = load_embedding_model()
    poi_embeddings = get_or_build_embeddings(
        df, embedding_model, col="poi_text", path="models/poi_embeddings.npy"
    )

    return df, intent_model, intent_vectorizer, vectorizer, tfidf_matrix, bm25, embedding_model, poi_embeddings


def run_search(query, df, intent_model, intent_vectorizer, vectorizer, tfidf_matrix, bm25, embedding_model, poi_embeddings, method):
    from sklearn.preprocessing import MinMaxScaler
    from src.retrieval.bm25 import search_bm25
    from src.retrieval.embeddings import search_embeddings
    from src.retrieval.geo import combine_with_geo, PORTLAND_CENTER as PC
    from src.retrieval.query import search, expand_query_synonyms, detect_specific_category, INTENT_TO_CATEGORY, parse_filters
    from src.retrieval.intent_classifier import predict
    from src.retrieval.normalize import normalize
    from src.preprocessing.normalize import normalize_text as preprocess_text
    import numpy as np
    from sklearn.metrics.pairwise import cosine_similarity

    if method == "TF-IDF":
        results = search(
            query, df,
            vectorizer=vectorizer,
            tfidf_matrix=tfidf_matrix,
            intent_model=intent_model,
            intent_vectorizer=intent_vectorizer,
            top_k=10,
        )
        return results

    elif method == "Embeddings":
        results = search_embeddings(query, embedding_model, poi_embeddings, df, top_k=10)
        return results

    elif method == "Hybrid":
        bm25_results = search_bm25(query, bm25, df, top_k=200)
        emb_results = search_embeddings(query, embedding_model, poi_embeddings, df, top_k=200)

        bm25_results = bm25_results.copy()
        emb_results = emb_results.copy()
        bm25_results["poi_id"] = bm25_results.index
        emb_results["poi_id"] = emb_results.index

        score_col = [c for c in bm25_results.columns if "score" in c.lower()]
        if score_col:
            bm25_results = bm25_results.rename(columns={score_col[0]: "bm25_score"})
        else:
            bm25_results["bm25_score"] = 1.0

        hybrid = pd.merge(
            bm25_results[["poi_id", "bm25_score"]],
            emb_results[["poi_id", "embedding_score"]],
            on="poi_id", how="outer"
        ).fillna(0)

        if hybrid["bm25_score"].max() > hybrid["bm25_score"].min():
            hybrid["bm25_norm"] = MinMaxScaler().fit_transform(hybrid[["bm25_score"]])
        else:
            hybrid["bm25_norm"] = 0

        if hybrid["embedding_score"].max() > hybrid["embedding_score"].min():
            hybrid["emb_norm"] = MinMaxScaler().fit_transform(hybrid[["embedding_score"]])
        else:
            hybrid["emb_norm"] = 0

        hybrid["hybrid_score"] = 0.2 * hybrid["bm25_norm"] + 0.8 * hybrid["emb_norm"]
        top = hybrid.sort_values("hybrid_score", ascending=False).head(10)

        results = df.loc[df.index.isin(top["poi_id"])].copy()
        results = results.merge(top[["poi_id", "hybrid_score"]], left_index=True, right_on="poi_id", how="left")

        near_me = any(w in query.lower() for w in ["near me", "nearby", "close by"])
        if near_me and "latitude" in results.columns and "longitude" in results.columns:
            results = combine_with_geo(results, PC[0], PC[1], score_col="hybrid_score")

        return results

    return pd.DataFrame()


# === UI ===
st.title("📍 Portland POI Search")
st.markdown("Search for places in Portland using natural language.")

with st.spinner("Loading models... (first run takes ~30 seconds)"):
    df, intent_model, intent_vectorizer, vectorizer, tfidf_matrix, bm25, embedding_model, poi_embeddings = load_models()

col1, col2 = st.columns([3, 1])
with col1:
    query = st.text_input(
        "What are you looking for?",
        placeholder="E.g. coffee near me, wheelchair accessible cafe, vet for my dog",
    )
with col2:
    method = st.selectbox("Retrieval method", ["Hybrid", "Embeddings", "TF-IDF"])

if query:
    with st.spinner("Searching..."):
        results = run_search(
            query, df, intent_model, intent_vectorizer,
            vectorizer, tfidf_matrix, bm25, embedding_model, poi_embeddings,
            method
        )

    if results is None or results.empty:
        st.warning("No results found. Try a different query.")
    else:
        st.success(f"Found {len(results)} results")

        # Map
        valid_results = results[results["latitude"].notna() & results["longitude"].notna()]

        if not valid_results.empty:
            center_lat = valid_results["latitude"].mean()
            center_lon = valid_results["longitude"].mean()
        else:
            center_lat, center_lon = PORTLAND_CENTER

        m = folium.Map(location=[center_lat, center_lon], zoom_start=13, tiles="OpenStreetMap")

        for _, row in valid_results.iterrows():
            name = row.get("name", "Unknown")
            category = row.get("category_final", "")
            popup_text = f"<b>{name}</b><br>{category}"
            folium.Marker(
                [row["latitude"], row["longitude"]],
                popup=folium.Popup(popup_text, max_width=200),
                tooltip=name,
                icon=folium.Icon(color="blue", icon="info-sign"),
            ).add_to(m)

        st_folium(m, width=900, height=500, returned_objects=[])

        # Results table
        st.subheader("Results")
        score_col = next((c for c in ["combined_score", "hybrid_score", "embedding_score", "similarity_score"] if c in results.columns), None)
        display_cols = ["name", "category_final", "addr:street"]
        if score_col:
            display_cols.append(score_col)
            results = results.sort_values(score_col, ascending=False)
        existing_cols = [c for c in display_cols if c in results.columns]
        st.dataframe(results[existing_cols].reset_index(drop=True), use_container_width=True)

else:
    # Default map
    m = folium.Map(location=PORTLAND_CENTER, zoom_start=12, tiles="OpenStreetMap")
    st_folium(m, width=900, height=500)