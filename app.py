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

# Custom CSS
st.markdown("""
<style>
    /* Import font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* Hide streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    /* Main background */
    .stApp {
        background-color: #0f1117;
    }

    /* Header */
    .header-container {
        padding: 2rem 0 1rem 0;
        border-bottom: 1px solid #1e2530;
        margin-bottom: 1.5rem;
    }

    .header-title {
        font-size: 1.8rem;
        font-weight: 700;
        color: #ffffff;
        margin: 0;
        letter-spacing: -0.02em;
    }

    .header-subtitle {
        font-size: 0.9rem;
        color: #6b7280;
        margin: 0.25rem 0 0 0;
    }

    .accent {
        color: #3b82f6;
    }

    /* Search input */
    .stTextInput > div > div > input {
        background-color: #1a1f2e !important;
        border: 1.5px solid #2d3748 !important;
        border-radius: 10px !important;
        color: #ffffff !important;
        font-size: 1rem !important;
        padding: 0.75rem 1rem !important;
        transition: border-color 0.2s;
    }

    .stTextInput > div > div > input:focus {
        border-color: #3b82f6 !important;
        box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.15) !important;
    }

    .stTextInput > div > div > input::placeholder {
        color: #4b5563 !important;
    }

    /* Selectbox */
    .stSelectbox > div > div {
        background-color: #1a1f2e !important;
        border: 1.5px solid #2d3748 !important;
        border-radius: 10px !important;
        color: #ffffff !important;
    }

    /* Results count badge */
    .results-badge {
        display: inline-flex;
        align-items: center;
        gap: 0.4rem;
        background: linear-gradient(135deg, #1e3a5f, #1e4d8c);
        border: 1px solid #2563eb;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        font-size: 0.875rem;
        font-weight: 500;
        color: #93c5fd;
        margin-bottom: 1rem;
    }

    /* Dataframe */
    .stDataFrame {
        border-radius: 10px !important;
        overflow: hidden;
    }

    /* Labels */
    .stTextInput label, .stSelectbox label {
        color: #9ca3af !important;
        font-size: 0.8rem !important;
        font-weight: 500 !important;
        text-transform: uppercase !important;
        letter-spacing: 0.05em !important;
    }

    /* Subheader */
    h3 {
        color: #e5e7eb !important;
        font-weight: 600 !important;
        font-size: 1rem !important;
        letter-spacing: -0.01em !important;
    }

    /* Spinner */
    .stSpinner > div {
        border-top-color: #3b82f6 !important;
    }

    /* Warning */
    .stAlert {
        border-radius: 10px !important;
    }

    /* Method badge */
    .method-tag {
        display: inline-block;
        background: #064e3b;
        color: #6ee7b7;
        border: 1px solid #10b981;
        border-radius: 6px;
        padding: 0.2rem 0.6rem;
        font-size: 0.75rem;
        font-weight: 600;
        margin-left: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

PORTLAND_CENTER = [45.5051, -122.6750]

CATEGORY_ICONS = {
    "cafe": "☕",
    "coffee": "☕",
    "restaurant": "🍽️",
    "fast_food": "🍔",
    "pharmacy": "💊",
    "hospital": "🏥",
    "dentist": "🦷",
    "doctors": "👨‍⚕️",
    "clinic": "🏥",
    "veterinary": "🐾",
    "bank": "🏦",
    "atm": "💳",
    "parking": "🅿️",
    "parking_space": "🅿️",
    "bicycle_rental": "🚲",
    "charging_station": "⚡",
    "books": "📚",
    "clothes": "👕",
    "electronics": "💻",
    "hairdresser": "✂️",
    "pet": "🐶",
    "fuel": "⛽",
}

MARKER_COLORS = {
    "cafe": "blue",
    "coffee": "blue",
    "restaurant": "red",
    "fast_food": "red",
    "pharmacy": "green",
    "hospital": "green",
    "dentist": "green",
    "veterinary": "purple",
    "bank": "orange",
    "atm": "orange",
    "parking": "gray",
    "books": "darkblue",
    "clothes": "pink",
}


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
    from src.retrieval.query import search

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
        print('latitude in results after merge:', 'latitude' in results.columns)

        near_me = any(w in query.lower() for w in ["near me", "nearby", "close by"])
        print('near_me detected:', near_me)
        if near_me and "latitude" in results.columns and "longitude" in results.columns:
            results = combine_with_geo(results, PC[0], PC[1], score_col="hybrid_score")

        return results

    return pd.DataFrame()


# === HEADER ===
st.markdown("""
<div class="header-container">
    <p class="header-title">📍 Portland <span class="accent">POI</span> Search</p>
    <p class="header-subtitle">Search 24,918 places using natural language — powered by semantic retrieval + geo ranking</p>
</div>
""", unsafe_allow_html=True)

# === LOAD MODELS ===
with st.spinner("Loading models..."):
    df, intent_model, intent_vectorizer, vectorizer, tfidf_matrix, bm25, embedding_model, poi_embeddings = load_models()

# === SEARCH INPUT ===
col1, col2 = st.columns([4, 1])
with col1:
    query = st.text_input(
        "WHAT ARE YOU LOOKING FOR?",
        placeholder="e.g.  coffee near me  ·  vet for my dog  ·  wheelchair accessible cafe",
    )
with col2:
    method = st.selectbox("METHOD", ["Hybrid", "Embeddings", "TF-IDF"])

# === SEARCH ===
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
        st.markdown(f"""
        <div class="results-badge">
            ✦ {len(results)} results
            <span class="method-tag">{method}</span>
        </div>
        """, unsafe_allow_html=True)

        # Map
        valid_results = results[results["latitude"].notna() & results["longitude"].notna()]

        if not valid_results.empty:
            center_lat = valid_results["latitude"].mean()
            center_lon = valid_results["longitude"].mean()
        else:
            center_lat, center_lon = PORTLAND_CENTER

        m = folium.Map(
            location=[center_lat, center_lon],
            zoom_start=13,
            tiles="CartoDB dark_matter",
        )

        score_col = next((c for c in ["combined_score", "hybrid_score", "embedding_score", "similarity_score"] if c in results.columns), None)

        for i, (_, row) in enumerate(valid_results.iterrows()):
            name = row.get("name", "Unknown")
            category = row.get("category_final", "")
            address = row.get("addr:street", "")
            icon_emoji = CATEGORY_ICONS.get(category, "📍")
            marker_color = MARKER_COLORS.get(category, "blue")

            score_text = ""
            if score_col and pd.notna(row.get(score_col)):
                score_text = f"<br><span style='color:#93c5fd;font-size:11px'>Score: {row[score_col]:.3f}</span>"

            addr_text = f"<br><span style='color:#9ca3af;font-size:11px'>{address}</span>" if address and address != "None" else ""

            popup_html = f"""
            <div style='font-family:Inter,sans-serif;min-width:160px'>
                <b style='font-size:13px;color:#111'>{icon_emoji} {name}</b>
                <br><span style='color:#6b7280;font-size:11px;text-transform:uppercase'>{category}</span>
                {addr_text}
                {score_text}
            </div>
            """

            folium.Marker(
                [row["latitude"], row["longitude"]],
                popup=folium.Popup(popup_html, max_width=220),
                tooltip=f"{icon_emoji} {name}",
                icon=folium.Icon(color=marker_color, icon="circle", prefix="fa"),
            ).add_to(m)

        st_folium(m, width=None, height=480, returned_objects=[])

        # Results table
        st.subheader(f"Top {len(results)} results")

        display_cols = ["name", "category_final", "addr:street"]
        if score_col:
            display_cols.append(score_col)
            results = results.sort_values(score_col, ascending=False)

        existing_cols = [c for c in display_cols if c in results.columns]

        rename_map = {
            "name": "Name",
            "category_final": "Category",
            "addr:street": "Address",
            "hybrid_score": "Score",
            "embedding_score": "Score",
            "similarity_score": "Score",
            "combined_score": "Score",
        }

        display_df = results[existing_cols].reset_index(drop=True)
        display_df = display_df.rename(columns=rename_map)

        if "Score" in display_df.columns:
            display_df["Score"] = display_df["Score"].round(3)

        st.dataframe(display_df, width='stretch', height=350)

else:
    m = folium.Map(location=PORTLAND_CENTER, zoom_start=12, tiles="CartoDB dark_matter")
    st_folium(m, width=None, height=480, returned_objects=[])
    st.markdown("""
    <div style='text-align:center;color:#4b5563;padding:1rem 0;font-size:0.875rem'>
        Try: <code>coffee near me</code> · <code>vet for my dog</code> · <code>wheelchair accessible cafe</code> · <code>bookshop nearby</code>
    </div>
    """, unsafe_allow_html=True)