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
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    .stApp {
        background-color: #0f1117;
    }

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

    .stSelectbox > div > div {
        background-color: #1a1f2e !important;
        border: 1.5px solid #2d3748 !important;
        border-radius: 10px !important;
        color: #ffffff !important;
    }

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

    .stDataFrame {
        border-radius: 10px !important;
        overflow: hidden;
    }

    .stTextInput label, .stSelectbox label {
        color: #9ca3af !important;
        font-size: 0.8rem !important;
        font-weight: 500 !important;
        text-transform: uppercase !important;
        letter-spacing: 0.05em !important;
    }

    h3 {
        color: #e5e7eb !important;
        font-weight: 600 !important;
        font-size: 1rem !important;
        letter-spacing: -0.01em !important;
    }

    .stSpinner > div {
        border-top-color: #3b82f6 !important;
    }

    .stAlert {
        border-radius: 10px !important;
    }

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

    .location-note {
        font-size: 0.75rem;
        color: #6b7280;
        margin-top: 0.3rem;
    }

    .hours-note {
        font-size: 0.75rem;
        color: #6b7280;
        margin-top: 0.3rem;
    }
</style>
""", unsafe_allow_html=True)

PORTLAND_CENTER = [45.5051, -122.6750]

CATEGORY_ICONS = {
    "cafe": "☕", "coffee": "☕", "restaurant": "🍽️", "fast_food": "🍔",
    "pharmacy": "💊", "hospital": "🏥", "dentist": "🦷", "doctors": "👨‍⚕️",
    "clinic": "🏥", "veterinary": "🐾", "bank": "🏦", "atm": "💳",
    "parking": "🅿️", "parking_space": "🅿️", "bicycle_rental": "🚲",
    "charging_station": "⚡", "books": "📚", "clothes": "👕",
    "electronics": "💻", "hairdresser": "✂️", "pet": "🐶", "fuel": "⛽",
}

MARKER_COLORS = {
    "cafe": "blue", "coffee": "blue", "restaurant": "red", "fast_food": "red",
    "pharmacy": "green", "hospital": "green", "dentist": "green",
    "veterinary": "purple", "bank": "orange", "atm": "orange",
    "parking": "gray", "books": "darkblue", "clothes": "pink",
}

OPEN_STATUS = {
    True: ("🟢", "Open now", "#10b981"),
    False: ("🔴", "Closed", "#ef4444"),
    None: ("⚪", "Hours unknown", "#6b7280"),
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


def run_search(query, df, intent_model, intent_vectorizer, vectorizer, tfidf_matrix, bm25, embedding_model, poi_embeddings, method, user_lat, user_lon):
    from sklearn.preprocessing import MinMaxScaler
    from src.retrieval.bm25 import search_bm25
    from src.retrieval.embeddings import search_embeddings
    from src.retrieval.common import apply_geo_reranking
    from src.retrieval.geo import haversine_distance
    from src.retrieval.query import search, OPEN_NOW_SIGNAL_PHRASES
    from src.retrieval.hours import is_open_now

    q = query.lower()
    q = q.replace("opened", "open")

    # FIX: koristi jedan izvor istine (query.py::OPEN_NOW_SIGNAL_PHRASES)
    # umjesto ručno kopirane liste koja se mogla razminuti s query.py
    is_hours_query = any(w in q for w in OPEN_NOW_SIGNAL_PHRASES)

    # FIX: near_me se sad računa jednom na vrhu, koristi se i za
    # Embeddings i za Hybrid granu (prije je postojao samo unutar Hybrid)
    near_me = any(w in q for w in ["near me", "nearby", "close by", "near downtown"])

    if method == "TF-IDF":
        results = search(
            query, df,
            vectorizer=vectorizer,
            tfidf_matrix=tfidf_matrix,
            intent_model=intent_model,
            intent_vectorizer=intent_vectorizer,
            top_k=10,
            user_lat=user_lat,
            user_lon=user_lon,
        )

    elif method == "Embeddings":
        # FIX: pool se sad širi i za "near me" upite, ne samo za hours upite
        # (prije je geo rerank imao pool od svega 10 kandidata za near_me upite)
        pool_size = 100 if (is_hours_query or near_me) else 10
        results = search_embeddings(
            query, embedding_model, poi_embeddings, df,
            top_k=pool_size,
            intent_model=intent_model,
            intent_vectorizer=intent_vectorizer,
            user_lat=user_lat,
            user_lon=user_lon,
            # apply_temporal ostaje True (default) -- search_embeddings već
            # sam primjenjuje apply_temporal_filter interno. Ranije se ovdje
            # DODATNO zvao apply_open_now_filter, pa se temporal boost
            # primjenjivao dvaput na isti rezultat -- uklonjeno.
        )
        results = results.head(10)

    elif method == "Hybrid":
        # apply_temporal=False: temporal/open_now boost se primjenjuje TAČNO
        # JEDNOM, niže, nakon što se bm25+embeddings rezultati spoje. Bez ovoga
        # bi se boost primjenjivao ovdje po metodi PA JOŠ JEDNOM na hybrid_score.
        bm25_results = search_bm25(
            query, bm25, df, top_k=200,
            intent_model=intent_model,
            intent_vectorizer=intent_vectorizer,
            apply_temporal=False,
        )
        emb_results = search_embeddings(
            query, embedding_model, poi_embeddings, df, top_k=200,
            intent_model=intent_model,
            intent_vectorizer=intent_vectorizer,
            apply_temporal=False,
        )

        bm25_results = bm25_results.copy()
        emb_results = emb_results.copy()
        # poi_id already present correctly on both (search_bm25/search_embeddings
        # include it via RESULT_COLS) -- do NOT overwrite with .index, that was
        # the source of a silent ID-misalignment bug for the embeddings side.

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

        hybrid["hybrid_score"] = 0.1 * hybrid["bm25_norm"] + 0.9 * hybrid["emb_norm"]

        pool_size = 100 if (is_hours_query or near_me) else 10
        top = hybrid.sort_values("hybrid_score", ascending=False).head(pool_size)

        results = df.loc[df["poi_id"].isin(top["poi_id"])].copy()
        results = results.merge(top[["poi_id", "hybrid_score"]], on="poi_id", how="left")

        results = apply_geo_reranking(results, query, user_lat, user_lon, score_col="hybrid_score")

        if is_hours_query and "opening_hours" in results.columns:
            from src.retrieval.query import apply_open_now_filter
            sort_col = "combined_score" if "combined_score" in results.columns else "hybrid_score"
            results = apply_open_now_filter(results, query=query, boost_pct=0.2, score_col=sort_col)

        results = results.head(10)

    else:
        return pd.DataFrame()

    # Add distance_km column for transparency (regardless of method)
    if not results.empty and "latitude" in results.columns and "longitude" in results.columns:
        results = results.copy()
        results["distance_km"] = results.apply(
            lambda r: round(haversine_distance(user_lat, user_lon, r["latitude"], r["longitude"]), 2)
            if pd.notna(r.get("latitude")) and pd.notna(r.get("longitude")) else None,
            axis=1,
        )

    # Add is_open_now status for Hybrid/Embeddings too (TF-IDF already has it via search())
    if not results.empty and "opening_hours" in results.columns and "is_open_now" not in results.columns:
        results = results.copy()
        results["is_open_now"] = results["opening_hours"].apply(
            lambda h: is_open_now(h) if pd.notna(h) else None
        )

    return results


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
        placeholder="e.g.  coffee near me  ·  is the pharmacy open right now  ·  wheelchair accessible cafe",
    )
with col2:
    method = st.selectbox("METHOD", ["Hybrid", "Embeddings", "TF-IDF"])

with st.expander("📍 Set test location (default: Portland city center)"):
    loc_col1, loc_col2 = st.columns(2)
    with loc_col1:
        user_lat = st.number_input("Latitude", value=PORTLAND_CENTER[0], format="%.6f")
    with loc_col2:
        user_lon = st.number_input("Longitude", value=PORTLAND_CENTER[1], format="%.6f")
    st.markdown(
        '<p class="location-note">Try Powell\'s Books: 45.523000, -122.681500 · Pioneer Courthouse Square: 45.518900, -122.679400</p>',
        unsafe_allow_html=True,
    )

# === SEARCH ===
if query:
    with st.spinner("Searching..."):
        results = run_search(
            query, df, intent_model, intent_vectorizer,
            vectorizer, tfidf_matrix, bm25, embedding_model, poi_embeddings,
            method, user_lat, user_lon,
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

        has_hours_status = "is_open_now" in results.columns

        if has_hours_status:
            st.markdown(
                '<p class="hours-note">🟢 Open now · 🔴 Closed · ⚪ Hours unknown (open places are ranked higher; closed places are ranked lower — nothing is filtered out)</p>',
                unsafe_allow_html=True,
            )

        # Map
        valid_results = results[results["latitude"].notna() & results["longitude"].notna()]

        if not valid_results.empty:
            center_lat = valid_results["latitude"].mean()
            center_lon = valid_results["longitude"].mean()
        else:
            center_lat, center_lon = user_lat, user_lon

        m = folium.Map(
            location=[center_lat, center_lon],
            zoom_start=13,
            tiles="CartoDB dark_matter",
        )

        # Mark user location
        folium.Marker(
            [user_lat, user_lon],
            tooltip="📍 Your location",
            icon=folium.Icon(color="red", icon="user", prefix="fa"),
        ).add_to(m)

        score_col = next((c for c in ["combined_score", "hybrid_score", "embedding_score", "similarity_score"] if c in results.columns), None)

        for _, row in valid_results.iterrows():
            name = row.get("name", "Unknown")
            category = row.get("category_final", "")
            address = row.get("addr:street", "")
            distance = row.get("distance_km")
            icon_emoji = CATEGORY_ICONS.get(category, "📍")
            marker_color = MARKER_COLORS.get(category, "blue")

            score_text = ""
            if score_col and pd.notna(row.get(score_col)):
                score_text = f"<br><span style='color:#93c5fd;font-size:11px'>Score: {row[score_col]:.3f}</span>"

            dist_text = ""
            if distance is not None and pd.notna(distance):
                dist_text = f"<br><span style='color:#fbbf24;font-size:11px'>📏 {distance} km away</span>"

            hours_text = ""
            if has_hours_status:
                status_val = row.get("is_open_now")
                emoji, label, color = OPEN_STATUS.get(status_val, OPEN_STATUS[None])
                hours_text = f"<br><span style='color:{color};font-size:11px'>{emoji} {label}</span>"

            addr_text = f"<br><span style='color:#9ca3af;font-size:11px'>{address}</span>" if address and address != "None" else ""

            popup_html = f"""
            <div style='font-family:Inter,sans-serif;min-width:160px'>
                <b style='font-size:13px;color:#111'>{icon_emoji} {name}</b>
                <br><span style='color:#6b7280;font-size:11px;text-transform:uppercase'>{category}</span>
                {addr_text}
                {dist_text}
                {hours_text}
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

        if has_hours_status:
            results = results.copy()
            results["Hours"] = results["is_open_now"].apply(
                lambda v: OPEN_STATUS.get(v, OPEN_STATUS[None])[0] + " " + OPEN_STATUS.get(v, OPEN_STATUS[None])[1]
            )

        display_cols = ["name", "category_final", "addr:street", "distance_km"]
        if has_hours_status:
            display_cols.append("Hours")
        if score_col:
            display_cols.append(score_col)
            results = results.sort_values(score_col, ascending=False)

        existing_cols = [c for c in display_cols if c in results.columns]

        rename_map = {
            "name": "Name",
            "category_final": "Category",
            "addr:street": "Address",
            "distance_km": "Distance (km)",
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
    m = folium.Map(location=[user_lat, user_lon], zoom_start=12, tiles="CartoDB dark_matter")
    folium.Marker(
        [user_lat, user_lon],
        tooltip="📍 Your location",
        icon=folium.Icon(color="red", icon="user", prefix="fa"),
    ).add_to(m)
    st_folium(m, width=None, height=480, returned_objects=[])
    st.markdown("""
    <div style='text-align:center;color:#4b5563;padding:1rem 0;font-size:0.875rem'>
        Try: <code>coffee near me</code> · <code>is the pharmacy open right now</code> · <code>wheelchair accessible cafe</code> · <code>bookshop nearby</code>
    </div>
    """, unsafe_allow_html=True)