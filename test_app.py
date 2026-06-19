import streamlit as st
import folium
from streamlit_folium import st_folium

PORTLAND = [45.5152, -122.6784]

query = st.text_input(
    "What are you looking for?",
    placeholder="E.g. wheelchair accessible cafe takeaway open 24/7"
)

# Simulacija retrieval rezultata
results = [
    {
        "name": "Coffee A",
        "lat": 45.52,
        "lon": -122.67,
        "score": 0.95,
    },
    {
        "name": "Coffee B",
        "lat": 45.51,
        "lon": -122.69,
        "score": 0.91,
    },
]

m = folium.Map(
    location=PORTLAND,
    zoom_start=12,
    tiles="OpenStreetMap"
)

for r in results:
    folium.Marker(
        [r["lat"], r["lon"]],
        popup=f"{r['name']}<br>Score: {r['score']:.2f}",
    ).add_to(m)

st_folium(
    m,
    width=900,
    height=600
)