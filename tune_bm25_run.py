"""
tune_bm25_run.py
----------------
Pokreće grid search za BM25 parametre k1 i b.
"""

from src.utils.io import load_csv
from src.retrieval import pipeline
from src.retrieval.intent_classifier import load_model
from src.retrieval.bm25 import build_bm25, tune_bm25

df = load_csv("data/processed/cleaned_pois.csv")
df["poi_id"] = df.index
from pandas import Series
original_poi_ids = df["poi_id"].copy()
df = pipeline.run(df)
df["poi_id"] = original_poi_ids.values

intent_model, intent_vectorizer = load_model("models/intent_classifier.pkl")
bm25 = build_bm25(df)

evaluation_queries = [
    {"query": "find me a dentist",          "expected_category": "dentist"},
    {"query": "coffee shop nearby",          "expected_category": "cafe"},
    {"query": "burger place near me",        "expected_category": "restaurant"},
    {"query": "car wash near me",            "expected_category": "car_wash"},
    {"query": "car repair shop nearby",      "expected_category": "car_repair"},
    {"query": "pet store nearby",            "expected_category": "pet"},
    {"query": "bookshop nearby",             "expected_category": "books"},
    {"query": "bicycle rental nearby",       "expected_category": "bicycle_rental"},
    {"query": "ev charging station",         "expected_category": "charging_station"},
    {"query": "vet for my dog",              "expected_category": "veterinary"},
    {"query": "eye doctor near me",          "expected_category": "optician"},
    {"query": "urgent care near me",         "expected_category": "clinic"},
    {"query": "vintage clothing store",      "expected_category": "clothes"},
    {"query": "electronics store downtown",  "expected_category": "electronics"},
    {"query": "tacos near me",              "expected_category": "restaurant"},
    {"query": "parking garage downtown",     "expected_category": "parking"},
    {"query": "where can i get espresso",    "expected_category": "cafe"},
]

tune_bm25(
    df,
    evaluation_queries=evaluation_queries,
    intent_model=intent_model,
    intent_vectorizer=intent_vectorizer,
)