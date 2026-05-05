import pandas as pd
from src.utils.io import load_csv
from src.retrieval import pipeline, query, tfidf
from src.retrieval.intent_classifier import load_model

intent_model, intent_vectorizer = load_model("models/intent_classifier.pkl")


# Učitaj cleaned dataset
df = load_csv("data/processed/cleaned_pois.csv")

# Pokreni retrieval pipeline (tokenize + normalize + linguistic)
df = pipeline.run(df)

# Buildi TF-IDF
vectorizer, tfidf_matrix = tfidf.run(df)

# Testiraj upite
test_queries = [
    "coffee near burnside",
    "mexican restaurant",
    "wheelchair accessible cafe",
    "24/7 pharmacy",
    "italian restaurant takeaway",
]

for q in test_queries:
    print("\n" + "=" * 50)
    results = query.search(
        q, df,
        vectorizer=vectorizer,
        tfidf_matrix=tfidf_matrix,
        intent_model=intent_model,
        intent_vectorizer=intent_vectorizer,
        top_k=5
    )
    print(results[["name", "category_final", "addr:street", "similarity_score"]].to_string())