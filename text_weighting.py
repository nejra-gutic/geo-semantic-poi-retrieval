import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from src.utils.io import load_csv
from src.retrieval import pipeline

# Učitaj dataset
df = load_csv("data/processed/cleaned_pois.csv")
df = pipeline.run(df)

test_queries = [
    "mexican restaurant",
    "coffee near burnside",
    "italian restaurant",
]

# Opcija A — name 2x, category 2x
df["poi_text_A"] = (
    df[["name_norm", "name_norm", "brand_norm", 
        "category_final", "category_final", 
        "cuisine_clean", "addr_norm"]]
    .fillna("")
    .apply(lambda row: " ".join(row.values.astype(str)), axis=1)
)

# Opcija B — name 3x, cuisine 2x
df["poi_text_B"] = (
    df[["name_norm", "name_norm", "name_norm", "brand_norm", 
        "category_final", "cuisine_clean", "cuisine_clean", "addr_norm"]]
    .fillna("")
    .apply(lambda row: " ".join(row.values.astype(str)), axis=1)
)

for test_query in test_queries:
    print(f"\n{'='*50}")
    print(f"Query: '{test_query}'")

    for label, col in [("Opcija A (name 2x, cat 2x)", "poi_text_A"), 
                        ("Opcija B (name 3x, cuisine 2x)", "poi_text_B")]:
        vec = TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=5000,
            token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z]+\b"
        )
        mat = vec.fit_transform(df[col].fillna(""))
        q_vec = vec.transform([test_query])
        scores = cosine_similarity(q_vec, mat).flatten()
        top = np.argsort(scores)[::-1][:5]

        print(f"\n  {label}:")
        for i in top:
            print(f"    {df.iloc[i]['name']} | {df.iloc[i]['category_final']} | score: {round(scores[i], 3)}")