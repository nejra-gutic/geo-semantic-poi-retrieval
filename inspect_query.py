import sys
import pandas as pd

from src.utils.io import load_csv
from src.retrieval import pipeline
from src.retrieval.bm25 import build_bm25, search_bm25

query = " ".join(sys.argv[1:])

df = load_csv("data/processed/cleaned_pois.csv")
df = df.rename(columns={"Unnamed: 0": "poi_id"})

if "poi_id" not in df.columns:
    df["poi_id"] = df.index

df = pipeline.run(df)

bm25 = build_bm25(df)

results = search_bm25(query, bm25, df, top_k=50)
results = results.copy()

if "poi_id" not in results.columns:
    results["poi_id"] = results.index

cols = ["poi_id", "name", "category_final", "cuisine_clean", "addr:street", "bm25_score"]
cols = [c for c in cols if c in results.columns]

print(f"\nQuery: {query}")
print(results[cols].to_string(index=False))