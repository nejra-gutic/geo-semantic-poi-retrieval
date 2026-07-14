"""
find_poi_id_batch_v2.py
-------------------------
Druga (nezavisna) batch pretraga -- termini ovdje NAMJERNO ne preklapaju
se sa onima koriscenim u relevance_labels_edge_cases.csv (koji ide u
VALIDATION). Ovi se koriste za relevance_labels_test_edgecases.csv (TEST).

Run:
    python3 find_poi_id_batch_v2.py
"""

import pandas as pd
from src.utils.io import load_csv
from src.retrieval import pipeline

DATA_PATH = "data/processed/cleaned_pois.csv"

SEARCH_TERMS = [
    ("sellwood moreland library", "name"),
    ("hillsdale family dental", "name"),
    ("regal cinemas pioneer place", "name"),
    ("nick rinard physical therapy", "name"),
    ("hawthorne boulevard", "addr:street"),
    ("sandy boulevard", "addr:street"),
    ("martin luther king", "addr:street"),
    ("stark street", "addr:street"),
    ("foster road", "addr:street"),
    ("tattoo", "category_final"),
    ("tailor", "category_final"),
    ("cinema", "category_final"),
    ("optician", "category_final"),
    ("ice_cream", "category_final"),
    ("car_wash", "category_final"),
]


def main():
    df = load_csv(DATA_PATH)
    df["poi_id"] = df.index
    original_poi_ids = df["poi_id"].copy()
    df = pipeline.run(df)
    df["poi_id"] = original_poi_ids.values

    show_cols = [c for c in ["poi_id", "name", "category_final", "addr:street"] if c in df.columns]

    for term, col in SEARCH_TERMS:
        if col not in df.columns:
            print(f"\n[SKIP] Kolona '{col}' ne postoji za term '{term}'")
            continue

        matches = df[df[col].astype(str).str.lower().str.contains(term.lower(), na=False)]

        print(f"\n{'=' * 60}")
        print(f"Term: '{term}'  (kolona: {col})  -> {len(matches)} rezultata")
        print(f"{'=' * 60}")

        if matches.empty:
            print("  (nema rezultata)")
        else:
            print(matches[show_cols].head(10).to_string(index=False))


if __name__ == "__main__":
    main()
