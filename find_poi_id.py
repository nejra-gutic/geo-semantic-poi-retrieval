"""
find_poi_id_batch.py
---------------------
Pretrazuje VISE termina odjednom (umjesto da pokrecesh find_poi_id.py
15 puta rucno). Ispisuje rezultate grupisano po terminu.

Uredi SEARCH_TERMS listu ispod sa terminima koji tebi imaju smisla
(imena POI-ja koja MISLIS da postoje u tvom gradu/datasetu, adrese,
rijetke kategorije) -- ja ne znam sta stvarno postoji u tvom CSV-u,
ovo je samo alat da brzo provjeris.

Run:
    python3 find_poi_id_batch.py
"""

import pandas as pd
from src.utils.io import load_csv
from src.retrieval import pipeline

DATA_PATH = "data/processed/cleaned_pois.csv"

# === UREDI OVU LISTU ===
# Format: (search_term, column_to_search)
# column_to_search je obicno "name" ili "addr:street" ili "category_final"
SEARCH_TERMS = [
    ("rudy's barbershop", "name"),
    ("rocky butte espresso", "name"),
    ("old church concert hall", "name"),
    ("division street", "addr:street"),
    ("morrison street", "addr:street"),
    ("capitol highway", "addr:street"),
    ("13th avenue", "addr:street"),
    ("chávez", "addr:street"),
    ("alder street", "addr:street"),
    ("letter_box", "category_final"),
    ("waste_basket", "category_final"),
    ("bicycle_parking", "category_final"),
    ("place_of_worship", "category_final"),
    ("parking_entrance", "category_final"),
    ("bar", "category_final"),
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