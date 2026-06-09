"""
auto_expand_labels.py
---------------------
Automatski dodaje category-match kandidate u relevance_labels_core.csv.
Za query-je gdje je category relevantnost jasna, dodaje sve POI-e odgovarajuće kategorije.
Accessibility query-ji se preskaču (zahtijevaju ručnu provjeru).
"""

import pandas as pd

# === UČITAJ PODATKE ===
df_pois = pd.read_csv("data/processed/cleaned_pois.csv")
df_pois["poi_id"] = df_pois.index  # index = poi_id

labels_df = pd.read_csv("data/relevance_labels_core.csv")

# === DEFINICIJA: query -> kategorije koje su automatski relevantne ===
# Samo query-ji gdje je odluka jasna po category_final
AUTO_EXPAND = {
    "find me a dentist": ["dentist"],
    "car repair shop nearby": ["car_repair"],
    "bookshop nearby": ["books"],
    "place for a haircut": ["hairdresser"],
    "bicycle rental nearby": ["bicycle_rental"],
    "ev charging station": ["charging_station"],
    "car wash near me": ["car_wash"],
    "coffee shop nearby": ["cafe", "coffee"],
    "pet store nearby": ["pet", "pet_grooming"],
    "tacos near me": ["restaurant", "fast_food"],  # samo taco-related, ručno
    "burger place near me": ["restaurant", "fast_food"],
    "eye doctor near me": ["optician"],
    "where can i get espresso": ["cafe", "coffee"],
    "vintage clothing store": ["clothes", "second_hand"],
    "parking garage downtown": ["parking", "parking_entrance", "parking_space"],
    "where to park my car": ["parking", "parking_entrance", "parking_space"],
    "vet for my dog": ["veterinary"],
    "urgent care near me": ["clinic"],
    "atm near city center": ["atm"],
    "need a cashpoint": ["atm"],
    "electronics store downtown": ["electronics", "radiotechnics"],
}

# Query-ji koji se preskaču (zahtijevaju ručnu provjeru wheelchair/takeaway podataka)
SKIP_QUERIES = {
    "wheelchair accessible cafe",
    "wheelchair friendly restaurant",
    "accessible pharmacy nearby",
    "disabled access hospital",
    "step free entrance cafe",
    "mobility friendly shop",
    "italian restaurant takeaway",
    "emergency room nearby",
    "place to lock my bike",
}

# === PROŠIRI LABELE ===
results = []
for _, row in labels_df.iterrows():
    query = row["query"].strip()
    current_ids = set(
        int(x.strip())
        for x in str(row["relevant_poi_ids"]).split(",")
        if x.strip()
    )

    if query in SKIP_QUERIES:
        print(f"[SKIP] '{query}' — ručna provjera potrebna")
        results.append({
            "query": query,
            "relevant_poi_ids": ",".join(str(i) for i in sorted(current_ids))
        })
        continue

    if query not in AUTO_EXPAND:
        print(f"[SKIP] '{query}' — nije u AUTO_EXPAND")
        results.append({
            "query": query,
            "relevant_poi_ids": ",".join(str(i) for i in sorted(current_ids))
        })
        continue

    cats = AUTO_EXPAND[query]
    new_ids = set(
        df_pois[df_pois["category_final"].isin(cats)]["poi_id"].tolist()
    )

    # Za tacos/burger — samo ako ime sadrži ključnu riječ (gruba provjera)
    if query == "tacos near me":
        new_ids = set(
            df_pois[
                df_pois["category_final"].isin(cats) &
                df_pois["name"].str.lower().str.contains("taco|taqueria|tacos", na=False)
            ]["poi_id"].tolist()
        )
    elif query == "burger place near me":
        new_ids = set(
            df_pois[
                df_pois["category_final"].isin(cats) &
                df_pois["name"].str.lower().str.contains("burger|burgerville|wendy|mcdonald|carl|a&w", na=False)
            ]["poi_id"].tolist()
        )
    elif query == "where can i get espresso":
        new_ids = set(
            df_pois[
                df_pois["category_final"].isin(cats) &
                df_pois["name"].str.lower().str.contains("espresso|coffee|cafe|café", na=False)
            ]["poi_id"].tolist()
        )

    added = new_ids - current_ids
    all_ids = current_ids | new_ids

    print(f"[OK] '{query}': {len(current_ids)} → {len(all_ids)} (+{len(added)} novih)")
    results.append({
        "query": query,
        "relevant_poi_ids": ",".join(str(i) for i in sorted(all_ids))
    })

# === SPREMI ===
out_df = pd.DataFrame(results)
out_df.to_csv("data/relevance_labels_expanded.csv", index=False)
print(f"\nSačuvano: data/relevance_labels_expanded.csv")
print(f"Ukupno query-ja: {len(out_df)}")