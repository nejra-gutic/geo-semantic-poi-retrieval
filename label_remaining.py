"""
label_remaining.py
-------------------
Labels remaining queries: missed auto-expand entries, accessibility, and hours_based.

Run:
    python3 label_remaining.py
"""

import pandas as pd

df_pois = pd.read_csv("data/processed/cleaned_pois.csv")
df_pois["poi_id"] = df_pois.index

df_labels = pd.read_csv("data/relevance_labels_expanded.csv")

# Ensure relevant_poi_ids is string type
df_labels["relevant_poi_ids"] = df_labels["relevant_poi_ids"].astype(str)


def get_ids(categories, extra_filter=None):
    subset = df_pois[df_pois["category_final"].isin(categories)]
    if extra_filter is not None:
        subset = subset[extra_filter(subset)]
    return set(subset["poi_id"].tolist())


def get_accessible_ids(categories):
    """POIs in given categories that are wheelchair accessible."""
    subset = df_pois[
        df_pois["category_final"].isin(categories)
        & (df_pois["wheelchair_accessible"] == 1)
    ]
    return set(subset["poi_id"].tolist())


def get_hours_ids(categories, require_24_7=False):
    """POIs in given categories that have opening_hours data (or is_24_7)."""
    if require_24_7:
        subset = df_pois[
            df_pois["category_final"].isin(categories)
            & (df_pois["is_24_7"] == True)
        ]
    else:
        subset = df_pois[
            df_pois["category_final"].isin(categories)
            & df_pois["opening_hours"].notna()
        ]
    return set(subset["poi_id"].tolist())


# === GRUPA 1: Missed auto-expand entries ===
GROUP_1 = {
    "cafe open early morning": get_ids(["cafe", "coffee"]),
    "need caffeine asap": get_ids(["cafe", "coffee"]),
    "place to lock my bike": get_ids(["bicycle_parking"]),
    "bus stop close by": get_ids(["bus_stop"]),
    "train station downtown": get_ids(["station", "halt"]),
    "nearest tram stop": get_ids(["tram_stop"]),
}

# === GRUPA 2: Accessibility — wheelchair_accessible=1 within relevant category ===
GROUP_2 = {
    "accessible coffee shop nearby": get_accessible_ids(["cafe", "coffee"]),
    "wheelchair access restaurant": get_accessible_ids(["restaurant"]),
    "wheelchair accessible bank": get_accessible_ids(["bank"]),
    "pharmacy with wheelchair access": get_accessible_ids(["pharmacy"]),
    "cafe with ramp entrance": get_accessible_ids(["cafe", "coffee"]),
    "restaurant suitable for wheelchair users": get_accessible_ids(["restaurant"]),
    "easy access clinic nearby": get_accessible_ids(["clinic", "doctors"]),
    "no stairs coffee shop": get_accessible_ids(["cafe", "coffee"]),
    "handicap accessible store": get_accessible_ids(["convenience", "clothes", "supermarket"]),
    "wheelchair friendly pharmacy": get_accessible_ids(["pharmacy"]),
    "wheelchair accessible grocery store": get_accessible_ids(["convenience", "supermarket"]),
    "step free pharmacy entrance": get_accessible_ids(["pharmacy"]),
    "wheelchair accessible restaurant": get_accessible_ids(["restaurant"]),
    "step free entrance pharmacy": get_accessible_ids(["pharmacy"]),
    "accessible cafe near me": get_accessible_ids(["cafe", "coffee"]),
    "restaurant with wheelchair access": get_accessible_ids(["restaurant"]),
    "can i get into this cafe with a wheelchair": get_accessible_ids(["cafe", "coffee"]),
    "mobility friendly pharmacy nearby": get_accessible_ids(["pharmacy"]),
    "disabled access restaurant": get_accessible_ids(["restaurant"]),
    "step-free coffee shop": get_accessible_ids(["cafe", "coffee"]),
    "wheelchair friendly bar": get_accessible_ids(["bar", "pub"]),
    "accessible entrance grocery store": get_accessible_ids(["convenience", "supermarket"]),
    "restaurant with no stairs": get_accessible_ids(["restaurant"]),
    "easy access cafe downtown": get_accessible_ids(["cafe", "coffee"]),
    "pharmacy with ramp entrance": get_accessible_ids(["pharmacy"]),
    "wheelchair accessible seating restaurant": get_accessible_ids(["restaurant"]),
    "can i bring a wheelchair into this cafe": get_accessible_ids(["cafe", "coffee"]),
    "accessible public restroom cafe": get_accessible_ids(["cafe", "coffee"]),
    "barrier free restaurant nearby": get_accessible_ids(["restaurant"]),
    "step free bakery": get_accessible_ids(["bakery"]),
    "accessible fast food place": get_accessible_ids(["fast_food"]),
    "restaurant suitable for mobility scooter users": get_accessible_ids(["restaurant"]),
    "which cafes have wheelchair access": get_accessible_ids(["cafe", "coffee"]),
    "disabled friendly pharmacy open now": get_accessible_ids(["pharmacy"]),
    "wheelchair accessible coffee place": get_accessible_ids(["cafe", "coffee"]),
    "restraunt with wheelchair access": get_accessible_ids(["restaurant"]),
    "step free entrnace cafe": get_accessible_ids(["cafe", "coffee"]),
    "accessible shop and cafe nearby": get_accessible_ids(["cafe", "coffee", "convenience", "clothes"]),
    "mobility scooter friendly restaurant": get_accessible_ids(["restaurant"]),
    "cafe with ramp access": get_accessible_ids(["cafe", "coffee"]),
    "pharmacy accessible for disabled customers": get_accessible_ids(["pharmacy"]),
    "wheelchair access near entrance restaurant": get_accessible_ids(["restaurant"]),
}

# === GRUPA 3: Hours-based — POIs with opening_hours OR is_24_7, within category ===
GROUP_3 = {
    "24 hour pharmacy nearby": get_hours_ids(["pharmacy"], require_24_7=True),
    "cafe open late tonight": get_hours_ids(["cafe", "coffee"]),
    "restaurant open right now": get_hours_ids(["restaurant", "fast_food"]),
    "coffee shop open early morning": get_hours_ids(["cafe", "coffee"]),
    "grocery store open 24 7": get_hours_ids(["convenience", "supermarket"], require_24_7=True),
    "late night food nearby": get_hours_ids(["restaurant", "fast_food"]),
    "pharmacy open after midnight": get_hours_ids(["pharmacy"]),
    "breakfast place open now": get_hours_ids(["cafe", "restaurant"]),
    "what cafes are still open": get_hours_ids(["cafe", "coffee"]),
    "open atm nearby right now": get_ids(["atm"]),  # ATMs are always available
    "diner open all night": get_hours_ids(["restaurant", "fast_food"]),
    "who is open this late": get_hours_ids(["restaurant", "fast_food", "cafe", "bar"]),
    "is the pharmacy open right now": get_hours_ids(["pharmacy"]),
    "what time does the bank close today": get_hours_ids(["bank"]),
    "coffee shop open late tonight": get_hours_ids(["cafe", "coffee"]),
    "is there a cafe still open": get_hours_ids(["cafe", "coffee"]),
    "when does the post office open": get_hours_ids(["post_office"]),
    "what are the bakery hours today": get_hours_ids(["bakery"]),
    "grocery store open now": get_hours_ids(["convenience", "supermarket"]),
    "does the pharmacy close at 8": get_hours_ids(["pharmacy"]),
    "what time does target close": get_hours_ids(["supermarket", "department_store"]),
    "is the library open on sunday": get_ids(["library"]),
    "late night food near me open now": get_hours_ids(["restaurant", "fast_food"]),
    "which cafes are open early morning": get_hours_ids(["cafe", "coffee"]),
    "atm available 24 hours nearby": get_ids(["atm"]),
    "restaurant open after midnight": get_hours_ids(["restaurant", "fast_food", "bar"]),
    "what time does the nearest gas station close": get_hours_ids(["fuel"]),
    "is the dentist office open today": get_hours_ids(["dentist"]),
    "supermarket opening hours": get_hours_ids(["supermarket"]),
    "are any pharmacies open 24/7": get_hours_ids(["pharmacy"], require_24_7=True),
    "coffee place open rn": get_hours_ids(["cafe", "coffee"]),
    "bank hours saturday": get_hours_ids(["bank"]),
    "what time do they stop serving breakfast": get_hours_ids(["cafe", "restaurant"]),
    "bookstore open this evening": get_hours_ids(["books"]),
    "any food places open right now": get_hours_ids(["restaurant", "fast_food"]),
    "when does the cafe start serving coffee": get_hours_ids(["cafe", "coffee"]),
    "open pharmacy near me now": get_hours_ids(["pharmacy"]),
    "is the bank still open": get_hours_ids(["bank"]),
    "what time does the hair salon close today": get_hours_ids(["hairdresser"]),
    "restaurants open on christmas day": get_hours_ids(["restaurant"]),
    "which stores are open this late": get_hours_ids(["convenience", "clothes", "supermarket"]),
    "bakery open before 7am": get_hours_ids(["bakery"]),
    "what time dose the pharmacy close": get_hours_ids(["pharmacy"]),
    "is the coffe shop open yet": get_hours_ids(["cafe", "coffee"]),
    "grocery store hours today": get_hours_ids(["convenience", "supermarket"]),
    "what time does that resturant close": get_hours_ids(["restaurant", "fast_food"]),
    "open late cafes downtown": get_hours_ids(["cafe", "coffee"]),
}

ALL_NEW = {**GROUP_1, **GROUP_2, **GROUP_3}

updated = 0
for query, ids in ALL_NEW.items():
    mask = df_labels["query"] == query
    if mask.sum() == 0:
        print(f"[WARN] Query not found in dataframe: '{query}'")
        continue

    id_str = ",".join(str(i) for i in sorted(ids))
    df_labels.loc[mask, "relevant_poi_ids"] = id_str
    print(f"[OK] '{query}': {len(ids)} relevant POIs")
    updated += 1

df_labels.to_csv("data/relevance_labels_expanded.csv", index=False)

print(f"\nUpdated {updated} queries.")
total = len(df_labels)
with_labels = (df_labels["relevant_poi_ids"].notna() & (df_labels["relevant_poi_ids"] != "") & (df_labels["relevant_poi_ids"] != "nan")).sum()
print(f"Total queries: {total}")
print(f"With labels: {with_labels}")
