"""
auto_expand_labels.py
---------------------
Automatski dodaje category-match kandidate u relevance_labels_core.csv.
Za query-je gdje je category relevantnost jasna, dodaje sve POI-e odgovarajuće kategorije.
Accessibility i hours_based query-ji se preskaču (zahtijevaju ručnu provjeru).
"""

import pandas as pd

# === UČITAJ PODATKE ===
df_pois = pd.read_csv("data/processed/cleaned_pois.csv")
df_pois["poi_id"] = df_pois.index  # index = poi_id

labels_df = pd.read_csv("data/relevance_labels_core.csv")

# === DEFINICIJA: query -> kategorije koje su automatski relevantne ===
AUTO_EXPAND = {
    # Original
    "find me a dentist": ["dentist"],
    "car repair shop nearby": ["car_repair"],
    "bookshop nearby": ["books"],
    "place for a haircut": ["hairdresser"],
    "bicycle rental nearby": ["bicycle_rental"],
    "ev charging station": ["charging_station"],
    "car wash near me": ["car_wash"],
    "coffee shop nearby": ["cafe", "coffee"],
    "pet store nearby": ["pet", "pet_grooming"],
    "tacos near me": ["restaurant", "fast_food"],
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
    # find_cafe (batch 2)
    "best cappuccino in town": ["cafe", "coffee"],
    "cozy cafe for studying": ["cafe", "coffee"],
    "latte near downtown": ["cafe", "coffee"],
    "good espresso bar nearby": ["cafe", "coffee"],
    "quiet cafe with wifi": ["cafe", "coffee"],
    "need coffee asap": ["cafe", "coffee"],
    "coffee and pastries near me": ["cafe", "coffee"],
    "trendy cafe downtown": ["cafe", "coffee"],
    "where can i grab a mocha": ["cafe", "coffee"],
    "breakfast cafe around here": ["cafe", "coffee"],
    "iced coffee place nearby": ["cafe", "coffee"],
    "cafe with outdoor seating": ["cafe", "coffee"],
    "good place for a flat white": ["cafe", "coffee"],
    "neighborhood coffee house": ["cafe", "coffee"],
    "grab a quick coffee": ["cafe", "coffee"],
    "cafe closeby": ["cafe", "coffee"],
    "coffe shop near me": ["cafe", "coffee"],
    "any good cafes around": ["cafe", "coffee"],
    "need caffeine nearby": ["cafe", "coffee"],
    "coffee place with free wifi": ["cafe", "coffee"],
    "best local roastery nearby": ["cafe", "coffee"],
    "coffe near me": ["cafe", "coffee"],
    # find_food (batch 2)
    "mexican food nearby": ["restaurant", "fast_food"],
    "sushi place downtown": ["restaurant"],
    "best pizza around here": ["restaurant", "fast_food"],
    "thai food near me": ["restaurant"],
    "chinese takeout nearby": ["restaurant", "fast_food"],
    "vegan restaurant downtown": ["restaurant"],
    "bbq joint close by": ["restaurant", "fast_food"],
    "fried chicken near me": ["restaurant", "fast_food"],
    "seafood restaurant nearby": ["restaurant"],
    "ramen shop downtown": ["restaurant"],
    "lunch spot around here": ["restaurant", "fast_food", "cafe"],
    "where can i get tacos al pastor": ["restaurant", "fast_food"],
    "kebab place nearby": ["restaurant", "fast_food"],
    "family friendly restaurant": ["restaurant"],
    "fancy dinner place downtown": ["restaurant"],
    "quick bite near me": ["restaurant", "fast_food"],
    "good brunch restaurant": ["restaurant", "cafe"],
    "pasta place open today": ["restaurant"],
    "best burgers downtown": ["restaurant", "fast_food"],
    "food near me pls": ["restaurant", "fast_food"],
    "dinner near me tonight": ["restaurant"],
    "takeaway pizza downtown": ["restaurant", "fast_food"],
    "good sushi rn": ["restaurant"],
    # find_service (batch 2)
    "nearest pharmacy open now": ["pharmacy"],
    "doctor office nearby": ["doctors", "clinic"],
    "family dentist downtown": ["dentist"],
    "need a bank branch": ["bank"],
    "hospital close by": ["hospital"],
    "find me a clinic": ["clinic", "doctors"],
    "urgent medical help nearby": ["hospital", "clinic"],
    "atm close to me": ["atm"],
    "where is the nearest bank": ["bank"],
    "pediatric doctor nearby": ["doctors", "clinic"],
    "healthcare center downtown": ["clinic", "doctors", "hospital"],
    "need cash withdrawal": ["atm"],
    "medical center around here": ["clinic", "doctors", "hospital"],
    "closest dental clinic": ["dentist"],
    "pharmacy near downtown": ["pharmacy"],
    "walk in clinic nearby": ["clinic"],
    "blood test center near me": ["clinic", "doctors"],
    "doctor near me asap": ["doctors", "clinic"],
    "need an atm machine": ["atm"],
    "nearest healthcare facility": ["hospital", "clinic", "doctors"],
    "nearest dentist accepting walk ins": ["dentist"],
    "bank machine nearby": ["atm"],
    "need doctor asap": ["doctors", "clinic"],
    # find_shop (batch 2)
    "grocery store nearby": ["convenience", "supermarket"],
    "supermarket around here": ["supermarket", "convenience"],
    "clothing shop downtown": ["clothes"],
    "sneaker store nearby": ["shoes"],
    "toy store close by": ["toys"],
    "where can i buy pet food": ["pet", "pet_grooming"],
    "furniture shop near me": ["furniture"],
    "gift store downtown": ["gift"],
    "hardware store nearby": ["hardware"],
    "florist around here": ["florist"],
    "mobile phone shop downtown": ["electronics"],
    "computer store nearby": ["electronics"],
    "buy headphones near me": ["electronics"],
    "shopping for books downtown": ["books"],
    "local market nearby": ["convenience", "supermarket"],
    "thrift store around here": ["second_hand", "clothes"],
    "fashion boutique nearby": ["clothes"],
    "grocery shop open now": ["convenience", "supermarket"],
    "cheap clothes store": ["clothes", "second_hand"],
    "sports equipment shop nearby": ["sports"],
    "buy a birthday gift nearby": ["gift"],
    "electronics shop near me pls": ["electronics", "radiotechnics"],
    "pet supplies nearby": ["pet", "pet_grooming"],
    # find_transport (batch 2)
    "gas station nearby": ["fuel"],
    "nearest fuel station": ["fuel"],
    "bus stop close by": ["bus_stop"],
    "train station downtown": ["station"],
    "bike parking nearby": ["bicycle_parking"],
    "public parking lot": ["parking", "parking_space"],
    "parking near city center": ["parking", "parking_space", "parking_entrance"],
    "charge my electric car": ["charging_station"],
    "ev charger close by": ["charging_station"],
    "rent a bike nearby": ["bicycle_rental"],
    "bicycle parking downtown": ["bicycle_parking"],
    "where can i park": ["parking", "parking_space", "parking_entrance"],
    "car rental nearby": ["car_rental"],
    "taxi stand downtown": ["taxi"],
    "park and ride location": ["parking"],
    "motorcycle parking nearby": ["motorcycle_parking"],
    "nearest tram stop": ["tram_stop"],
    "charging station for ev": ["charging_station"],
    "need parking asap": ["parking", "parking_space"],
    "secure bicycle parking": ["bicycle_parking"],
    "parking structure downtown": ["parking", "parking_entrance"],
    "fuel up my car nearby": ["fuel"],
    "bike rental station downtown": ["bicycle_rental"],

    # === BATCH 3 (new, June sync expansion) ===
    # find_cafe
    "good coffee near me": ["cafe", "coffee"],
    "espresso nearby": ["cafe", "coffee"],
    "cafe with wifi": ["cafe", "coffee"],
    "quiet coffee shop to work from": ["cafe", "coffee"],
    "where can i get a decent latte": ["cafe", "coffee"],
    "coffee and pastries nearby": ["cafe", "coffee"],
    "need a cafe for studying": ["cafe", "coffee"],
    "local coffee place open now": ["cafe", "coffee"],
    "any specialty coffee around here": ["cafe", "coffee"],
    "cheap coffee nearby": ["cafe", "coffee"],
    "best espresso downtown": ["cafe", "coffee"],
    "grab a quick coffee": ["cafe", "coffee"],
    "somewhere to work and drink coffee": ["cafe", "coffee"],
    "late night cafe": ["cafe", "coffee"],
    "good iced coffee around here": ["cafe", "coffee"],
    "where can i sit and read with coffee": ["cafe", "coffee"],
    "coffee near burnside": ["cafe", "coffee"],
    "coffee place with good internet": ["cafe", "coffee"],
    "independent cafe nearby": ["cafe", "coffee"],
    "breakfast and coffee spot": ["cafe", "coffee"],
    "small cozy cafe": ["cafe", "coffee"],
    "good flat white nearby": ["cafe", "coffee"],
    "cafe near transit stop": ["cafe", "coffee"],
    "best coffee for remote work": ["cafe", "coffee"],
    "latte and croissant near me": ["cafe", "coffee"],
    "where's a nice cafe to meet a friend": ["cafe", "coffee"],
    "cafee open now": ["cafe", "coffee"],
    "coffee shop close by": ["cafe", "coffee"],
    "place for pour over coffee": ["cafe", "coffee"],
    "best mocha in portland": ["cafe", "coffee"],

    # find_food
    "mexican food near me": ["restaurant", "fast_food"],
    "where can i get tacos": ["restaurant", "fast_food"],
    "good pizza nearby": ["restaurant", "fast_food"],
    "hungry need food now": ["restaurant", "fast_food"],
    "best burger place around here": ["restaurant", "fast_food"],
    "thai food close by": ["restaurant"],
    "restaurant for dinner tonight": ["restaurant"],
    "sushi near downtown": ["restaurant"],
    "cheap lunch spot": ["restaurant", "fast_food"],
    "where can i get ramen": ["restaurant"],
    "good indian food nearby": ["restaurant"],
    "resturant with good tacos": ["restaurant", "fast_food"],
    "family friendly restaurant near me": ["restaurant"],
    "late night food options": ["restaurant", "fast_food"],
    "best fried chicken around here": ["restaurant", "fast_food"],

    # find_transport
    "parking near me": ["parking", "parking_space", "parking_entrance"],
    "where can i park downtown": ["parking", "parking_space", "parking_entrance"],
    "charging station close by": ["charging_station"],
    "need gas station": ["fuel"],
    "car wash around here": ["car_wash"],
    "electric vehicle charging nearby": ["charging_station"],
    "nearest parking garage": ["parking", "parking_entrance"],
    "motorcycle parking downtown": ["motorcycle_parking"],
    "fuel station open now": ["fuel"],
    "taxi stand close by": ["taxi"],
    "parkng near convention center": ["parking", "parking_space"],
    "where can i charge my car tonight": ["charging_station"],

    # find_shop
    "convenience store open now": ["convenience"],
    "clothing shop near me": ["clothes"],
    "buy gifts nearby": ["gift"],
    "supermarket close by": ["supermarket"],
    "shoe store around here": ["shoes"],
    "shop open late tonight": ["convenience", "clothes"],
    "book store nearby": ["books"],
    "nearest furniture store": ["furniture"],
    "convience store close by": ["convenience"],

    # find_service
    "pharmacy near me": ["pharmacy"],
    "find a dentist nearby": ["dentist"],
    "hospital close by": ["hospital"],
    "atm nearby": ["atm"],
    "doctor office open today": ["doctors", "clinic"],
    "hair salon nearby": ["hairdresser"],
    "pharamcy open now": ["pharmacy"],
    "where can i get a prescription filled": ["pharmacy"],
}

# Query-ji koji se preskaču (zahtijevaju ručnu provjeru wheelchair/hours podataka)
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
    # accessibility (batch 2)
    "accessible coffee shop nearby",
    "wheelchair access restaurant",
    "wheelchair accessible bank",
    "pharmacy with wheelchair access",
    "cafe with ramp entrance",
    "restaurant suitable for wheelchair users",
    "easy access clinic nearby",
    "no stairs coffee shop",
    "handicap accessible store",
    "wheelchair friendly pharmacy",
    "wheelchair accessible grocery store",
    "step free pharmacy entrance",
    # hours_based (batch 2)
    "24 hour pharmacy nearby",
    "cafe open late tonight",
    "restaurant open right now",
    "coffee shop open early morning",
    "grocery store open 24 7",
    "late night food nearby",
    "pharmacy open after midnight",
    "breakfast place open now",
    "what cafes are still open",
    "open atm nearby right now",
    "diner open all night",
    "who is open this late",

    # === BATCH 3 (accessibility) ===
    "wheelchair accessible restaurant",
    "step free entrance pharmacy",
    "accessible cafe near me",
    "restaurant with wheelchair access",
    "can i get into this cafe with a wheelchair",
    "mobility friendly pharmacy nearby",
    "disabled access restaurant",
    "step-free coffee shop",
    "wheelchair friendly bar",
    "accessible entrance grocery store",
    "restaurant with no stairs",
    "easy access cafe downtown",
    "pharmacy with ramp entrance",
    "wheelchair accessible seating restaurant",
    "can i bring a wheelchair into this cafe",
    "accessible public restroom cafe",
    "barrier free restaurant nearby",
    "step free bakery",
    "accessible fast food place",
    "restaurant suitable for mobility scooter users",
    "which cafes have wheelchair access",
    "disabled friendly pharmacy open now",
    "wheelchair accessible coffee place",
    "restraunt with wheelchair access",
    "step free entrnace cafe",
    "accessible shop and cafe nearby",
    "mobility scooter friendly restaurant",
    "cafe with ramp access",
    "pharmacy accessible for disabled customers",
    "wheelchair access near entrance restaurant",

    # === BATCH 3 (hours_based) ===
    "is the pharmacy open right now",
    "what time does the bank close today",
    "coffee shop open late tonight",
    "is there a cafe still open",
    "when does the post office open",
    "what are the bakery hours today",
    "grocery store open now",
    "does the pharmacy close at 8",
    "what time does target close",
    "is the library open on sunday",
    "late night food near me open now",
    "which cafes are open early morning",
    "atm available 24 hours nearby",
    "restaurant open after midnight",
    "what time does the nearest gas station close",
    "is the dentist office open today",
    "supermarket opening hours",
    "are any pharmacies open 24/7",
    "coffee place open rn",
    "bank hours saturday",
    "what time do they stop serving breakfast",
    "bookstore open this evening",
    "any food places open right now",
    "when does the cafe start serving coffee",
    "open pharmacy near me now",
    "is the bank still open",
    "what time does the hair salon close today",
    "restaurants open on christmas day",
    "which stores are open this late",
    "bakery open before 7am",
    "what time dose the pharmacy close",
    "is the coffe shop open yet",
    "grocery store hours today",
    "what time does that resturant close",
    "open late cafes downtown",
}

# === PROŠIRI LABELE ===
results = []
for _, row in labels_df.iterrows():
    query = row["query"].strip()
    current_ids = set(
        int(x.strip())
        for x in str(row["relevant_poi_ids"]).split(",")
        if x.strip() and x.strip() != "nan"
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

    # Posebni filteri za specifične query-je
    if query in ["tacos near me", "where can i get tacos al pastor", "where can i get tacos"]:
        new_ids = set(
            df_pois[
                df_pois["category_final"].isin(cats) &
                df_pois["name"].str.lower().str.contains("taco|taqueria|tacos", na=False)
            ]["poi_id"].tolist()
        )
    elif query in ["burger place near me", "best burgers downtown", "best burger place around here"]:
        new_ids = set(
            df_pois[
                df_pois["category_final"].isin(cats) &
                df_pois["name"].str.lower().str.contains("burger|burgerville|wendy|mcdonald|carl|a&w", na=False)
            ]["poi_id"].tolist()
        )
    elif query in ["where can i get espresso", "good espresso bar nearby", "espresso nearby", "best espresso downtown"]:
        new_ids = set(
            df_pois[
                df_pois["category_final"].isin(cats) &
                df_pois["name"].str.lower().str.contains("espresso|coffee|cafe|café", na=False)
            ]["poi_id"].tolist()
        )
    elif query in ["sushi place downtown", "good sushi rn", "sushi near downtown"]:
        new_ids = set(
            df_pois[
                df_pois["category_final"].isin(cats) &
                df_pois["name"].str.lower().str.contains("sushi|japanese|ramen", na=False)
            ]["poi_id"].tolist()
        )
    elif query in ["where can i get ramen"]:
        new_ids = set(
            df_pois[
                df_pois["category_final"].isin(cats) &
                df_pois["name"].str.lower().str.contains("ramen|japanese", na=False)
            ]["poi_id"].tolist()
        )
    elif query in ["good pizza nearby"]:
        new_ids = set(
            df_pois[
                df_pois["category_final"].isin(cats) &
                df_pois["name"].str.lower().str.contains("pizza", na=False)
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