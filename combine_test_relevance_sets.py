"""
combine_test_relevance_sets.py
--------------------------------
Spaja postojeci finalni test/expanded set sa NOVIM edge-case upitima
(razlicitim od onih koriscenih za validation tuning) da bi eval.py
davao realniju konacnu ocjenu.

VAZNO: pokreni ovo SAMO JEDNOM, nakon sto su tezine (w_bm25/w_emb)
vec izabrane preko tune_hybrid_weights.py na validation setu.
Ne diraj ovaj fajl ponovo dok ne mijenjas metodologiju testiranja.

Run:
    python3 combine_test_relevance_sets.py
"""

import pandas as pd

# prilagodi ako ti je finalni test fajl "expanded" umjesto "test"
EXISTING_TEST_PATH = "data/relevance_labels_expanded.csv"
NEW_EDGECASE_PATH = "data/relevance_labels_test_edgecases.csv"
OUT_PATH = "data/relevance_labels_expanded_v2.csv"

existing = pd.read_csv(EXISTING_TEST_PATH)
new_edge = pd.read_csv(NEW_EDGECASE_PATH)

existing_q = set(existing["query"].astype(str).str.strip().str.lower())
new_q = set(new_edge["query"].astype(str).str.strip().str.lower())

# provjera 1: da se novi test-edgecase upiti ne preklapaju sa postojecim testom
overlap_existing = existing_q & new_q
if overlap_existing:
    print(f"[UPOZORENJE] {len(overlap_existing)} query-ja se preklapa sa postojecim test setom:")
    for q in overlap_existing:
        print(f"  - {q}")
else:
    print("Nema preklapanja sa postojecim test setom. OK.")

# provjera 2: podsjetnik da provjeris i da se NE preklapaju sa validation edge-case setom
try:
    validation_edge = pd.read_csv("data/relevance_labels_edge_cases.csv")
    validation_q = set(validation_edge["query"].astype(str).str.strip().str.lower())
    overlap_validation = validation_q & new_q
    if overlap_validation:
        print(f"\n[UPOZORENJE] {len(overlap_validation)} query-ja se preklapa sa VALIDATION edge-case setom "
              f"-- ovo je data leakage, ukloni ih prije nastavka:")
        for q in overlap_validation:
            print(f"  - {q}")
    else:
        print("Nema preklapanja sa validation edge-case setom. OK.")
except FileNotFoundError:
    print("[INFO] data/relevance_labels_edge_cases.csv nije nadjen, preskačem tu provjeru.")

combined = pd.concat([existing, new_edge], ignore_index=True)
combined.to_csv(OUT_PATH, index=False)

print(f"\nSpojeno: {len(existing)} postojecih + {len(new_edge)} novih edge-case = {len(combined)} ukupno")
print(f"Sacuvano u: {OUT_PATH}")
print(f"\nSljedece: promijeni RELEVANCE_PATH u eval.py na '{OUT_PATH}' i pokreni finalni eval.")
