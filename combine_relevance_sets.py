"""
combine_relevance_sets.py
--------------------------
Spaja postojeci (prirodni) relevance set i edge-case set u jedan
kombinovani fajl -- koristi ga SAMO za finalni weight tuning
(tune_hybrid_weights.py), ne za dijagnostiku po tipu upita.

Run:
    python3 combine_relevance_sets.py
"""

import pandas as pd

# FIX: spajamo sa VALIDATION setom (koristi ga tune_hybrid_weights.py),
# ne sa test/expanded setom -- test set mora ostati netaknut da bi finalna
# evaluacija (eval.py) bila nepristrasna, bez data leakage-a.
NATURAL_PATH = "data/relevance_labels_validation.csv"
EDGE_CASE_PATH = "data/relevance_labels_edge_cases.csv"
OUT_PATH = "data/relevance_labels_validation_combined.csv"

natural = pd.read_csv(NATURAL_PATH)
edge = pd.read_csv(EDGE_CASE_PATH)

# provjera preklapanja prije spajanja
natural_q = set(natural["query"].astype(str).str.strip().str.lower())
edge_q = set(edge["query"].astype(str).str.strip().str.lower())
overlap = natural_q & edge_q

if overlap:
    print(f"[UPOZORENJE] {len(overlap)} preklapajucih queryja -- provjeri prije spajanja:")
    for q in overlap:
        print(f"  - {q}")
else:
    print("Nema preklapanja po queryju. OK.")

combined = pd.concat([natural, edge], ignore_index=True)
combined.to_csv(OUT_PATH, index=False)

print(f"\nSpojeno: {len(natural)} prirodnih + {len(edge)} edge-case = {len(combined)} ukupno")
print(f"Sacuvano u: {OUT_PATH}")
