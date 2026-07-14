"""
eval_edge_cases.py
------------------
Pokrece isti eval pipeline kao eval.py, ali samo na
relevance_labels_edge_cases.csv (tacna imena, adrese, rijetki termini).

Run:
    python3 eval_edge_cases.py
"""

import eval as base_eval

# preusmjeri na edge-case fajl
base_eval.RELEVANCE_PATH = "data/relevance_labels_edge_cases.csv"
base_eval.K_VALUES = [5, 10]  # fokus na ono sto ide u UI

if __name__ == "__main__":
    base_eval.main()
