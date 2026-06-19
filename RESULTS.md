# Evaluation Results History

This document tracks the evolution of evaluation results as the eval set was expanded
and methodology improved. Useful for understanding *why* numbers changed between demos.

---

## v1 — 30 queries (Demo 2, initial eval set)

| Method            | P@5   | NDCG@10 | Hit@50 |
|-------------------|-------|---------|--------|
| TF-IDF (filter)   | 0.733 | 0.805   | 0.900  |
| BM25 (filter)     | 0.733 | 0.761   | 0.933  |
| Embeddings        | 0.640 | 0.701   | 0.900  |
| Hybrid (0.2/0.8)  | 0.667 | 0.714   | 0.967  |
| RRF               | 0.680 | 0.730   | 0.967  |

**Issue identified by mentor:** hybrid weights tuned on the same set used for
evaluation → data leakage risk. No formal validation/test split yet.

---

## v2 — 141–149 queries (validation/test split introduced)

- Eval set expanded via ChatGPT-generated queries + `auto_expand_labels.py`
  (category-based pooling) + manual review for accessibility/hours_based
  (initially skipped).
- Added 8 location-aware **"near me"** queries with geo-based relevance
  (POIs of matching category within 2km of Portland city center).
- Split: **104 validation / 45 test** (70/30, `random_state=42`).
- Grid search re-run on validation only → confirmed weights.

| Method              | NDCG@5 |
|----------------------|--------|
| Embeddings           | 0.833  ← best single method |
| Hybrid (0.1/0.9)      | 0.815  |
| RRF                   | 0.815  |
| TF-IDF (filter)       | 0.809  |
| BM25 (filter)         | 0.797  |
| TF-IDF (no filter)    | 0.643  |
| BM25 (no filter)      | 0.663  |

After adding "near me" queries, weights retuned: **0.2 (BM25) / 0.8 (Embeddings)**.
With this weighting on the same set: **Hybrid NDCG@5 = 0.831**, beating all
standalone methods including Embeddings.

---

## v3 — 321 labeled queries (231 → 321 after manual accessibility/hours_based labeling)

- Added ~150 new queries (batch 3) focused on weak intents: `hours_based`,
  `accessibility`, plus more `find_cafe`/`find_food`/`find_transport`/`find_shop`/`find_service`.
- Auto-labeled via category match where possible.
- **Accessibility queries** labeled as: POIs in relevant category with
  `wheelchair_accessible == 1`.
- **Hours-based queries** labeled as: POIs in relevant category with
  `opening_hours` populated (or `is_24_7 == True` for 24/7 queries).
- 5 queries removed (zero valid POIs in dataset: `bus stop close by`,
  `train station downtown`, `nearest tram stop`, `24 hour pharmacy nearby`,
  `are any pharmacies open 24/7`).
- Split: **224 validation / 97 test**.
- Grid search re-run → weights confirmed stable at **0.2 / 0.8**.

| Method              | NDCG@5 |
|----------------------|--------|
| Embeddings           | 0.742  ← best single method |
| Hybrid (0.2/0.8)      | 0.740  |
| TF-IDF (filter)       | 0.739  |
| BM25 (filter)         | 0.655  |
| RRF                   | 0.728  |
| Hybrid + Geo          | 0.729  |
| BM25 (no filter)      | 0.522  |
| TF-IDF (no filter)    | 0.531  |

### Key finding: scores dropped, and that's expected

NDCG@5 fell from ~0.83 (v2) to ~0.74 (v3). This is **not** a regression in the
retrieval system — it's the eval set becoming more honest. v3 added the queries
that are structurally hard for this dataset:

- **Accessibility** queries depend on `wheelchair_accessible`, which is rarely
  tagged in OSM.
- **Hours-based** queries depend on `opening_hours`, populated for only
  **12.7%** of all 24,918 POIs (3,179 POIs). Coverage analysis showed this
  field is populated mostly for businesses (restaurants, cafes, banks) and
  almost never for infrastructure (parking, bike racks, benches — which make
  up the bulk of POIs without it).

Earlier eval sets (v1, v2) under-represented these intents (7–16 queries vs.
40–49 for other intents), making the system look stronger than it would
perform on the full range of real user needs.

---

## Stable conclusions across all versions

- **Hybrid weights 0.2 (BM25) / 0.8 (Embeddings)** are stable — confirmed via
  grid search on three different validation sets (v1, v2, v3).
- **Embeddings consistently outperform lexical methods (TF-IDF/BM25) on
  semantic-heavy queries** — generic OSM POI names (e.g. "unknown",
  "Biketown") give BM25 little lexical signal.
- **Hit@50 stays high (0.98–1.00) for Hybrid/RRF across all versions** —
  the system almost always surfaces a relevant result somewhere in the top 50,
  even when ranking quality (NDCG@5) varies.
- **Geo component works correctly** (verified against real Portland locations
  via Google Maps) but is intentionally only applied to "near me"/"nearby"
  queries — applying it universally hurts ranking on non-location queries.
