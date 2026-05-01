# Geo-Semantic POI Retrieval

A pipeline for mapping natural language user queries to relevant Points of Interest (POIs) using a combination of text matching, semantic similarity, rule-based filtering, and geospatial logic.

## Project Goal

**Input:** Natural language user query (e.g. "wheelchair accessible Italian restaurant open on weekends")  
**Output:** Ranked list of matching POIs from an OSM dataset

## Approach

- **Rule-based filtering** — structured flags (wheelchair, opening hours, takeaway, etc.)
- **Text matching** — TF-IDF over joined POI text fields
- **Semantic similarity** — embeddings for deeper query understanding (later stage)
- **Geo component** — spatial proximity and layout (later stage)

## Project Structure

```
geo-semantic-poi-retrieval/
├── src/
│   ├── preprocessing/
│   │   ├── clean.py          # null handling, unknown values
│   │   ├── normalize.py      # unidecode, lowercasing, regex
│   │   ├── geometry.py       # centroid extraction, lat/lon
│   │   ├── address.py        # address parsing (libpostal/usaddress)
│   │   ├── opening_hours.py  # hours parsing + boolean flags
│   │   ├── cuisine.py        # cuisine tag normalization
│   │   ├── flags.py          # wheelchair, takeaway → booleans
│   │   ├── text_join.py      # joins text fields → poi_text
│   │   └── pipeline.py       # runs the full preprocessing pipeline
│   ├── retrieval/            # TF-IDF, embeddings (later)
│   └── utils/
│       └── io.py             # load/save CSV and GeoJSON
├── data/
│   ├── raw/                  # original OSM exports (.gitignored)
│   ├── processed/            # cleaned outputs
│   └── samples/              # small samples for testing (committed)
├── notebooks/                # exploration notebooks (numbered)
├── outputs/
├── tests/
├── requirements.txt
└── .gitignore
```

## Setup

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

## Running the Pipeline

```python
from src.preprocessing.pipeline import run_pipeline
import pandas as pd

df = pd.read_csv("data/raw/pois_portland.csv")
df_processed = run_pipeline(df)
df_processed.to_csv("data/processed/pois_processed.csv", index=False)
```

## Data Source

OpenStreetMap data for Portland, Oregon, USA — fetched via `osmnx`.

## Status

- [x] Data acquisition & EDA
- [x] Cleaning & normalization
- [x] Tokenization & linguistic processing
- [x] TF-IDF + n-gram analysis
- [ ] Preprocessing pipeline refactor (in progress)
- [ ] Embedding-based retrieval
- [ ] Geo component
- [ ] Query interface
