# 🌍 Geo-Semantic POI Retrieval

An NLP-based search and ranking system that maps natural-language queries to relevant **Points of Interest (POIs)** using lexical retrieval, semantic embeddings, intent classification, temporal reasoning, and geographic re-ranking.

The system is built on **OpenStreetMap data for Portland, Oregon** and is designed to handle realistic queries such as:

> "24/7 wheelchair accessible pharmacy near me"

> "Italian restaurant open tonight"

> "coffee shop nearby"

Instead of relying only on keyword matching, the pipeline combines multiple signals to understand **what the user wants, where they want it, and when they need it**.

---

## 🎯 Project Goal

Traditional keyword search can struggle with natural-language queries, especially when the query contains semantic, temporal, accessibility, or geographic information.

The goal of this project was to build a retrieval system that can transform:

**Input**

```text
"wheelchair accessible Italian restaurant open tonight near me"
```

into:

**Output**

```text
A ranked list of relevant POIs
```

while considering:

- textual relevance
- semantic similarity
- predicted query intent
- accessibility and other structured attributes
- opening hours
- geographic distance

---

## 🧠 System Overview

The final retrieval pipeline follows this general flow:

```text
                     User Query
                         │
                         ▼
                 Query Processing
                         │
          ┌──────────────┴──────────────┐
          ▼                             ▼
   Retrieval Methods              Intent Classifier
 TF-IDF / BM25 / Embeddings             │
          │                              ▼
          └──────────────► Intent Soft Boost
                         │
                         ▼
                  Boolean Filters
                         │
                         ▼
                  Geo Re-ranking
                         │
                         ▼
                Temporal Re-ranking
                         │
                         ▼
                   Final Ranking
                         │
                         ▼
                    Top-k POIs
```

The pipeline keeps the retrieval methods under a common processing flow so that they can be evaluated and compared consistently.

---

## 🔎 Retrieval Methods

### TF-IDF

TF-IDF provides a lexical baseline by representing queries and POI descriptions as sparse vectors.

Cosine similarity is used to rank POIs according to keyword-based similarity.

This works well when the query and POI contain similar words, but it has limited ability to understand semantic relationships.

### BM25

BM25 provides a stronger lexical ranking method.

Compared with basic TF-IDF retrieval, it handles term frequency and document length more carefully and is useful for queries containing important exact terms such as:

```text
"Starbucks Burnside"
```

### Semantic Embeddings

For semantic retrieval, POI descriptions and queries are represented using **Sentence Transformer embeddings**.

Model:

```text
sentence-transformers/all-MiniLM-L6-v2
```

Each query and POI is represented as a **384-dimensional embedding**.

Cosine similarity is then used to retrieve POIs that are semantically similar even when the exact words are different.

For example:

```text
Query: "cheap place to eat"
POI:   "affordable restaurant"
```

Lexical overlap is limited, but their semantic meaning is similar.

### Hybrid Retrieval

Lexical and semantic retrieval provide complementary information:

- **BM25** is strong for exact terms and keyword matching.
- **Embeddings** are strong for semantic similarity.

The project therefore experiments with hybrid ranking that combines lexical and semantic scores.

This allows the system to benefit from both exact matching and deeper semantic understanding.

---

## 🎯 Intent Classification

A lightweight intent classifier is used to understand the main purpose of a query.

Example intents include:

```text
find_cafe
find_food
find_shop
find_service
find_transport
accessibility
```

Two classification approaches were compared:

- Logistic Regression
- Naive Bayes

Logistic Regression performed better and was selected for the final classifier. The final classifier reached approximately **85% accuracy** in the later evaluation stage.

### Hard Filter → Soft Boost

An important improvement was replacing strict intent-based filtering with **soft boosting**.

Instead of:

```text
Predicted intent = cafe
→ remove every non-cafe result
```

the system uses:

```text
Predicted intent = cafe
→ boost cafe results
→ keep other potentially relevant results
```

This prevents useful POIs from disappearing when the intent classifier makes an imperfect prediction.

---

## 📍 Geographic Re-ranking

Semantic relevance alone is not enough for location-based search.

A highly relevant POI may be several kilometers away while another relevant result is nearby.

The system computes geographic distance using the **Haversine formula** and uses a distance-decay component during ranking.

Conceptually:

```text
Retrieval relevance
        +
Geographic relevance
        ↓
Final ranking
```

This makes it possible to prioritize POIs that are both **relevant and nearby**, rather than simply returning the closest location.

---

## 🕒 Temporal Search

The pipeline also supports time-aware queries such as:

```text
"restaurant open now"
"coffee shop open tonight"
"pharmacy open Saturday"
```

Temporal expressions are detected from the query and resolved to a target time.

POI opening-hours information is then used during ranking. Opening-hours parsing is handled with `opening-hours-py`.

Because OpenStreetMap metadata can be incomplete, the system distinguishes between cases where a POI is known to be open, known to be closed, or its status is unknown.

---

## 🧹 Data Processing

The project works with approximately **25,000 POIs** from Portland, Oregon.

The preprocessing pipeline reduces noisy OpenStreetMap data into structured and searchable POI representations.

Main preprocessing steps include:

- missing-value handling
- category normalization
- address processing
- cuisine normalization
- accessibility and takeaway flags
- opening-hours processing
- text normalization
- tokenization / linguistic processing
- latitude and longitude extraction
- creation of searchable `poi_text`

The resulting text representation is shared across the retrieval methods.

---

## 📊 Evaluation

A major part of the project was building a reliable evaluation framework.

The evaluation set grew from a small initial set to **300+ queries**, covering different query types and retrieval scenarios.

Retrieval quality was evaluated using metrics such as:

### Precision@5

Measures how many of the first five retrieved POIs are relevant.

### NDCG@5

Measures retrieval quality while also rewarding systems that place the most relevant results near the top of the ranking.

Evaluation was used throughout development to compare TF-IDF, BM25, embeddings, hybrid retrieval, intent boosting, and geographic ranking.

Embedding-based retrieval was particularly strong on semantic queries, while hybrid approaches allowed lexical and semantic signals to be combined.

---

## 🔬 Failure Analysis

The project included explicit failure analysis rather than relying only on aggregate metrics.

Three recurring issues were especially important:

### 1. Ground-truth quality

Some evaluation labels did not perfectly represent the best real-world result, showing that evaluation quality is as important as model quality.

### 2. Missing OpenStreetMap metadata

Important fields such as opening hours, accessibility, and takeaway information can be incomplete.

A retrieval system cannot reliably use information that is missing from the underlying data.

### 3. Intent classification errors

Ambiguous queries can be assigned to the wrong intent.

This was one of the reasons for moving from hard filtering to soft boosting.

---

## 🖥️ Interactive Demo

A **Streamlit application** provides an interactive interface for testing the retrieval system.

The application allows users to:

- enter natural-language POI queries
- select a retrieval method
- provide a location
- view ranked POI results
- inspect distance and opening status
- visualize retrieved POIs on a map

Run the application with:

```bash
streamlit run app.py
```

---

## 🗂️ Project Structure

```text
geo-semantic-poi-retrieval/
│
├── src/
│   ├── preprocessing/      # cleaning and POI feature preparation
│   ├── retrieval/          # TF-IDF, BM25, embeddings and hybrid retrieval
│   ├── intent/             # query intent classification
│   └── utils/              # shared utilities
│
├── data/
│   ├── raw/                # raw OSM data
│   ├── processed/          # processed POI datasets
│   └── samples/            # smaller development samples
│
├── notebooks/              # experiments and exploratory analysis
├── reports/                # evaluation and analysis outputs
│
├── app.py                  # Streamlit demo
├── eval.py                 # retrieval evaluation
├── failure_analysis.py     # failure analysis
├── tune_bm25_run.py        # BM25 tuning
├── tune_hybrid_weights.py  # hybrid weight tuning
├── RESULTS.md              # experiment results
├── requirements.txt
└── README.md
```

---

## ⚙️ Setup

Clone the repository:

```bash
git clone https://github.com/nejra-gutic/geo-semantic-poi-retrieval.git
cd geo-semantic-poi-retrieval
```

Install dependencies:

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

Run the Streamlit application:

```bash
streamlit run app.py
```

---

## 🗺️ Data Source

The project uses **OpenStreetMap** POI data for **Portland, Oregon, USA**.

The data was collected using `osmnx` and processed into a smaller set of structured features used by the retrieval pipeline.

---

## 🛠️ Technologies

- Python
- pandas / NumPy
- scikit-learn
- Sentence Transformers
- TF-IDF
- BM25
- spaCy
- OpenStreetMap / OSMnx
- Haversine distance
- Streamlit

---

## 💡 Key Lessons

This project showed that building a useful search system is not only about choosing the strongest model.

**Data quality matters.**  
Missing metadata directly limits what the retrieval system can understand.

**Evaluation quality matters.**  
Better ground-truth labels lead to more reliable conclusions about model performance.

**Ranking signals need to be balanced.**  
Lexical relevance, semantic similarity, intent, time, and geographic distance can all help, but combining them correctly is as important as the individual methods.

---

## 🚀 Future Improvements

Possible next steps include:

- adaptive BM25/embedding weighting based on query type
- improved address and street-name search
- automatic user-location detection
- more human-validated relevance labels
- evaluation across multiple cities
- fine-tuning the embedding model on POI-specific query-result pairs
- vector indexing with tools such as FAISS for larger-scale retrieval

---

## 📌 Project Context

This project was developed as an exploration of **information retrieval, NLP, semantic search, machine learning, geospatial ranking, and evaluation** on real-world OpenStreetMap data.
