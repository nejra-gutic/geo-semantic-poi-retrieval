# Geo Semantic Retrieval Pipeline for OSM POIs

This project focuses on building a geo-semantic retrieval pipeline for OpenStreetMap (OSM) Points of Interest data.

The goal is to map a natural language user query to a ranked list of relevant POIs by combining textual, semantic, rule-based, and geospatial logic.

## Project Idea

Input:

```text
user query
```

Example:

```text
pizza places open now with wheelchair access
```

Output:

```text
ranked list of relevant POIs
```

The system is designed to combine several retrieval components:

- text matching between user queries and POI textual attributes
- rule-based filtering for structured fields such as opening hours and wheelchair access
- semantic matching using embeddings
- geospatial logic based on POI locations and geometry representations

## Current Focus

The current stage of the project focuses on preparing clean and structured OSM data that can later be used for retrieval.

Main preprocessing goals:

- preserve original OSM fields
- create normalized versions of important attributes
- standardize POI categories
- normalize textual attributes for search
- parse and structure fields such as cuisine, wheelchair access, and opening hours
- create joined text fields for comparison with user input
- prepare geometry representations for spatial logic
- save processed data in both CSV and GeoJSON formats

## Current Work

So far, the project includes:

- OSM dataset exploration
- missing value analysis
- category inspection and standardization
- wheelchair attribute normalization
- opening hours analysis and basic normalization
- text preprocessing for NLP tasks
- TF-IDF vectorization
- n-gram experiments
- sparsity analysis of TF-IDF matrices

## Planned Pipeline

```text
Raw OSM data
     ↓
Column-specific preprocessing
     ↓
Structured and normalized POI dataset
     ↓
Query preprocessing
     ↓
Text matching + rule-based filters + semantic matching + geo logic
     ↓
Ranked POI results
```

## Project Structure

```text
osm-poi-nlp-search/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── notebooks/
│
├── src/
│   ├── preprocessing/
│   ├── retrieval/
│   ├── geo/
│   └── utils/
│
├── reports/
│
├── README.md
├── requirements.txt
└── .gitignore
```

## Technologies

Current and planned tools:

- Python
- Pandas
- GeoPandas
- Shapely
- Scikit-learn
- TF-IDF
- NLP preprocessing
- embeddings
- Google Colab

Potential libraries for later stages:

- unidecode
- opening-hours parsing libraries
- address parsing libraries
- Hugging Face embeddings
- vector search libraries

## Status

Project is currently in active development as part of the AtlantBH Internship Program.
