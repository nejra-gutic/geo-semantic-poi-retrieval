"""
retrieval/linguistic.py
-----------------------
Linguistic processing for POI text fields.
Applies POS tagging and NER using spaCy.

Extracts:
  - pos_tags: list of (token, POS tag) tuples
  - entities: list of (entity text, entity label) tuples
  - lemmas: list of lemmatized tokens (excl. stopwords and punctuation)
"""

import pandas as pd
import spacy

nlp = spacy.load("en_core_web_sm")


def extract_linguistic_features(text: str) -> dict:
    """
    Extract POS tags, named entities and lemmas from a single text.
    Returns dict with keys: pos_tags, entities, lemmas.
    """
    empty = {"pos_tags": [], "entities": [], "lemmas": []}

    if pd.isna(text) or str(text).strip().lower() in ("", "unknown"):
        return empty

    doc = nlp(str(text))

    pos_tags = [(token.text, token.pos_) for token in doc]
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    lemmas = [
        token.lemma_.lower()
        for token in doc
        if not token.is_stop and not token.is_punct and token.text.strip() != ""
    ]

    return {"pos_tags": pos_tags, "entities": entities, "lemmas": lemmas}


def add_linguistic_features(df: pd.DataFrame, col: str = "poi_text") -> pd.DataFrame:
    """
    Add pos_tags, entities and lemmas columns to the dataframe.
    Default source column is poi_text.
    """
    df = df.copy()

    if col not in df.columns:
        print(f"[linguistic] Warning - '{col}' not found, skipping")
        df["pos_tags"] = [[] for _ in range(len(df))]
        df["entities"] = [[] for _ in range(len(df))]
        df["lemmas"] = [[] for _ in range(len(df))]
        return df

    print(f"[linguistic] Processing {df[col].notna().sum()} rows...")
    features = df[col].apply(extract_linguistic_features)

    df["pos_tags"] = features.apply(lambda x: x["pos_tags"])
    df["entities"] = features.apply(lambda x: x["entities"])
    df["lemmas"]   = features.apply(lambda x: x["lemmas"])

    has_entities = df["entities"].apply(len).gt(0).sum()
    has_lemmas   = df["lemmas"].apply(len).gt(0).sum()
    print(f"[linguistic] Rows with entities: {has_entities}")
    print(f"[linguistic] Rows with lemmas:   {has_lemmas}")
    return df


def run(df: pd.DataFrame) -> pd.DataFrame:
    return add_linguistic_features(df)