"""
retrieval/pipeline.py
---------------------
Orchestrates the full NLP retrieval preprocessing pipeline.
Applies tokenization, normalization and linguistic processing
to the cleaned POI dataset.

Steps:
  1. normalize  - lowercase, stopwords, lemmatization
  2. tokenize   - spaCy tokenization
  3. linguistic - POS tagging, NER, lemmatization → poi_text_lemma
  4. augment    - synonym augmentation on poi_text_lemma
"""

import pandas as pd
from src.retrieval import tokenize, normalize, linguistic
from src.preprocessing.text_join import CATEGORY_SYNONYMS


def run(df: pd.DataFrame) -> pd.DataFrame:
    print("=" * 50)
    print("Starting retrieval preprocessing pipeline")
    print(f"Input shape: {df.shape}")
    print("=" * 50)

    print("\n--- Step 1: Normalize ---")
    df = normalize.run(df)

    print("\n--- Step 2: Tokenize ---")
    df = tokenize.run(df)

    print("\n--- Step 3: Linguistic ---")
    df = linguistic.run(df)

    print("\n--- Step 4: Synonym Augmentation ---")
    if "poi_text_lemma" in df.columns and "category_final" in df.columns:
        def augment_lemma(row):
            cat = str(row.get("category_final", ""))
            extra = CATEGORY_SYNONYMS.get(cat, "")
            if extra:
                return str(row["poi_text_lemma"]) + " " + extra
            return str(row["poi_text_lemma"])

        df["poi_text_lemma"] = df.apply(augment_lemma, axis=1)
        augmented = df["category_final"].isin(CATEGORY_SYNONYMS.keys()).sum()
        print(f"[retrieval.pipeline] Synonym augmentation applied to {augmented} POIs")

    print("\n" + "=" * 50)
    print(f"Pipeline complete. Output shape: {df.shape}")
    print("=" * 50)
    return df