"""
retrieval/pipeline.py
---------------------
Orchestrates the full NLP retrieval preprocessing pipeline.
Applies tokenization, normalization and linguistic processing
to the cleaned POI dataset.

Steps:
   1. normalize - lowercase, stopwords, lemmatization
   2. tokenize  - spaCy tokenization
   3. linguistic - POS tagging, NER
"""

import pandas as pd
from src.retrieval import tokenize, normalize, linguistic


def run(df: pd.DataFrame) -> pd.DataFrame:
    """
    Run the full retrieval preprocessing pipeline.
    Input: cleaned POI dataframe (output of src.preprocessing.pipeline)
    Output: dataframe enriched with NLP columns
    """
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

    print("\n" + "=" * 50)
    print(f"Pipeline complete. Output shape: {df.shape}")
    print("=" * 50)
    return df