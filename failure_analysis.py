"""
failure_analysis.py
-------------------
Analysis of intent classification quality and soft boost effectiveness
for the geo-semantic POI retrieval pipeline.

Since switching to soft boost (no hard filtering), zero-overlap is
structurally impossible. This script instead measures:

  1. Intent prediction accuracy (where ground truth is available)
  2. Confidence distribution across eval queries
  3. Boost effectiveness — do boosted POIs rank in top-K?
  4. Filter type distribution (specific_override, intent_boost, no_signal, low_confidence)

Usage:
    python3 failure_analysis.py
"""

import pandas as pd
import numpy as np
from src.utils.io import load_csv
from src.retrieval import pipeline
from src.retrieval.intent_classifier import load_model, predict
from src.retrieval.query import (
    detect_specific_category,
    has_category_signal,
    INTENT_TO_CATEGORY,
)

CONFIDENCE_THRESHOLD = 40.0


def run():
    # Load data
    df = load_csv('data/processed/cleaned_pois.csv')
    df = pipeline.run(df)
    intent_model, intent_vectorizer = load_model('models/intent_classifier.pkl')

    labels_df = pd.read_csv('data/relevance_labels_expanded.csv')
    labels_df = labels_df[
        labels_df['relevant_poi_ids'].notna() &
        (labels_df['relevant_poi_ids'] != '')
    ]

    # Load ground truth intent labels from training data
    train_df = pd.read_csv('data/queries_annotated.csv')
    true_intent_map = dict(zip(train_df['query'].str.strip(), train_df['intent']))

    results = []

    for _, row in labels_df.iterrows():
        query = row['query'].strip()
        relevant_ids = set(
            int(x) for x in str(row['relevant_poi_ids']).split(',')
            if x.strip()
        )

        # Predict intent
        intent, confidence = predict(query, intent_model, intent_vectorizer)
        specific = detect_specific_category(query)
        signal = has_category_signal(query)
        true_intent = true_intent_map.get(query, None)

        # Determine filter type
        if specific:
            filter_type = "specific_override"
            categories = specific
        elif not signal:
            filter_type = "no_signal_fallback"
            categories = None
        elif confidence >= CONFIDENCE_THRESHOLD:
            filter_type = "intent_boost"
            categories = INTENT_TO_CATEGORY.get(intent)
        else:
            filter_type = "low_confidence_fallback"
            categories = None

        # Check intent correctness (only for queries in training set)
        intent_correct = None
        if true_intent:
            intent_correct = (intent == true_intent)

        # Check boost effectiveness — are relevant POIs in boosted category?
        if categories and 'poi_id' in df.columns:
            boosted_ids = set(
                df[df['category_final'].isin(categories)]['poi_id'].tolist()
            )
        elif categories:
            boosted_ids = set(
                df[df['category_final'].isin(categories)].index.tolist()
            )
        else:
            boosted_ids = set()

        relevant_in_boost = len(relevant_ids & boosted_ids)
        relevant_total = len(relevant_ids)
        boost_coverage = relevant_in_boost / relevant_total if relevant_total > 0 else None

        results.append({
            "query": query,
            "predicted_intent": intent,
            "confidence": round(confidence, 1),
            "true_intent": true_intent,
            "intent_correct": intent_correct,
            "filter_type": filter_type,
            "categories": categories,
            "relevant_total": relevant_total,
            "relevant_in_boost": relevant_in_boost,
            "boost_coverage": round(boost_coverage, 3) if boost_coverage is not None else None,
        })

    results_df = pd.DataFrame(results)
    total = len(results_df)

    print(f"\n{'='*60}")
    print(f"FAILURE ANALYSIS — SOFT BOOST MODE")
    print(f"{'='*60}")
    print(f"Total eval queries: {total}")

    # 1. Intent accuracy
    known = results_df[results_df['intent_correct'].notna()]
    if len(known) > 0:
        accuracy = known['intent_correct'].mean() * 100
        print(f"\n--- Intent Accuracy (queries in training set: {len(known)}) ---")
        print(f"Correct: {known['intent_correct'].sum()} / {len(known)} ({accuracy:.1f}%)")
        wrong = known[known['intent_correct'] == False]
        if not wrong.empty:
            print(f"\nMisclassified ({len(wrong)}):")
            for _, r in wrong.iterrows():
                print(f"  '{r['query']}'")
                print(f"    predicted: {r['predicted_intent']} ({r['confidence']}%) | true: {r['true_intent']}")

    # 2. Confidence distribution
    print(f"\n--- Confidence Distribution ---")
    bins = [0, 40, 60, 80, 100]
    labels = ["<40% (low)", "40-60%", "60-80%", ">80%"]
    results_df['conf_bin'] = pd.cut(results_df['confidence'], bins=bins, labels=labels)
    print(results_df['conf_bin'].value_counts().sort_index().to_string())

    # 3. Filter type distribution
    print(f"\n--- Filter Type Distribution ---")
    print(results_df['filter_type'].value_counts().to_string())

    # 4. Boost coverage
    boosted = results_df[results_df['filter_type'] == 'intent_boost']
    if not boosted.empty:
        print(f"\n--- Boost Coverage (intent_boost queries: {len(boosted)}) ---")
        print(f"Mean coverage: {boosted['boost_coverage'].mean()*100:.1f}%")
        print(f"Zero coverage (relevant POIs NOT in boosted category): {(boosted['boost_coverage'] == 0).sum()}")
        zero_cov = boosted[boosted['boost_coverage'] == 0]
        if not zero_cov.empty:
            print("\nQueries where boost misses all relevant POIs:")
            for _, r in zero_cov.iterrows():
                print(f"  '{r['query']}' -> predicted: {r['predicted_intent']} | categories: {r['categories']}")

    # 5. Intent distribution
    print(f"\n--- Predicted Intent Distribution ---")
    print(results_df['predicted_intent'].value_counts().to_string())

    # Save
    results_df.to_csv('data/failure_analysis.csv', index=False)
    print(f"\n[failure_analysis] Saved: data/failure_analysis.csv")


if __name__ == "__main__":
    run()