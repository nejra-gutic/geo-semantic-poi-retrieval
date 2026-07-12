"""
failure_analysis.py
-------------------
Systematic analysis of where and why the intent classification pipeline
fails for each query in the eval set.

For each query, tracks failure at 3 levels:
  1. Intent classifier wrong (model predicted wrong class)
  2. Confidence too low (correct class but below threshold -> filter skipped)
  3. Category mapping gap (correct intent, but relevant POIs not in INTENT_TO_CATEGORY)

Usage:
    python3 failure_analysis.py
"""

import pandas as pd
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

    # Load ground truth intent labels from training data (for comparison)
    train_df = pd.read_csv('data/queries_annotated.csv')
    true_intent_map = dict(zip(train_df['query'].str.strip(), train_df['intent']))

    results = []

    for _, row in labels_df.iterrows():
        query = row['query'].strip()
        relevant_ids = set(
            int(x) for x in str(row['relevant_poi_ids']).split(',')
            if x.strip()
        )

        # Step 1: Predict intent
        intent, confidence = predict(query, intent_model, intent_vectorizer)
        specific = detect_specific_category(query)
        signal = has_category_signal(query)
        true_intent = true_intent_map.get(query, None)

        # Step 2: Determine what categories would be used
        if specific:
            categories = specific
            filter_type = "specific_override"
        elif not signal:
            categories = None
            filter_type = "no_signal_fallback"
        elif confidence >= CONFIDENCE_THRESHOLD:
            categories = INTENT_TO_CATEGORY.get(intent)
            filter_type = "intent_filter"
        else:
            categories = None
            filter_type = "low_confidence_fallback"

        # Step 3: Check overlap
        if categories:
            if 'poi_id' in df.columns:
                filtered_ids = set(
                    df[df['category_final'].isin(categories)]['poi_id'].tolist()
                )
            else:
                filtered_ids = set(
                    df[df['category_final'].isin(categories)].index.tolist()
                )
            overlap = relevant_ids & filtered_ids
            zero_overlap = len(overlap) == 0
        else:
            zero_overlap = False  # no filter = all POIs available
            filtered_ids = set()

        # Step 4: Diagnose failure type
        failure_type = None
        if zero_overlap:
            if specific:
                failure_type = "SPECIFIC_OVERRIDE_WRONG_CATEGORY"
            elif true_intent and true_intent != intent:
                failure_type = "CLASSIFIER_WRONG_INTENT"
            elif confidence < CONFIDENCE_THRESHOLD:
                failure_type = "LOW_CONFIDENCE"
            else:
                failure_type = "CATEGORY_MAPPING_GAP"

        results.append({
            "query": query,
            "predicted_intent": intent,
            "confidence": round(confidence, 1),
            "true_intent": true_intent,
            "filter_type": filter_type,
            "categories_used": categories,
            "zero_overlap": zero_overlap,
            "failure_type": failure_type,
        })

    results_df = pd.DataFrame(results)

    # Summary
    total = len(results_df)
    zero_overlap_df = results_df[results_df['zero_overlap'] == True]
    print(f"\n{'='*60}")
    print(f"FAILURE ANALYSIS SUMMARY")
    print(f"{'='*60}")
    print(f"Total queries: {total}")
    print(f"Zero overlap:  {len(zero_overlap_df)} ({len(zero_overlap_df)/total*100:.1f}%)")
    print(f"\nFailure breakdown:")
    if not zero_overlap_df.empty:
        print(zero_overlap_df['failure_type'].value_counts().to_string())

    print(f"\n{'='*60}")
    print(f"DETAILED FAILURES")
    print(f"{'='*60}")
    for _, row in zero_overlap_df.iterrows():
        print(f"\n[{row['failure_type']}]")
        print(f"  Query:     '{row['query']}'")
        print(f"  Predicted: {row['predicted_intent']} ({row['confidence']}%)")
        print(f"  True:      {row['true_intent'] or 'NOT IN TRAINING SET'}")
        print(f"  Filter:    {row['filter_type']}")
        print(f"  Categories: {row['categories_used']}")

    # Additional stats for non-failure queries
    print(f"\n{'='*60}")
    print(f"FILTER TYPE DISTRIBUTION (all queries)")
    print(f"{'='*60}")
    print(results_df['filter_type'].value_counts().to_string())

    print(f"\n{'='*60}")
    print(f"INTENT DISTRIBUTION (predicted)")
    print(f"{'='*60}")
    print(results_df['predicted_intent'].value_counts().to_string())

    # Save full results
    results_df.to_csv('data/failure_analysis.csv', index=False)
    print(f"\n[failure_analysis] Full results saved: data/failure_analysis.csv")


if __name__ == "__main__":
    run()
