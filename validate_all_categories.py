import pandas as pd
from src.utils.io import load_csv
from src.retrieval import pipeline
from src.retrieval.intent_classifier import load_model, predict
from collections import Counter

df = load_csv('data/processed/cleaned_pois.csv')
df = pipeline.run(df)
intent_model, intent_vectorizer = load_model('models/intent_classifier.pkl')

labels_df = pd.read_csv('data/relevance_labels_expanded.csv')
labels_df = labels_df[labels_df['relevant_poi_ids'].notna() & (labels_df['relevant_poi_ids'] != '')]

intent_category_counter = {}

for _, row in labels_df.iterrows():
    query = row['query'].strip()
    relevant_ids = set(int(x) for x in str(row['relevant_poi_ids']).split(',') if x.strip())

    intent, conf = predict(query, intent_model, intent_vectorizer)

    if 'poi_id' in df.columns:
        relevant_pois = df[df['poi_id'].isin(relevant_ids)]
    else:
        relevant_pois = df.loc[df.index.isin(relevant_ids)]

    cats = relevant_pois['category_final'].value_counts()

    if intent not in intent_category_counter:
        intent_category_counter[intent] = Counter()

    for cat, count in cats.items():
        intent_category_counter[intent][cat] += count

for intent in sorted(intent_category_counter.keys()):
    print(f"\n=== {intent} ===")
    total = sum(intent_category_counter[intent].values())
    for cat, count in intent_category_counter[intent].most_common(15):
        pct = count / total * 100
        print(f"  {cat}: {count} ({pct:.1f}%)")
