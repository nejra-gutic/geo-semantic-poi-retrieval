import pandas as pd
from src.utils.io import load_csv
from src.retrieval import pipeline
from src.retrieval.intent_classifier import load_model, predict
from src.retrieval.query import detect_specific_category, INTENT_TO_CATEGORY, has_category_signal

df = load_csv('data/processed/cleaned_pois.csv')
df = pipeline.run(df)
intent_model, intent_vectorizer = load_model('models/intent_classifier.pkl')

labels_df = pd.read_csv('data/relevance_labels_expanded.csv')
labels_df = labels_df[labels_df['relevant_poi_ids'].notna() & (labels_df['relevant_poi_ids'] != '')]

zero_hit_after_filter = 0
total = 0
skipped_no_signal = 0

for _, row in labels_df.iterrows():
    query = row['query'].strip()
    relevant_ids = set(int(x) for x in str(row['relevant_poi_ids']).split(',') if x.strip())

    intent, conf = predict(query, intent_model, intent_vectorizer)
    specific = detect_specific_category(query)

    if specific:
        categories = specific
    elif not has_category_signal(query):
        categories = None
        skipped_no_signal += 1
    else:
        categories = INTENT_TO_CATEGORY.get(intent)

    if categories:
        if 'poi_id' in df.columns:
            filtered_ids = set(df[df['category_final'].isin(categories)]['poi_id'].tolist())
        else:
            filtered_ids = set(df[df['category_final'].isin(categories)].index.tolist())
        overlap = relevant_ids & filtered_ids
        total += 1
        if len(overlap) == 0:
            zero_hit_after_filter += 1
            print(f"[ZERO OVERLAP] '{query}' -> intent={intent}, categories={categories}")
    else:
        # No filter applied -> search would cover all POIs, can't have zero overlap
        total += 1

print()
print(f"Queries sa 0 preklapanja nakon filtera: {zero_hit_after_filter} / {total}")
print(f"Queries skipped (no category signal): {skipped_no_signal}")