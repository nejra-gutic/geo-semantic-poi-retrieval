import pandas as pd
from src.utils.io import load_csv
from src.retrieval import pipeline
from src.retrieval.intent_classifier import load_model, predict

df = load_csv('data/processed/cleaned_pois.csv')
df = pipeline.run(df)
intent_model, intent_vectorizer = load_model('models/intent_classifier.pkl')

labels_df = pd.read_csv('data/relevance_labels_expanded.csv')
labels_df = labels_df[labels_df['relevant_poi_ids'].notna() & (labels_df['relevant_poi_ids'] != '')]

from collections import Counter
category_counter = Counter()

for _, row in labels_df.iterrows():
    query = row['query'].strip()
    relevant_ids = set(int(x) for x in str(row['relevant_poi_ids']).split(',') if x.strip())

    intent, conf = predict(query, intent_model, intent_vectorizer)
    if intent != "find_cafe":
        continue

    relevant_pois = df[df['poi_id'].isin(relevant_ids)] if 'poi_id' in df.columns else df.loc[df.index.isin(relevant_ids)]
    cats = relevant_pois['category_final'].value_counts()
    for cat, count in cats.items():
        category_counter[cat] += count

print("Stvarne kategorije POI-a koji su relevantni za find_cafe queries:")
for cat, count in category_counter.most_common(15):
    print(f"  {cat}: {count}")
