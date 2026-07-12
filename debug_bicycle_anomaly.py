import pandas as pd
from src.utils.io import load_csv
from src.retrieval import pipeline
from src.retrieval.intent_classifier import load_model, predict

df = load_csv('data/processed/cleaned_pois.csv')
df = pipeline.run(df)
intent_model, intent_vectorizer = load_model('models/intent_classifier.pkl')

labels_df = pd.read_csv('data/relevance_labels_expanded.csv')
labels_df = labels_df[labels_df['relevant_poi_ids'].notna() & (labels_df['relevant_poi_ids'] != '')]

for _, row in labels_df.iterrows():
    query = row['query'].strip()
    intent, conf = predict(query, intent_model, intent_vectorizer)
    if intent != 'find_cafe':
        continue
    relevant_ids = set(int(x) for x in str(row['relevant_poi_ids']).split(',') if x.strip())
    if 'poi_id' in df.columns:
        relevant_pois = df[df['poi_id'].isin(relevant_ids)]
    else:
        relevant_pois = df.loc[df.index.isin(relevant_ids)]
    if (relevant_pois['category_final'] == 'bicycle_parking').any():
        print(f"Query: {query}")
        print(relevant_pois[relevant_pois["category_final"] == "bicycle_parking"][["name", "category_final"]].head())
        print()
