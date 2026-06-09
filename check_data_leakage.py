import pandas as pd

# Query-ji korišteni za treniranje intent classifiera
train_queries = set(
    pd.read_csv("data/queries_annotated.csv")["query"].str.strip().str.lower()
)

# Query-ji korišteni za evaluaciju retrievala
eval_queries = {
    "where can i get espresso",
    "tacos near me",
    "find me a dentist",
    "need a cashpoint",
    "emergency room nearby",
    "where to park my car",
    "place to lock my bike",
    "gluten free restaurant",
    "place for a haircut",
    "grab a coffee and work",
}


overlap = train_queries & eval_queries

print(f"Training queries:   {len(train_queries)}")
print(f"Evaluation queries: {len(eval_queries)}")
print(f"Overlap (leakage):  {len(overlap)}")

if overlap:
    print("\nLeaking queries:")
    for q in sorted(overlap):
        print(f"  - {q}")
else:
    print("\nNema data leakage!")