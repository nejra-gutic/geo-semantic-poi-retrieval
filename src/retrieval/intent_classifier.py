"""
retrieval/intent_classifier.py
-------------------------------
Intent classification for user queries.

Trains a Logistic Regression classifier on TF-IDF features
to predict user intent from natural language queries.

Usage:
    python3 -m src.retrieval.intent_classifier \
        --data data/queries_annotated.csv \
        --save models/intent_classifier.pkl
"""

import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import cross_val_score, StratifiedKFold, train_test_split
from sklearn.metrics import classification_report, confusion_matrix

SYNONYMS = {
    # cafe
    "coffee shop": "cafe",
    "espresso": "cafe",
    "latte": "cafe",
    "cappuccino": "cafe",
    "tea house": "cafe",
    "bakery": "cafe",
    "brunch spot": "cafe",

    # food
    "pizza": "restaurant",
    "burger": "restaurant",
    "sushi": "restaurant",
    "tacos": "restaurant",
    "noodles": "restaurant",
    "bbq": "restaurant",
    "fast food": "restaurant",

    # service
    "doctor": "medical",
    "dentist": "medical",
    "pharmacy": "medical",
    "clinic": "medical",
    "urgent care": "medical",
    "veterinarian": "medical",
    "laundromat": "service",
    "post office": "service",
    "atm": "service",
    "bank": "service",

    # shop
    "bookstore": "shop",
    "grocery": "shop",
    "supermarket": "shop",
    "hardware store": "shop",
    "clothing store": "shop",
    "shoe store": "shop",
    "electronics store": "shop",
    "furniture store": "shop",
    "pet store": "shop",
    "gift shop": "shop",

    # transport
    "taxi": "transport",
    "bus stop": "transport",
    "bicycle parking": "transport",
    "bike rack": "transport",
    "car park": "transport",
    "parking garage": "transport",
    "parking lot": "transport",
    "motorcycle parking": "transport",

    # accessibility
    "disability friendly": "accessible",
    "disabled access": "accessible",
    "wheelchair friendly": "accessible",
    "accessible entrance": "accessible",
}

def normalize_query(q: str) -> str:
    q = q.lower()
    for k, v in SYNONYMS.items():
        q = q.replace(k, v)
    return q


def load_data(path: str) -> pd.DataFrame:
    """Load annotated query dataset."""
    df = pd.read_csv(path)
    print(f"[intent] Loaded {len(df)} queries, {df['intent'].nunique()} intents")
    print(f"[intent] Distribution:\n{df['intent'].value_counts().to_string()}")
    return df


def build_features(df: pd.DataFrame) -> tuple:
    """Vectorize queries using TF-IDF."""

    df = df.copy()
    df["query"] = df["query"].apply(normalize_query)

    vectorizer = TfidfVectorizer(ngram_range=(1, 2))
    X = vectorizer.fit_transform(df["query"])
    y = df["intent"]

    print(f"[intent] Feature matrix: {X.shape}")
    return vectorizer, X, y


def cross_validate(X, y) -> None:
    """Compare Logistic Regression vs Naive Bayes via cross-validation."""
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    lr = LogisticRegression(max_iter=1000, random_state=42)
    nb = MultinomialNB()

    lr_scores = cross_val_score(lr, X, y, cv=cv, scoring="accuracy")
    nb_scores = cross_val_score(nb, X, y, cv=cv, scoring="accuracy")

    print("\n[intent] Cross-validation results:")
    print(f"  Logistic Regression: {lr_scores.mean():.3f} (+/- {lr_scores.std():.3f})")
    print(f"  Naive Bayes:         {nb_scores.mean():.3f} (+/- {nb_scores.std():.3f})")


def train(X, y) -> LogisticRegression:
    """Train final Logistic Regression model."""
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X, y)

    print(f"[intent] Model trained on {X.shape[0]} samples")
    return model


def evaluate(model, X, y, queries=None) -> None:
    """Print classification report and plot confusion matrix."""
    y_pred = model.predict(X)

    print("\n[intent] Classification Report:")
    print(classification_report(y, y_pred, zero_division=0))

    if queries is not None:
        print("\n[intent] Misclassified examples:")
        for q, true, pred in zip(queries, y, y_pred):
            if true != pred:
                print(f"  '{q}' → true: {true} | pred: {pred}")

    labels = model.classes_
    cm = confusion_matrix(y, y_pred, labels=labels)

    Path("models").mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        xticklabels=labels,
        yticklabels=labels,
        cmap="Blues",
    )
    plt.title("Confusion Matrix - Logistic Regression")
    plt.ylabel("True")
    plt.xlabel("Predicted")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("models/confusion_matrix.png")
    plt.show()

    print("[intent] Confusion matrix saved: models/confusion_matrix.png")


def top_features(model, vectorizer, n: int = 10) -> None:
    """Print top TF-IDF features per intent class."""
    feature_names = vectorizer.get_feature_names_out()

    print("\n[intent] Top features per class:")

    for intent in model.classes_:
        idx = list(model.classes_).index(intent)
        top_indices = np.argsort(model.coef_[idx])[::-1][:n]
        top = [(feature_names[i], round(model.coef_[idx][i], 3)) for i in top_indices]

        print(f"\n  {intent}:")
        for word, score in top:
            print(f"    {word}: {score}")


def predict(query: str, model, vectorizer) -> tuple[str, float]:
    """
    Predict intent for a single query.
    Returns (intent, confidence).
    """
    query_norm = normalize_query(query)
    X = vectorizer.transform([query_norm])

    intent = model.predict(X)[0]
    probs = model.predict_proba(X)[0]
    confidence = round(max(probs) * 100, 1)

    return intent, confidence


def save_model(model, vectorizer, path: str = "models/intent_classifier.pkl") -> None:
    """Save model and vectorizer to disk."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "wb") as f:
        pickle.dump({"model": model, "vectorizer": vectorizer}, f)

    print(f"[intent] Model saved: {path}")


def load_model(path: str = "models/intent_classifier.pkl") -> tuple:
    """Load model and vectorizer from disk."""
    with open(path, "rb") as f:
        obj = pickle.load(f)

    print(f"[intent] Model loaded: {path}")
    return obj["model"], obj["vectorizer"]


def main():
    parser = argparse.ArgumentParser(description="Train intent classifier.")
    parser.add_argument("--data", type=str, default="data/queries_annotated.csv")
    parser.add_argument("--save", type=str, default="models/intent_classifier.pkl")
    args = parser.parse_args()

    df = load_data(args.data)

    vectorizer, X, y = build_features(df)

    # Realistic evaluation using cross-validation
    cross_validate(X, y)

    # Train/test split for classification report + confusion matrix
    X_train, X_test, y_train, y_test, q_train, q_test = train_test_split(
        X,
        y,
        df["query"],
        test_size=0.2,
        stratify=y,
        random_state=42,
    )

    model = train(X_train, y_train)

    print("\n[intent] Evaluation on held-out test set:")
    evaluate(model, X_test, y_test, q_test)

    top_features(model, vectorizer)

    # Save trained model
    save_model(model, vectorizer, args.save)

    # Manual test queries
    test_queries = [
        "coffee near burnside",
        "wheelchair accessible pharmacy",
        "24/7 burger place",
        "parking near division street",
        "italian restaurant takeaway",
    ]

    print("\n[intent] Test predictions:")
    for q in test_queries:
        intent, confidence = predict(q, model, vectorizer)
        print(f"  '{q}' → {intent} ({confidence}%)")


if __name__ == "__main__":
    main()