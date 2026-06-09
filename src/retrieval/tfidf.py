"""
retrieval/tfidf.py
------------------
TF-IDF and n-gram representation of POI text fields.
Builds TF-IDF matrix for semantic text matching.

Uses scikit-learn TfidfVectorizer.
TF-IDF preferred over BOW because it downweights common terms
and highlights distinctive terms per POI.
"""

import pandas as pd
import pickle
import numpy as np
import re
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from src.retrieval.normalize import normalize
from src.preprocessing.normalize import normalize_text as preprocess_text


def _tfidf_tokenizer(text: str) -> list[str]:
    tokens = [t for t in text.lower().split() if len(t) > 1 and not re.match(r'^\d+\w*$', t)]
    bigrams = [f"{tokens[i]}_{tokens[i+1]}" for i in range(len(tokens) - 1)]
    return tokens + bigrams

def debug_tokenization(query: str, df: pd.DataFrame, row_idx: int = 0) -> None:
    """
    Spot-check tokenization for one query and one POI document.
    Useful for comparing query tokens vs document tokens.
    """
    query_norm = normalize(query) or query
    query_tokens = _tfidf_tokenizer(query_norm)

    poi_raw = df.iloc[row_idx].get("poi_text", "")
    poi_lemma = df.iloc[row_idx].get("poi_text_lemma", "")
    poi_tokens = _tfidf_tokenizer(str(poi_lemma))

    print("\n=== TF-IDF TOKENIZATION DEBUG ===")
    print(f"Query raw:        {query}")
    print(f"Query normalized: {query_norm}")
    print(f"Query tokens:     {query_tokens}")

    print("\nPOI:")
    print(f"Name:             {df.iloc[row_idx].get('name', '')}")
    print(f"Category:         {df.iloc[row_idx].get('category_final', '')}")
    print(f"POI raw text:     {poi_raw}")
    print(f"POI lemma text:   {poi_lemma}")
    print(f"POI tokens:       {poi_tokens[:80]}")

def build_tfidf(
    df: pd.DataFrame,
    col: str = "poi_text_lemma",
    max_features: int = 5000,
    min_df: int = 2,
) -> tuple:
    """
    Build TF-IDF matrix from a text column.
    N-grami (unigrams + bigrams) su ugrađeni u _tfidf_tokenizer.
    """
    if col not in df.columns:
        print(f"[tfidf] Warning - '{col}' not found, falling back to poi_text")
        col = "poi_text"

    corpus = df[col].fillna("").tolist()

    vectorizer = TfidfVectorizer(
        analyzer=_tfidf_tokenizer,
        max_features=max_features,
        min_df=min_df,
    )

    tfidf_matrix = vectorizer.fit_transform(corpus)
    vocabulary = vectorizer.get_feature_names_out()

    print(f"[tfidf] Corpus size:    {len(corpus)}")
    print(f"[tfidf] Vocabulary size: {len(vocabulary)}")
    print(f"[tfidf] TF-IDF matrix:   {tfidf_matrix.shape}")
    print(f"[tfidf] Top 20 terms:    {vocabulary[:20].tolist()}")

    return vectorizer, tfidf_matrix

def check_query_doc_tokenization(
    query: str,
    vectorizer: TfidfVectorizer,
    df: pd.DataFrame,
    row_idx: int = 0,
) -> None:
    """
    Provjera da li query i dokument prolaze kroz identičnu tokenizaciju
    prije nego stignu u vectorizer.
    """
    from src.preprocessing.normalize import normalize_text as preprocess_text

    # Kako query ide kroz pipeline
    query_preprocessed = preprocess_text(query) or query
    query_normalized = normalize(query_preprocessed) or query_preprocessed
    query_tokens = _tfidf_tokenizer(query_normalized)

    # Kako dokument ide kroz pipeline (poi_text_lemma je već normaliziran pri buildu)
    doc_raw = df.iloc[row_idx].get("poi_text_lemma", "")
    doc_tokens = _tfidf_tokenizer(str(doc_raw))

    # Što vectorizer vidi za query
    query_vec = vectorizer.transform([query_normalized])
    feature_names = vectorizer.get_feature_names_out()
    nonzero_indices = query_vec.nonzero()[1]
    query_vocab_hits = [feature_names[i] for i in nonzero_indices]

    print(f"\n=== TOKENIZATION CONSISTENCY CHECK ===")
    print(f"Query raw:          '{query}'")
    print(f"After preprocess:   '{query_preprocessed}'")
    print(f"After normalize:    '{query_normalized}'")
    print(f"Query tokens:       {query_tokens}")
    print(f"Query vocab hits:   {query_vocab_hits}")
    print(f"\nDoc raw (lemma):    '{doc_raw[:100]}...'")
    print(f"Doc tokens[:20]:    {doc_tokens[:20]}")

    # Presjek — koliko query tokena se pojavljuje u dokumentu
    overlap = set(query_tokens) & set(doc_tokens)
    print(f"\nToken overlap:      {overlap}")

def search_tfidf(
    query: str,
    vectorizer: TfidfVectorizer,
    tfidf_matrix,
    df: pd.DataFrame,
    top_k: int = 10,
) -> pd.DataFrame:
    """
    Search POI dataset using TF-IDF cosine similarity.

    Args:
        query:        user input string
        vectorizer:   fitted TfidfVectorizer
        tfidf_matrix: fitted TF-IDF matrix
        df:           cleaned POI dataframe
        top_k:        number of results to return

    Returns:
        DataFrame of top_k matching POIs with similarity scores
    """

    query_norm = normalize(preprocess_text(query) or query)
    if not query_norm:
        return pd.DataFrame()
    query_vec = vectorizer.transform([query_norm])
    
    scores = cosine_similarity(query_vec, tfidf_matrix).flatten()

    top_indices = np.argsort(scores)[::-1][:top_k]
    results = df.iloc[top_indices].copy()
    results["similarity_score"] = scores[top_indices]

    print(f"[tfidf] Query: '{query}'")
    print(f"[tfidf] Top {top_k} results (similarity scores):")
    print(results[["name", "category_final", "similarity_score"]].to_string())

    return results


def save_vectorizer(vectorizer, path: str = "models/tfidf_vectorizer.pkl") -> None:
    """Save fitted TfidfVectorizer to disk."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(vectorizer, f)
    print(f"[tfidf] Vectorizer saved: {path}")


def load_vectorizer(path: str = "models/tfidf_vectorizer.pkl") -> TfidfVectorizer:
    """Load fitted TfidfVectorizer from disk."""
    with open(path, "rb") as f:
        vectorizer = pickle.load(f)
    print(f"[tfidf] Vectorizer loaded: {path}")
    return vectorizer



def compare_ngrams(
    df: pd.DataFrame,
    col: str = "poi_text_lemma",
    test_queries: list = None,
) -> None:
    """
    Compare TF-IDF performance across different n-gram ranges.
    Prints top 5 results per query for each n-gram setting.
    """
    if test_queries is None:
        test_queries = [
            "coffee near burnside",
            "mexican restaurant",
            "wheelchair accessible cafe",
        ]

    ngram_settings = [(1, 1), (1, 2), (1, 3)]
    corpus = df[col].fillna("").tolist()

    for ngram in ngram_settings:
        print(f"\n{'='*50}")
        print(f"N-gram range: {ngram}")
        print(f"{'='*50}")

        vec = TfidfVectorizer(
            analyzer=_tfidf_tokenizer,
            max_features=5000,
            min_df=2,
        )
        
        matrix = vec.fit_transform(corpus)

        for query in test_queries:
            query_norm = normalize(query) or query
            query_vec = vec.transform([query_norm])
            scores = cosine_similarity(query_vec, matrix).flatten()
            top_indices = np.argsort(scores)[::-1][:3]

            print(f"\nQuery: '{query}'")
            for i in top_indices:
                print(f"  {df.iloc[i]['name']} | {df.iloc[i]['category_final']} | score: {round(scores[i], 3)}")


def run(df: pd.DataFrame) -> tuple:
    return build_tfidf(df)