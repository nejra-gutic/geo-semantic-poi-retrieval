"""
retrieval/embeddings.py
-----------------------
Semantic embedding retrieval using Sentence Transformers.

Uses transformer-based embeddings to compare query meaning with POI text meaning.
"""

from pathlib import Path
import numpy as np
import pandas as pd

from sentence_transformers import SentenceTransformer


MODEL_NAME = "all-MiniLM-L6-v2"


def load_embedding_model(model_name: str = MODEL_NAME) -> SentenceTransformer:
    print(f"[embeddings] Loading model: {model_name}")
    return SentenceTransformer(model_name)


def build_embeddings(
    df: pd.DataFrame,
    model: SentenceTransformer,
    col: str = "poi_text_lemma",
    save_path: str = "models/poi_embeddings.npy",
) -> np.ndarray:
    if col not in df.columns:
        print(f"[embeddings] Warning - '{col}' not found, falling back to poi_text")
        col = "poi_text"

    texts = df[col].fillna("").astype(str).tolist()

    print(f"[embeddings] Encoding {len(texts)} POIs...")
    embeddings = model.encode(
        texts,
        batch_size=64,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(save_path, embeddings)

    print(f"[embeddings] Saved embeddings: {save_path}")
    print(f"[embeddings] Shape: {embeddings.shape}")

    return embeddings


def load_embeddings(path: str = "models/poi_embeddings.npy") -> np.ndarray:
    embeddings = np.load(path)
    print(f"[embeddings] Loaded embeddings: {path}")
    print(f"[embeddings] Shape: {embeddings.shape}")
    return embeddings


def get_or_build_embeddings(
    df: pd.DataFrame,
    model: SentenceTransformer,
    col: str = "poi_text_lemma",
    path: str = "models/poi_embeddings.npy",
) -> np.ndarray:
    path_obj = Path(path)

    if path_obj.exists():
        return load_embeddings(path)

    return build_embeddings(df, model, col=col, save_path=path)


def search_embeddings(
    query: str,
    model: SentenceTransformer,
    embeddings: np.ndarray,
    df: pd.DataFrame,
    top_k: int = 10,
    df_filtered: pd.DataFrame = None,
) -> pd.DataFrame:
    if df_filtered is not None:
        if df_filtered.empty:
            print(f"[embeddings] Query: '{query}' → 0 results (empty filter)")
            return pd.DataFrame(columns=list(df.columns) + ["embedding_score"])

        mask = df.index.get_indexer(df_filtered.index)
        mask = mask[mask >= 0]

        if len(mask) == 0:
            print(f"[embeddings] Query: '{query}' → 0 results (empty embedding subset)")
            return pd.DataFrame(columns=list(df.columns) + ["embedding_score"])

        embeddings_subset = embeddings[mask]
        search_df = df_filtered.copy()
        if "poi_id" not in search_df.columns:
            search_df["poi_id"] = search_df.index
        search_df = search_df.reset_index(drop=True)
    else:
        embeddings_subset = embeddings
        search_df = df.copy()
        if "poi_id" not in search_df.columns:
            search_df["poi_id"] = search_df.index
        search_df = search_df.reset_index(drop=True)

    query_embedding = model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    # Since both query and POI embeddings are normalized,
    # dot product is equivalent to cosine similarity.
    scores = embeddings_subset @ query_embedding[0]

    top_indices = np.argsort(scores)[::-1][:top_k]

    results = search_df.iloc[top_indices].copy()
    results["embedding_score"] = scores[top_indices]

    print(f"[embeddings] Query: '{query}' → {len(results)} results")

    if "poi_id" not in results.columns:
        if "index" in results.columns:
            results["poi_id"] = results["index"]
        else:
            results["poi_id"] = results.index

    return results