"""
cuisine.py
----------
Normalizes the 'cuisine' OSM field into a clean list of tags.

OSM cuisine values are semicolon-separated (e.g. "pizza;italian;pasta").
This module:
  - splits on ; and ,
  - normalizes each tag (lowercase, underscores → spaces, unidecode)
  - maps known synonyms to canonical forms
  - returns both a list (cuisine_tags) and a joined string (cuisine_norm)
"""

import pandas as pd
from unidecode import unidecode
import re


# Canonical mappings for common cuisine variations
CUISINE_SYNONYMS = {
    "burger": "burger",
    "burgers": "burger",
    "fast food": "fast_food",
    "fast-food": "fast_food",
    "fastfood": "fast_food",
    "american": "american",
    "italian": "italian",
    "pizza": "pizza",
    "chinese": "chinese",
    "japanese": "japanese",
    "sushi": "sushi",
    "thai": "thai",
    "indian": "indian",
    "mexican": "mexican",
    "mediterranean": "mediterranean",
    "greek": "greek",
    "turkish": "turkish",
    "vietnamese": "vietnamese",
    "korean": "korean",
    "french": "french",
    "spanish": "spanish",
    "middle eastern": "middle_eastern",
    "middle_eastern": "middle_eastern",
    "sandwich": "sandwich",
    "sandwiches": "sandwich",
    "coffee": "coffee",
    "cafe": "cafe",
    "bakery": "bakery",
    "vegan": "vegan",
    "vegetarian": "vegetarian",
    "seafood": "seafood",
    "fish": "seafood",
    "steak": "steak",
    "barbecue": "barbecue",
    "bbq": "barbecue",
    "chicken": "chicken",
    "noodle": "noodle",
    "noodles": "noodle",
    "ramen": "ramen",
    "asian": "asian",
    "fusion": "fusion",
    "regional": "regional",
    "international": "international",
}


def _normalize_tag(tag: str) -> str:
    """Normalize a single cuisine tag."""
    tag = unidecode(tag.strip().lower())
    tag = tag.replace("-", " ").replace("_", " ")
    tag = re.sub(r"\s+", " ", tag).strip()
    return CUISINE_SYNONYMS.get(tag, tag.replace(" ", "_"))  # canonical or underscore form


def parse_cuisine(value) -> list[str]:
    """
    Parses a raw OSM cuisine string into a list of normalized tags.

    Examples:
      "pizza;italian"    → ["pizza", "italian"]
      "fast_food"        → ["fast_food"]
      "unknown" / NaN    → []
    """
    if pd.isna(value):
        return []

    text = str(value).strip().lower()

    if text in {"unknown", "", "nan", "none", "yes"}:
        return []

    # Split on ; or ,
    raw_tags = re.split(r"[;,]", text)
    tags = [_normalize_tag(t) for t in raw_tags if t.strip()]
    tags = [t for t in tags if t]  # remove empty

    return tags


def apply_cuisine(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies cuisine parsing.

    Input:  df with 'cuisine' column
    Output: df with:
              cuisine_tags  (list of normalized strings)
              cuisine_norm  (single joined string, for text matching)
    """
    df = df.copy()

    if "cuisine" not in df.columns:
        print("[cuisine.py] Warning: no 'cuisine' column found.")
        return df

    df["cuisine_tags"] = df["cuisine"].apply(parse_cuisine)

    # Joined string for TF-IDF / text matching
    df["cuisine_norm"] = df["cuisine_tags"].apply(
        lambda tags: " ".join(tags) if tags else None
    )

    n_parsed = (df["cuisine_tags"].apply(len) > 0).sum()
    print(f"[cuisine.py] Parsed cuisine for {n_parsed}/{len(df)} rows.")

    return df
