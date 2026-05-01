"""
text_join.py
------------
Joins normalized text fields into a single 'poi_text' field.
This is the field that will be compared against user queries
via TF-IDF and later embeddings.

Field priority / weighting logic:
  1. name_norm        — primary identifier, highest importance
  2. category_norm    — always present, good discriminator
  3. cuisine_norm     — high discriminative power for F&B
  4. addr_street      — useful for "restaurant on X street" queries
  
  NOT included in poi_text:
  - addr_city         — same for all POIs in dataset, adds noise
  - is_24_7 etc.      — boolean flags, handled by rule-based filter
  - wheelchair        — same, rule-based

The result is a plain text string, suitable for:
  - TF-IDF vectorization
  - Sentence embedding (later stage)
"""

import pandas as pd


# Fields to join, in priority order
# Each tuple: (column_name, repeat_times)
# Repeat = weight proxy for TF-IDF (name appears 3x → higher weight)
_JOIN_FIELDS = [
    ("name_norm", 3),       # most important
    ("category_norm", 2),
    ("cuisine_norm", 1),
    ("addr_street", 1),
]


def build_poi_text(row: pd.Series) -> str | None:
    """
    Builds the joined text string for a single POI row.
    Repeats fields according to their weight.
    Returns None if all fields are empty.
    """
    parts = []

    for col, repeat in _JOIN_FIELDS:
        value = row.get(col)
        if value is not None and str(value).strip() not in {"", "nan", "none"}:
            text = str(value).strip()
            parts.extend([text] * repeat)

    if not parts:
        return None

    return " ".join(parts)


def apply_text_join(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies text joining to create 'poi_text' column.

    Expects normalized columns from normalize.py and address.py:
      name_norm, category_norm, cuisine_norm, addr_street

    Output:
      poi_text   : joined weighted text string (for TF-IDF/embeddings)
    """
    df = df.copy()

    df["poi_text"] = df.apply(build_poi_text, axis=1)

    n_filled = df["poi_text"].notna().sum()
    print(f"[text_join.py] poi_text created for {n_filled}/{len(df)} rows.")

    # Quick sample
    sample = df["poi_text"].dropna().sample(min(3, n_filled), random_state=42).tolist()
    for s in sample:
        print(f"  Example: {s[:120]}")

    return df
