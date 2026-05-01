"""
address.py
----------
Parses the 'addr' field into structured components using usaddress.

usaddress handles US addresses well (Portland dataset).
For non-US addresses (e.g. Sarajevo), falls back to raw normalized string.

Output columns:
  - addr_norm        : normalized full address string (from normalize.py)
  - addr_street      : extracted street name
  - addr_street_num  : extracted street number
  - addr_city        : extracted city (if present)

Note: usaddress sometimes fails on unusual formats → fallback to None.
"""

import pandas as pd
import re

try:
    import usaddress
    _USADDRESS_AVAILABLE = True
except ImportError:
    _USADDRESS_AVAILABLE = False
    print("[address.py] Warning: usaddress not installed. Install with: pip install usaddress")
    print("[address.py] Falling back to regex-based parsing.")


# usaddress tag → our field mapping
_TAG_MAP = {
    "AddressNumber": "addr_street_num",
    "StreetName": "addr_street",
    "StreetNamePreDirectional": "addr_street",
    "StreetNamePostType": "addr_street",
    "PlaceName": "addr_city",
}


def _parse_with_usaddress(addr_str: str) -> dict:
    """
    Uses usaddress to parse a US address string.
    Returns dict with addr_street, addr_street_num, addr_city.
    """
    result = {"addr_street": None, "addr_street_num": None, "addr_city": None}

    try:
        tagged, _ = usaddress.tag(addr_str)

        street_parts = []
        for tag, value in tagged.items():
            if tag == "AddressNumber":
                result["addr_street_num"] = value
            elif tag in ("StreetNamePreDirectional", "StreetName", "StreetNamePostType"):
                street_parts.append(value)
            elif tag == "PlaceName":
                result["addr_city"] = value

        if street_parts:
            result["addr_street"] = " ".join(street_parts)

    except Exception:
        pass  # usaddress.RepeatedLabelError or similar

    return result


def _parse_with_regex(addr_str: str) -> dict:
    """
    Simple regex fallback: tries to extract street number from the start.
    """
    result = {"addr_street": None, "addr_street_num": None, "addr_city": None}

    addr_str = addr_str.strip()

    # Try: "123 Some Street Name"
    match = re.match(r"^(\d+)\s+(.+)$", addr_str)
    if match:
        result["addr_street_num"] = match.group(1)
        result["addr_street"] = match.group(2)
    else:
        result["addr_street"] = addr_str

    return result


def parse_address(value) -> dict:
    """
    Main address parser. Returns dict with addr_street, addr_street_num, addr_city.
    """
    empty = {"addr_street": None, "addr_street_num": None, "addr_city": None}

    if pd.isna(value) or str(value).strip() == "":
        return empty

    addr_str = str(value).strip()

    if _USADDRESS_AVAILABLE:
        return _parse_with_usaddress(addr_str)
    else:
        return _parse_with_regex(addr_str)


def apply_address(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies address parsing to the 'addr_norm' column (output of normalize.py).
    Falls back to 'addr_clean' or 'addr' if addr_norm not present.

    Output columns added:
      addr_street, addr_street_num, addr_city
    """
    df = df.copy()

    # Use best available address column
    addr_col = None
    for candidate in ("addr_norm", "addr_clean", "addr"):
        if candidate in df.columns:
            addr_col = candidate
            break

    if addr_col is None:
        print("[address.py] Warning: no address column found.")
        return df

    parsed = df[addr_col].apply(parse_address)

    df["addr_street"] = parsed.apply(lambda x: x["addr_street"])
    df["addr_street_num"] = parsed.apply(lambda x: x["addr_street_num"])
    df["addr_city"] = parsed.apply(lambda x: x["addr_city"])

    n_parsed = df["addr_street"].notna().sum()
    print(f"[address.py] Parsed addresses from '{addr_col}': {n_parsed}/{len(df)} rows.")

    return df
