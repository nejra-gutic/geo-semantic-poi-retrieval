"""
data/acquire.py
---------------
Fetches Points of Interest (POI) from OpenStreetMap using osmnx
and saves them locally as CSV and GeoJSON.

Usage:
    python -m src.data.acquire --place "Portland, Oregon, USA" --output data/raw
"""

import argparse
import osmnx as ox
from pathlib import Path
from src.utils.io import save_csv, save_geojson


TAGS = {
    "amenity": True,
    "shop": True,
    "tourism": True,
}


def fetch_pois(place: str):
    """Download POI features from OSM for a given place name."""
    print(f"[acquire] Fetching POIs for: {place}")
    pois = ox.features_from_place(place, TAGS)
    print(f"[acquire] Fetched {len(pois)} features")
    print(f"[acquire] Columns: {pois.columns.tolist()}")
    return pois


def main():
    parser = argparse.ArgumentParser(description="Fetch OSM POI data for a place.")
    parser.add_argument(
        "--place",
        type=str,
        default="Portland, Oregon, USA",
        help='Place name to query, e.g. "Portland, Oregon, USA"',
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/raw",
        help="Output directory for raw CSV and GeoJSON files",
    )
    args = parser.parse_args()

    pois = fetch_pois(args.place)

    out_dir = Path(args.output)

    place_name = args.place.lower().replace(", ", "_").replace(" ", "_")

    save_csv(pois, out_dir / f"pois_{place_name}.csv")
    save_geojson(pois, out_dir / f"pois_{place_name}.geojson")


if __name__ == "__main__":
    main()
