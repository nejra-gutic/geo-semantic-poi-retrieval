import pandas as pd
import geopandas as gpd


gdf = gpd.read_file("data/raw/pois_portland_oregon_usa.geojson")
print(gdf.columns[:5].tolist())
print(type(gdf))
print("geometry" in gdf.columns)

# CSV provjera
df = pd.read_csv("data/processed/cleaned_pois.csv", index_col=0)
print("=== CSV ===")
print(f"Shape: {df.shape}")
print(f"\nKolone: {df.columns.tolist()}")
print(f"\nMissing values:")
print(df.isnull().sum()[df.isnull().sum() > 0])
print(f"\nSample:")
print(df[["name", "category_final", "cuisine_clean", "latitude", "longitude", "poi_text"]].head(10))

# GeoJSON provjera
gdf = gpd.read_file("data/processed/cleaned_pois.geojson")
print("\n=== GeoJSON ===")
print(f"Shape: {gdf.shape}")
print(f"Geometry types:\n{gdf.geometry.geom_type.value_counts()}")
print(f"CRS: {gdf.crs}")