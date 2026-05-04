import geopandas as gpd

gdf = gpd.read_file("data/raw/pois_portland_oregon_usa.geojson")
print(gdf.geometry.geom_type.value_counts())
print(f"\nTotal: {len(gdf)}")