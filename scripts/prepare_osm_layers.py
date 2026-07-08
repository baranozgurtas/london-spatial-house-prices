import geopandas as gpd
from pathlib import Path

GPKG = Path("data/external/greater-london-260706/greater-london.gpkg")
OUT = Path("data/external")

water = gpd.read_file(GPKG, layer="gis_osm_waterways_free")
thames = water[water["name"].str.contains("Thames", case=False, na=False)]
thames.to_file(OUT / "thames.geojson", driver="GeoJSON")
print(f"thames features: {len(thames)}")

transport = gpd.read_file(GPKG, layer="gis_osm_transport_free")
print("transport fclasses:", sorted(transport["fclass"].dropna().unique()))
stations = transport[transport["fclass"].str.contains("station", case=False, na=False)]
stations.to_file(OUT / "london_stations.geojson", driver="GeoJSON")
print(f"station features: {len(stations)}")

pois = gpd.read_file(GPKG, layer="gis_osm_pois_free")
pois.to_file(OUT / "london_pois.geojson", driver="GeoJSON")
print(f"poi features: {len(pois)}")
