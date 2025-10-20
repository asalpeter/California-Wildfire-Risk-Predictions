import math
from pathlib import Path

import geopandas as gpd
import numpy as np
import requests
from shapely.geometry import Polygon, shape

OUT = Path("src/features/hexgrid_ca.geojson")
OUT.parent.mkdir(parents=True, exist_ok=True)

# --- 1) Get California boundary (GeoJSON) ---
US_STATES_URL = (
    "https://raw.githubusercontent.com/PublicaMundi/MappingAPI/master/data/geojson/us-states.json"
)
resp = requests.get(US_STATES_URL, timeout=20)
resp.raise_for_status()
states = resp.json()
ca_geom = None
for f in states["features"]:
    if f["properties"].get("name") == "California":
        ca_geom = shape(f["geometry"])
        break
if ca_geom is None:
    raise RuntimeError("Could not fetch California boundary.")

# --- 2) Work in a projected CRS for accurate hex sizing (EPSG:3310 - California Albers) ---
g_ca = gpd.GeoSeries([ca_geom], crs="EPSG:4326").to_crs("EPSG:3310")
ca_poly_3310 = g_ca.iloc[0]
minx, miny, maxx, maxy = ca_poly_3310.bounds

# Hex parameters (~10 km across flats). Adjust for coarser/finer grid.
hex_d = 10_000.0  # nominal hex "diameter" across flats in meters
r = hex_d / 2.0  # apothem
R = r / math.cos(math.pi / 6.0)  # radius (center to vertex)
dx = 2 * r
dy = 1.5 * R

# --- 3) Generate hex centers over CA bbox ---
xs = np.arange(minx - dx, maxx + dx, dx)
ys = np.arange(miny - dy, maxy + dy, dy)

hexes = []
row_idx = 0
for j, y in enumerate(ys):
    x_offset = 0.0 if (j % 2 == 0) else r
    for i, x in enumerate(xs):
        cx = x + x_offset
        cy = y
        # Skip centers clearly outside bbox to speed up
        if cx < minx - dx or cx > maxx + dx or cy < miny - dy or cy > maxy + dy:
            continue
        # Build hexagon around center (pointy-topped)
        angles = np.deg2rad([0, 60, 120, 180, 240, 300])
        verts = [(cx + R * math.sin(a), cy + R * math.cos(a)) for a in angles]
        poly = Polygon(verts)
        # Keep only hexes whose centroid lies within CA
        if poly.centroid.within(ca_poly_3310):
            hexes.append(poly)

# --- 4) Pack into GeoDataFrame, clip to CA, id them, write out ---
gdf = gpd.GeoDataFrame({"hex_id": range(1, len(hexes) + 1)}, geometry=hexes, crs="EPSG:3310")
gdf = gpd.overlay(
    gdf, gpd.GeoDataFrame(geometry=[ca_poly_3310], crs="EPSG:3310"), how="intersection"
)
gdf["hex_id"] = range(1, len(gdf) + 1)
gdf = gdf.to_crs("EPSG:4326")

gdf.to_file(OUT, driver="GeoJSON")
print(f"Wrote {len(gdf)} CA hexes -> {OUT}")
