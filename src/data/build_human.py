import tempfile
import time
from pathlib import Path

import certifi
import geopandas as gpd
import requests
import yaml
from shapely.geometry import shape

CFG = yaml.safe_load(open("src/config.yaml"))

HEX = Path("src/features/hexgrid_ca.geojson")
OUT = Path("src/features/hex_static_human.parquet")
OUT.parent.mkdir(parents=True, exist_ok=True)

TIGER_PRIMARY_URL = (
    "https://www2.census.gov/geo/tiger/TIGER2020/PRIMARYROADS/tl_2020_us_primaryroads.zip"
)
TIGER_PLACES_CA_URL = "https://www2.census.gov/geo/tiger/TIGER2020/PLACE/tl_2020_06_place.zip"


def fetch_to_tmp(url: str) -> Path:
    tmpdir = Path(tempfile.mkdtemp())
    out = tmpdir / url.split("/")[-1]
    err = None
    for attempt in range(4):
        try:
            with requests.get(url, stream=True, timeout=90, verify=certifi.where()) as r:
                r.raise_for_status()
                with open(out, "wb") as f:
                    for chunk in r.iter_content(1024 * 1024):
                        if chunk:
                            f.write(chunk)
            return out
        except Exception as e:
            err = e
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"Failed to download {url}: {err}")


US_STATES_URL = (
    "https://raw.githubusercontent.com/PublicaMundi/MappingAPI/master/data/geojson/us-states.json"
)
ca = next(
    shape(f["geometry"])
    for f in requests.get(US_STATES_URL, timeout=30, verify=certifi.where()).json()["features"]
    if f["properties"].get("name") == "California"
)
g_ca = gpd.GeoSeries([ca], crs="EPSG:4326")
ca_3310 = g_ca.to_crs("EPSG:3310").iloc[0]

hexes = gpd.read_file(HEX)
hexes_3310 = hexes.to_crs("EPSG:3310")
cent = gpd.GeoDataFrame(
    {"hex_id": hexes["hex_id"].to_numpy()},
    geometry=hexes_3310.geometry.centroid,  # centroid of polygon
    crs="EPSG:3310",
)

primary_zip = fetch_to_tmp(TIGER_PRIMARY_URL)
places_zip = fetch_to_tmp(TIGER_PLACES_CA_URL)

roads = gpd.read_file(primary_zip)[["MTFCC", "FULLNAME", "geometry"]]
roads = roads[roads["MTFCC"] == "S1100"].to_crs("EPSG:3310")
roads_ca = gpd.clip(roads, ca_3310).reset_index(drop=True)
r_sindex = roads_ca.sindex

urban = gpd.read_file(places_zip)[["NAME", "geometry"]].to_crs("EPSG:3310").reset_index(drop=True)
u_sindex = urban.sindex


def candidate_idxs_by_expanding_bbox(pt, sindex, r0=5000, rmax=100000, factor=2.0):
    """Return some candidate indices near point by expanding a bbox until we hit anything."""
    if sindex is None:
        return []
    r = r0
    while r <= rmax:
        bbox = pt.buffer(r).bounds
        idxs = list(sindex.intersection(bbox))
        if idxs:
            return idxs
        r = int(r * factor)
    return []


def nearest_distance_m(pt, gdf, sindex):
    idxs = candidate_idxs_by_expanding_bbox(pt, sindex)
    if not idxs:
        return float("nan")
    return float(min(pt.distance(gdf.iloc[i].geometry) for i in idxs))


def inside_any(pt, gdf, sindex):
    idxs = candidate_idxs_by_expanding_bbox(pt, sindex)
    if not idxs:
        return False
    return any(pt.within(gdf.iloc[i].geometry) for i in idxs)


cent["dist_road_m"] = cent.geometry.apply(lambda p: nearest_distance_m(p, roads_ca, r_sindex))
cent["dist_urban_m"] = cent.geometry.apply(lambda p: nearest_distance_m(p, urban, u_sindex))
cent["in_urban"] = cent.geometry.apply(lambda p: inside_any(p, urban, u_sindex))

out = cent[["hex_id", "dist_road_m", "dist_urban_m", "in_urban"]].copy()
out["dist_road_km"] = out["dist_road_m"] / 1000.0
out["dist_urban_km"] = out["dist_urban_m"] / 1000.0
out = out.drop(columns=["dist_road_m", "dist_urban_m"])

OUT.write_bytes(out.to_parquet(index=False))
print(f"Wrote human proxies -> {OUT} (rows={len(out)})")
