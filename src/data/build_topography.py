import gzip
import math
from pathlib import Path
from typing import Dict, List, Tuple

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import requests

TILES_DIR = Path("data/raw/dem/tiles")
TILES_DIR.mkdir(parents=True, exist_ok=True)

HEX_PATH = Path("src/features/hexgrid_ca.geojson")
OUT = Path("src/features/hex_static_topo.parquet")
OUT.parent.mkdir(parents=True, exist_ok=True)

AWS_SKADI = "https://s3.amazonaws.com/elevation-tiles-prod/skadi"


# ---------- small utilities ----------
def log(msg: str):
    print(f"[topo] {msg}", flush=True)


def tile_name(lat: float, lon: float) -> str:
    """Return SRTM 1° x 1° tile name like 'N34W118' for a given lat/lon."""
    lat_floor = math.floor(lat)
    lon_floor = math.floor(lon)
    ns = "N" if lat_floor >= 0 else "S"
    ew = "E" if lon_floor >= 0 else "W"
    return f"{ns}{abs(lat_floor):02d}{ew}{abs(lon_floor):03d}"


def tile_url(name: str) -> str:
    lat_band = name[:3]  # e.g., N34
    return f"{AWS_SKADI}/{lat_band}/{name}.hgt.gz"


def ensure_tile_local(name: str) -> Path:
    """Download the .hgt.gz if needed, decompress to .hgt, return .hgt path."""
    gz = TILES_DIR / f"{name}.hgt.gz"
    hgt = TILES_DIR / f"{name}.hgt"
    if not hgt.exists():
        if not gz.exists():
            url = tile_url(name)
            log(f"Downloading {name} …")
            r = requests.get(url, timeout=60)
            if r.status_code != 200:
                raise FileNotFoundError(f"Tile not found: {url} (HTTP {r.status_code})")
            gz.write_bytes(r.content)
        log(f"Decompressing {gz.name} …")
        with gzip.open(gz, "rb") as f_in:
            hgt.write_bytes(f_in.read())
    return hgt


def horn_slope_aspect(z3x3: np.ndarray, lat_deg: float) -> Tuple[float, float]:
    """
    Horn (1981) 3x3 slope/aspect in degrees.
    SRTM spacing is 1 arc-second (~1/3600 deg). Convert to meters using latitude.
    """
    if z3x3.shape != (3, 3) or np.any(~np.isfinite(z3x3)):
        return 0.0, np.nan  # safe fallback

    cell_deg = 1.0 / 3600.0
    lat_rad = np.deg2rad(lat_deg)
    m_per_deg_lat = 111_320.0
    m_per_deg_lon = 111_320.0 * np.cos(lat_rad)
    dx = cell_deg * m_per_deg_lon
    dy = cell_deg * m_per_deg_lat

    dzdx = (
        (z3x3[0, 2] + 2 * z3x3[1, 2] + z3x3[2, 2]) - (z3x3[0, 0] + 2 * z3x3[1, 0] + z3x3[2, 0])
    ) / (8.0 * dx)
    dzdy = (
        (z3x3[2, 0] + 2 * z3x3[2, 1] + z3x3[2, 2]) - (z3x3[0, 0] + 2 * z3x3[0, 1] + z3x3[0, 2])
    ) / (8.0 * dy)

    slope_rad = np.arctan(np.hypot(dzdx, dzdy))
    slope_deg = float(np.degrees(slope_rad))

    # Aspect: 0=N, 90=E, 180=S, 270=W → 0..360
    aspect_rad = np.arctan2(dzdy, -dzdx)
    aspect_deg = float(np.degrees(aspect_rad))
    if aspect_deg < 0:
        aspect_deg += 360.0

    return slope_deg, aspect_deg


# ---------- load hex centroids (POINTS) ----------
if not HEX_PATH.exists():
    raise FileNotFoundError(f"Missing hex grid: {HEX_PATH}")

log("Loading hex grid …")
hexes = gpd.read_file(HEX_PATH)

# Compute accurate centroids in a projected CRS, then return to WGS84
hexes_proj = hexes.to_crs("EPSG:3310")
hex_centroids_proj = hexes_proj.geometry.buffer(0).centroid  # buffer(0) to fix any slivers
cent = gpd.GeoDataFrame(
    hexes[["hex_id"]].copy(), geometry=hex_centroids_proj, crs="EPSG:3310"
).to_crs("EPSG:4326")

# Now geometry is POINT → safe to read .x/.y
cent["lon"] = cent.geometry.x
cent["lat"] = cent.geometry.y

# Group hex centroids by tile (download & open each tile once)
cent["tile"] = [
    tile_name(lat, lon) for lat, lon in zip(cent["lat"].to_numpy(), cent["lon"].to_numpy())
]
tiles = cent["tile"].unique().tolist()
log(f"Unique tiles needed: {len(tiles)}")

# Ensure tiles exist
for t in tiles:
    ensure_tile_local(t)

# Process tiles
results: List[Dict] = []
for t in tiles:
    hgt = ensure_tile_local(t)
    with rasterio.open(hgt) as src:
        sub = cent[cent["tile"] == t].copy()
        rows, cols = src.index(sub["lon"].to_numpy(), sub["lat"].to_numpy())

        band1 = src.read(1)  # 3601x3601
        height, width = band1.shape

        for hex_id, lat, lon, r, c in zip(
            sub["hex_id"].to_numpy(), sub["lat"].to_numpy(), sub["lon"].to_numpy(), rows, cols
        ):
            r0, r1 = max(0, r - 1), min(height, r + 2)
            c0, c1 = max(0, c - 1), min(width, c + 2)
            z = band1[r0:r1, c0:c1]
            elev = float(band1[r, c]) if (0 <= r < height and 0 <= c < width) else np.nan

            slope_deg, aspect_deg = (0.0, np.nan)
            if z.shape == (3, 3):
                slope_deg, aspect_deg = horn_slope_aspect(z.astype(float), lat_deg=float(lat))

            if np.isfinite(aspect_deg):
                arad = math.radians(aspect_deg)
                aspect_sin = math.sin(arad)
                aspect_cos = math.cos(arad)
            else:
                aspect_sin = 0.0
                aspect_cos = 0.0

            results.append(
                {
                    "hex_id": int(hex_id),
                    "elev": elev,
                    "slope_deg": float(slope_deg),
                    "aspect_sin": float(aspect_sin),
                    "aspect_cos": float(aspect_cos),
                }
            )

out_df = pd.DataFrame(results).sort_values("hex_id").drop_duplicates("hex_id", keep="first")
OUT.write_bytes(out_df.to_parquet(index=False))
log(f"Wrote topography features -> {OUT} (rows={len(out_df)})")
