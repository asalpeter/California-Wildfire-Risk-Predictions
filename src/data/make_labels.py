import os
import glob
from pathlib import Path

import geopandas as gpd
import pandas as pd
import yaml

CFG = yaml.safe_load(open("src/config.yaml"))

HEX = Path("src/features/hexgrid_ca.geojson")
RAW_FIRMS = Path("data/raw/firms")
PROC = Path("data/processed")
PROC.mkdir(parents=True, exist_ok=True)

start_str = os.getenv("START", CFG["history"]["start"])
end_str = os.getenv("END", CFG["history"]["end"])
start_dt = pd.to_datetime(start_str).normalize()
end_dt = pd.to_datetime(end_str).normalize()

# Load hex grid (WGS84)
hexes = gpd.read_file(HEX).to_crs("EPSG:4326")
hex_ids = hexes["hex_id"].tolist()

# Load FIRMS CSVs
files = sorted(glob.glob(str(RAW_FIRMS / "*.csv")))
if not files:
    raise FileNotFoundError(
        "No FIRMS CSVs found in data/raw/firms/. Run src/data/download_firms.py first."
    )

dfs = []
for f in files:
    df = pd.read_csv(f)
    df.columns = [c.lower() for c in df.columns]
    if "latitude" not in df or "longitude" not in df:
        continue
    dcol = "acq_date" if "acq_date" in df.columns else ("date" if "date" in df.columns else None)
    if not dcol:
        continue
    df["date"] = pd.to_datetime(df[dcol]).dt.normalize()
    dfs.append(df[["latitude", "longitude", "date"]])

firms = (
    pd.concat(dfs, ignore_index=True)
    if dfs
    else pd.DataFrame(columns=["latitude", "longitude", "date"])
)
gfirms = gpd.GeoDataFrame(
    firms, geometry=gpd.points_from_xy(firms["longitude"], firms["latitude"]), crs="EPSG:4326"
)

# Spatial join detections → hex_id
joined = gpd.sjoin(
    gfirms,
    hexes[["hex_id", "geometry"]],
    how="inner",
    predicate="within",
)

# Aggregate to hex-day detection counts
daily = joined.groupby(["hex_id", "date"]).size().rename("fires").reset_index()

# Build full hex × date grid over history window
all_dates = pd.date_range(start_dt, end_dt, freq="D").normalize()
grid = pd.MultiIndex.from_product([hex_ids, all_dates], names=["hex_id", "date"]).to_frame(
    index=False
)

# Merge counts, fill 0
counts = grid.merge(daily, on=["hex_id", "date"], how="left").fillna({"fires": 0})
counts["fires"] = counts["fires"].astype(int)

# Binary label: same-day any detection (keep current behavior)
labels = counts.assign(y=(counts["fires"] > 0).astype(int))[["hex_id", "date", "y"]]

# Save both
counts.to_parquet(PROC / "firms_counts.parquet", index=False)
labels.to_parquet(PROC / "labels.parquet", index=False)
print("Wrote", PROC / "firms_counts.parquet", "and", PROC / "labels.parquet")
