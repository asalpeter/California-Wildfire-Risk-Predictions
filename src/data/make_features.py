import faulthandler
import glob
import re
import time
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr
import yaml

faulthandler.enable()
faulthandler.dump_traceback_later(600)  # dump stack if we stall for 10 minutes
t0 = time.time()


def log(msg: str):
    print(f"[make_features +{time.time()-t0:6.1f}s] {msg}", flush=True)


CFG = yaml.safe_load(open("src/config.yaml"))
HEX = Path("src/features/hexgrid_ca.geojson")
PROC = Path("data/processed")
PARTS = PROC / "features_parts"
for p in (PROC, PARTS):
    p.mkdir(parents=True, exist_ok=True)

# ----------------- hex centroids --------------------
log("Loading hex grid…")
hexes = gpd.read_file(HEX).to_crs("EPSG:3310")
hex_centroids = hexes.copy()
hex_centroids["geometry"] = hex_centroids.geometry.centroid
hex_centroids = hex_centroids.to_crs("EPSG:4326")
log(f"Hex count: {len(hex_centroids):,}")


# ----------------- helpers -------------------------
def add_rollups(feat: pd.DataFrame, windows=(1, 3, 7)) -> pd.DataFrame:
    feat = feat.sort_values(["hex_id", "date"]).copy()
    grp = feat.groupby("hex_id", group_keys=False)
    for w in windows:
        feat[f"temp_max_{w}d"] = grp["temp"].apply(lambda s: s.rolling(w, min_periods=1).max())
        feat[f"rh_min_{w}d"] = grp["rh"].apply(lambda s: s.rolling(w, min_periods=1).min())
        feat[f"wind_mean_{w}d"] = grp["wind"].apply(lambda s: s.rolling(w, min_periods=1).mean())
        feat[f"precip_sum_{w}d"] = grp["precip"].apply(lambda s: s.rolling(w, min_periods=1).sum())
    return feat


def files_for(var: str):
    return sorted(glob.glob(f"data/raw/gridmet/{var}_*.nc")) + sorted(
        glob.glob(f"data/raw/gridmet/{var}_*.nc4")
    )


def open_one(fp):
    last = None
    for engine in ("h5netcdf", "netcdf4"):
        try:
            ds = xr.open_dataset(fp, engine=engine, decode_times=True)
            if "day" in ds.dims:
                ds = ds.rename({"day": "time"})
            return ds
        except Exception as e:
            last = e
    raise last


def pick_var(ds, candidates):
    for name in candidates:
        if name in ds.data_vars:
            return ds[name]
    for name in ds.data_vars:
        if name.lower() != "crs":
            return ds[name]
    raise KeyError("No data variable found")


def to_celsius(da):
    units = (da.attrs.get("units", "") or "").lower()
    return da - 273.15 if units in ("k", "kelvin") else da


def nearest_indices(sorted_vals, query_vals):
    ascending = sorted_vals[0] < sorted_vals[-1]
    if not ascending:
        arr = sorted_vals[::-1]
        idx = np.searchsorted(arr, query_vals, side="left")
        idx = np.clip(idx, 0, len(arr) - 1)
        left = np.clip(idx - 1, 0, len(arr) - 1)
        right = idx
        choose_left = np.abs(query_vals - arr[left]) <= np.abs(query_vals - arr[right])
        nearest = np.where(choose_left, left, right)
        return (len(sorted_vals) - 1) - nearest
    else:
        arr = sorted_vals
        idx = np.searchsorted(arr, query_vals, side="left")
        idx = np.clip(idx, 0, len(arr) - 1)
        left = np.clip(idx - 1, 0, len(arr) - 1)
        right = idx
        choose_left = np.abs(query_vals - arr[left]) <= np.abs(query_vals - arr[right])
        return np.where(choose_left, left, right)


# ----------------- gather files & years -------------
tmmx_files = files_for("tmmx")
rmin_files = files_for("rmin")
vs_files = files_for("vs")
pr_files = files_for("pr")
if not (tmmx_files and rmin_files and vs_files and pr_files):
    raise FileNotFoundError("Missing GridMET files. Run: python src/data/download_gridmet.py")
log(
    f"Found GridMET files: tmmx={len(tmmx_files)}, rmin={len(rmin_files)}, vs={len(vs_files)}, pr={len(pr_files)}"
)


def year_from(fp):
    m = re.search(r"_(\d{4})\.nc4?$", str(fp))
    return int(m.group(1)) if m else None


def map_by_year(files):
    out = {}
    for f in files:
        y = year_from(f)
        if y is not None:
            out[y] = f
    return out


m_t = map_by_year(tmmx_files)
m_r = map_by_year(rmin_files)
m_v = map_by_year(vs_files)
m_p = map_by_year(pr_files)

start = pd.to_datetime(CFG["history"]["start"]).year
end = pd.to_datetime(CFG["history"]["end"]).year
years = sorted(set(m_t) & set(m_r) & set(m_v) & set(m_p))
years = [y for y in years if start <= y <= end]
if not years:
    raise RuntimeError("No overlapping yearly files in requested history window.")
log(f"Years to process: {years[0]}–{years[-1]} ({len(years)} years)")

# ----------------- build index map once -------------
idx_map_path = Path("src/features/grid_index_map.parquet")
lat_idx = lon_idx = None
n_lon = None


def build_index_map_from(ds_like):
    global lat_idx, lon_idx, n_lon
    lat_name = "lat" if "lat" in ds_like.coords else "latitude"
    lon_name = "lon" if "lon" in ds_like.coords else "longitude"
    lat_vals = ds_like[lat_name].values
    lon_vals = ds_like[lon_name].values
    cent = hex_centroids.copy()
    cent["lon"] = cent.geometry.x
    cent["lat"] = cent.geometry.y
    lat_idx = nearest_indices(lat_vals, cent["lat"].to_numpy())
    lon_idx = nearest_indices(lon_vals, cent["lon"].to_numpy())
    n_lon = ds_like.sizes[lon_name]
    idx = pd.DataFrame({"hex_id": cent["hex_id"], "ilat": lat_idx, "ilon": lon_idx})
    idx.to_parquet(idx_map_path, index=False)
    log(f"Cached grid index map -> {idx_map_path}")


if idx_map_path.exists():
    idx = pd.read_parquet(idx_map_path)
    hex_centroids = hex_centroids.merge(idx, on="hex_id", how="inner")
    lat_idx = hex_centroids["ilat"].to_numpy()
    lon_idx = hex_centroids["ilon"].to_numpy()
    log("Loaded cached grid index map.")
else:
    probe = open_one(m_t[years[0]])
    build_index_map_from(probe)


# ----------------- per-year sampling ----------------
def sample_year(y: int) -> Path:
    """Open one year per variable, sample to hexes, compute VPD, write a year parquet, return path."""
    fp_out = PARTS / f"features_{y}.parquet"
    if fp_out.exists():
        log(f"[{y}] Skip (exists): {fp_out.name}")
        return fp_out

    log(f"[{y}] Opening yearly datasets…")
    ds_t = open_one(m_t[y])
    da_t = pick_var(ds_t, ["tmmx", "air_temperature", "tmmx_daily"])
    ds_r = open_one(m_r[y])
    da_r = pick_var(ds_r, ["rmin", "relative_humidity", "relative_humidity_min"])
    ds_v = open_one(m_v[y])
    da_v = pick_var(ds_v, ["vs", "wind_speed", "wind"])
    ds_p = open_one(m_p[y])
    da_p = pick_var(ds_p, ["pr", "precipitation_amount", "precip", "precipitation"])

    temp = to_celsius(da_t)  # °C
    rh = da_r                # %
    wind = da_v              # m/s
    precip = da_p            # mm/day

    daily = xr.Dataset({"temp": temp, "rh": rh, "wind": wind, "precip": precip})

    lat_name = "lat" if "lat" in daily.coords else "latitude"
    lon_name = "lon" if "lon" in daily.coords else "longitude"
    daily_flat = daily.stack(cell=(lat_name, lon_name))
    flat_idx = (lat_idx * daily.sizes[lon_name] + lon_idx).astype(int)
    sub = daily_flat.isel(cell=xr.DataArray(flat_idx, dims="point")).transpose("time", "point")

    df = sub.to_dataframe().reset_index()
    df = df.rename(columns={"time": "date"})
    df["hex_id"] = hex_centroids["hex_id"].to_numpy()[df["point"].to_numpy()]
    feat_y = df.drop(columns=["point"]).copy()
    feat_y["date"] = pd.to_datetime(feat_y["date"]).dt.normalize()

    # VPD
    es = 6.112 * np.exp((17.67 * feat_y["temp"].to_numpy()) / (feat_y["temp"].to_numpy() + 243.5))
    ea = es * (feat_y["rh"].to_numpy() / 100.0)
    feat_y["vpd"] = np.maximum(es - ea, 0.0)

    feat_y.to_parquet(fp_out, index=False)
    log(f"[{y}] Wrote -> {fp_out}")
    for ds in (ds_t, ds_r, ds_v, ds_p):
        ds.close()
    return fp_out


log("Sampling per year and writing parts…")
part_paths = [sample_year(y) for y in years]

# ----------------- concat all parts -----------------
log("Concatenating yearly parts…")
feat = pd.concat([pd.read_parquet(p) for p in sorted(part_paths)], ignore_index=True)
feat = feat.sort_values(["hex_id", "date"]).reset_index(drop=True)
log(f"All rows: {len(feat):,}")

# -------- memory downcast early to avoid OOM --------
feat["hex_id"] = feat["hex_id"].astype(np.int32)
num_cols = feat.select_dtypes(include=["float", "int", "bool"]).columns.tolist()
if num_cols:
    feat[num_cols] = feat[num_cols].apply(pd.to_numeric, downcast="float")
    feat[num_cols] = feat[num_cols].apply(pd.to_numeric, downcast="integer")
log(f"Memory after downcast ~{feat.memory_usage(deep=True).sum()/1e9:.2f} GB")

# ----------------- engineer features ----------------
log("Adding rolling weather features…")
feat = add_rollups(feat, windows=tuple(CFG["features"].get("agg_windows_days", [1, 3, 7])))

log("Adding VPD rollups…")
g = feat.groupby("hex_id", group_keys=False)
for w in (1, 3, 7):
    feat[f"vpd_max_{w}d"] = g["vpd"].apply(lambda s: s.rolling(w, min_periods=1).max())

log("Adding seasonality…")
feat["doy"] = feat["date"].dt.dayofyear
feat["sin_doy"] = np.sin(2 * np.pi * feat["doy"] / 365.25)
feat["cos_doy"] = np.cos(2 * np.pi * feat["doy"] / 365.25)
feat["month"] = feat["date"].dt.month.astype(int)
feat = pd.get_dummies(feat, columns=["month"], prefix="m", drop_first=False)

log("Adding drought proxies…")
feat["precip_sum_7d"] = g["precip"].apply(lambda s: s.rolling(7, min_periods=1).sum())
feat["precip_sum_30d"] = g["precip"].apply(lambda s: s.rolling(30, min_periods=1).sum())
med30 = feat.groupby("hex_id")["precip_sum_30d"].transform("median")
feat["precip_30d_deficit"] = (med30 - feat["precip_sum_30d"]).clip(lower=0)

# --------- Lagged FIRMS (memory-safe join) ----------
counts_path = PROC / "firms_counts.parquet"
if counts_path.exists():
    log("Merging lagged FIRMS features (memory-safe)…")
    counts = pd.read_parquet(counts_path, columns=["hex_id", "date", "fires"])
    counts["hex_id"] = counts["hex_id"].astype(np.int32)
    counts["date"] = pd.to_datetime(counts["date"]).dt.normalize()

    counts = counts.sort_values(["hex_id", "date"]).reset_index(drop=True)
    gg = counts.groupby("hex_id", group_keys=False)
    counts["fires_lag1"]  = gg["fires"].shift(1).fillna(0)
    counts["fires_last3"] = gg["fires_lag1"].rolling(3, min_periods=1).sum().reset_index(level=0, drop=True)
    counts["fires_last7"] = gg["fires_lag1"].rolling(7, min_periods=1).sum().reset_index(level=0, drop=True)

    counts = counts[["hex_id", "date", "fires_last3", "fires_last7"]].copy()
    counts["fires_last3"] = pd.to_numeric(counts["fires_last3"], downcast="integer").astype(np.int16)
    counts["fires_last7"] = pd.to_numeric(counts["fires_last7"], downcast="integer").astype(np.int16)

    # MultiIndex alignment -> assign columns without creating a huge merged copy
    feat_indexed = feat.set_index(["hex_id", "date"], drop=False)
    counts_indexed = counts.set_index(["hex_id", "date"])
    feat["fires_last3"] = counts_indexed["fires_last3"].reindex(feat_indexed.index).to_numpy()
    feat["fires_last7"] = counts_indexed["fires_last7"].reindex(feat_indexed.index).to_numpy()
    feat[["fires_last3", "fires_last7"]] = feat[["fires_last3", "fires_last7"]].fillna(0)
    feat["fires_last3"] = feat["fires_last3"].astype(np.int16)
    feat["fires_last7"] = feat["fires_last7"].astype(np.int16)

    del counts, counts_indexed, feat_indexed
else:
    log("firms_counts.parquet not found; filling lagged fire features with 0.")
    feat["fires_last3"] = np.int16(0)
    feat["fires_last7"] = np.int16(0)

# -------- final tighten before write & save ----------
num_cols = feat.select_dtypes(include=["float", "int", "bool"]).columns.tolist()
if num_cols:
    feat[num_cols] = feat[num_cols].apply(pd.to_numeric, downcast="float")
    feat[num_cols] = feat[num_cols].apply(pd.to_numeric, downcast="integer")
log(f"Memory before write ~{feat.memory_usage(deep=True).sum()/1e9:.2f} GB")

out_path = PROC / "features.parquet"
log(f"Writing {out_path} with {len(feat):,} rows and {feat.shape[1]} columns…")
feat.to_parquet(out_path, index=False)  # pyarrow available
log("Done.")
