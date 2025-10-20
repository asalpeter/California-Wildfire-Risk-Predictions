import io
import os
import time
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import requests
import yaml

CFG = yaml.safe_load(open("src/config.yaml"))
RAW = Path("data/raw/firms")
RAW.mkdir(parents=True, exist_ok=True)

MAP_KEY = os.getenv("FIRMS_MAP_KEY")
if not MAP_KEY:
    raise RuntimeError("Set FIRMS_MAP_KEY in GitHub Secrets or env")

# --- Sources: use SP for historical ---
SOURCES = ["VIIRS_SNPP_SP"]  # add more: "VIIRS_NOAA20_SP", "VIIRS_NOAA21_SP"

# California bbox (W, S, E, N) — slightly padded
minx, miny, maxx, maxy = -124.6, 32.4, -114.0, 42.1
AREA = f"{minx},{miny},{maxx},{maxy}"

start = datetime.fromisoformat(CFG["history"]["start"])
end = datetime.fromisoformat(CFG["history"]["end"])

DAY_RANGE = 10  # max allowed by FIRMS


def fetch_window(src: str, d: datetime) -> pd.DataFrame:
    day = d.strftime("%Y-%m-%d")
    url = f"https://firms.modaps.eosdis.nasa.gov/api/area/csv/{MAP_KEY}/{src}/{AREA}/{DAY_RANGE}/{day}"
    r = requests.get(url, timeout=60)
    if r.status_code != 200 or not r.text.strip():
        return pd.DataFrame()
    df = pd.read_csv(io.StringIO(r.text))
    df.columns = [c.lower() for c in df.columns]
    # coerce date
    if "acq_date" in df.columns:
        df["date"] = pd.to_datetime(df["acq_date"])
    return df


def window_path(d: datetime, end_d: datetime) -> Path:
    return RAW / f"firms_{d:%Y%m%d}_{end_d:%Y%m%d}.csv"


all_frames = []
d = start
while d <= end:
    window_end = min(d + timedelta(days=DAY_RANGE - 1), end)
    out = window_path(d, window_end)

    # Skip if cached
    if out.exists() and out.stat().st_size > 0:
        print(f"Skip (exists): {out.name}", flush=True)
        d += timedelta(days=DAY_RANGE)
        continue

    window_frames = []
    for src in SOURCES:
        df = fetch_window(src, d)
        if not df.empty:
            df["source"] = src
            window_frames.append(df)

    if window_frames:
        win = pd.concat(window_frames, ignore_index=True)
        out.write_text(win.to_csv(index=False))
        print(f"Downloaded {out.name} rows={len(win)} from {', '.join(SOURCES)}", flush=True)
        all_frames.append(win)
    else:
        print(f"No data for window {d:%Y-%m-%d}..{window_end:%Y-%m-%d}", flush=True)

    time.sleep(0.8)  # be gentle to the API
    d += timedelta(days=DAY_RANGE)

print("Done. Windows with data:", len(all_frames), flush=True)
