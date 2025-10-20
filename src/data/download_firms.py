import io
import os
import time
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import requests
import yaml
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

CFG = yaml.safe_load(open("src/config.yaml"))
RAW = Path("data/raw/firms")
RAW.mkdir(parents=True, exist_ok=True)

MAP_KEY = os.getenv("FIRMS_MAP_KEY")
if not MAP_KEY:
    raise RuntimeError("Set FIRMS_MAP_KEY in GitHub Secrets or env")

# --- Sources: use SP for historical ---
SOURCES = ["VIIRS_SNPP_SP"]  # optionally add: "VIIRS_NOAA20_SP", "VIIRS_NOAA21_SP"

# California bbox (W, S, E, N) — slightly padded
minx, miny, maxx, maxy = -124.6, 32.4, -114.0, 42.1
AREA = f"{minx},{miny},{maxx},{maxy}"

start_str = os.getenv("START", CFG["history"]["start"])
end_str = os.getenv("END", CFG["history"]["end"])
start = datetime.fromisoformat(start_str)
end = datetime.fromisoformat(end_str)

DAY_RANGE = 10  # max allowed by FIRMS

# --- Robust HTTP session with retries ---
def make_session() -> requests.Session:
    sess = requests.Session()
    retry = Retry(
        total=6,
        connect=6,
        read=6,
        backoff_factor=1.5,  # 0s, 1.5s, 3s, 4.5s, 6s, 7.5s ...
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
        raise_on_status=False,
        respect_retry_after_header=True,
    )
    adapter = HTTPAdapter(max_retries=retry, pool_connections=10, pool_maxsize=10)
    sess.mount("https://", adapter)
    sess.headers.update({"User-Agent": "wildfire-pipeline/1.0 (+github-actions)"})
    return sess

SESSION = make_session()

def fetch_window(src: str, d: datetime) -> pd.DataFrame:
    day = d.strftime("%Y-%m-%d")
    url = f"https://firms.modaps.eosdis.nasa.gov/api/area/csv/{MAP_KEY}/{src}/{AREA}/{DAY_RANGE}/{day}"
    try:
        # Longer timeout because files can be chunky
        r = SESSION.get(url, timeout=180)
        if r.status_code != 200 or not r.text.strip():
            print(f"Warn: HTTP {r.status_code} or empty body for {url}", flush=True)
            return pd.DataFrame()
        df = pd.read_csv(io.StringIO(r.text))
        df.columns = [c.lower() for c in df.columns]
        if "acq_date" in df.columns:
            df["date"] = pd.to_datetime(df["acq_date"])
        return df
    except requests.RequestException as e:
        print(f"Warn: exception fetching {url}: {e}", flush=True)
        return pd.DataFrame()

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

    # Be gentle to the API
    time.sleep(0.8)
    d += timedelta(days=DAY_RANGE)

print("Done. Windows with data:", len(all_frames), flush=True)
