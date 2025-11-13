import os
from datetime import datetime
from pathlib import Path

import requests
import yaml

CFG = yaml.safe_load(open("src/config.yaml"))

start_str = os.getenv("START", CFG["history"]["start"])
end_str = os.getenv("END", CFG["history"]["end"])
START = datetime.fromisoformat(start_str)
END = datetime.fromisoformat(end_str)

VARS = ["tmmx", "rmin", "vs", "pr"]

OUTDIR = Path("data/raw/gridmet")
OUTDIR.mkdir(parents=True, exist_ok=True)

BASE = "https://www.northwestknowledge.net/metdata/data"


def url_for(var: str, year: int) -> str:
    return f"{BASE}/{var}_{year}.nc"


def download_file(url: str, path: Path):
    path_tmp = path.with_suffix(path.suffix + ".part")
    with requests.get(url, stream=True, timeout=60) as r:
        if r.status_code != 200:
            raise RuntimeError(f"{url} -> HTTP {r.status_code}")
        r.raise_for_status()
        with open(path_tmp, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
    path_tmp.replace(path)


for yr in range(START.year, END.year + 1):
    for var in VARS:
        out = OUTDIR / f"{var}_{yr}.nc"
        if out.exists() and out.stat().st_size > 0:
            print("Skip (exists):", out.name)
            continue
        url = url_for(var, yr)
        try:
            print("Downloading:", url)
            download_file(url, out)
            print("Saved:", out)
        except Exception as e:
            print(f"Warning: failed {var} {yr} -> {e}")

print("Done.")
