# src/data/predict.py
import json
from pathlib import Path

import geopandas as gpd
import joblib
import numpy as np
import pandas as pd

ART = Path("src/models")
PROC = Path("data/processed")
HEX = Path("src/features/hexgrid_ca.geojson")
OUT_GJ = Path("src/features/risk_predictions.geojson")
OUT_META = Path("src/features/metadata.json")

bundle = joblib.load(ART / "model.pkl")
model = bundle["model"]
feat_cols = bundle["features"]
q_lo = bundle["scaler"]["q_lo"]
q_hi = bundle["scaler"]["q_hi"]
use_xgb = bundle.get("use_xgb", True)

# Use the last date in features as "today's" score (or swap in your desired date)
feat = pd.read_parquet(PROC / "features.parquet")
last_date = feat["date"].max()
X = feat[feat["date"] == last_date].copy()

# keep only trained columns
X = X[feat_cols + ["hex_id"]]

# raw margins
if use_xgb:
    margin = model.predict(X[feat_cols], output_margin=True)
else:
    margin = model.decision_function(X[feat_cols])

# map margin -> [0,1] score (not a probability)
risk = np.clip((margin - q_lo) / max(1e-9, (q_hi - q_lo)), 0.0, 1.0)

df = pd.DataFrame({"hex_id": X["hex_id"].values, "risk": risk, "date": last_date})
gdf = gpd.read_file(HEX)[["hex_id", "geometry"]].merge(df, on="hex_id", how="left")

gdf.to_file(OUT_GJ, driver="GeoJSON")
OUT_META.write_text(
    json.dumps({"last_updated": pd.Timestamp.utcnow().isoformat(), "date": str(last_date)})
)
print("Wrote predictions ->", OUT_GJ)
