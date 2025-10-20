import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import average_precision_score, roc_auc_score

# ---- Try XGBoost; fall back to GBDT if needed
USE_XGB = True
try:
    from xgboost import XGBClassifier  # type: ignore
    from xgboost.callback import EarlyStopping  # callback API works across versions
except Exception as e:
    print(
        "XGBoost unavailable; falling back to GradientBoostingClassifier. Reason:",
        e,
        file=sys.stderr,
    )
    from sklearn.ensemble import GradientBoostingClassifier

    USE_XGB = False

CFG = yaml.safe_load(open("src/config.yaml"))
PROC = Path("data/processed")
ART = Path("src/models")
ART.mkdir(parents=True, exist_ok=True)

# --------------------- load data ---------------------
feat = pd.read_parquet(PROC / "features.parquet")
lab = pd.read_parquet(PROC / "labels.parquet")

df = feat.merge(lab, on=["hex_id", "date"], how="inner").sort_values(["date", "hex_id"])
df["date"] = pd.to_datetime(df["date"]).dt.normalize()

# ----------------- recent window filter --------------
recent_years = int(CFG.get("train", {}).get("recent_years", 4))
max_day = df["date"].max()
cutoff = max_day - pd.Timedelta(days=int(365.25 * recent_years))
df_recent = df[df["date"] >= cutoff].copy()

# --------------------- split by day ------------------
dates = df_recent["date"].drop_duplicates().sort_values().to_list()
split_idx = max(1, int(round(len(dates) * 0.8)))
train_days = set(dates[:split_idx])
valid_days = set(dates[split_idx:]) or {dates[-1]}

train = df_recent[df_recent["date"].isin(train_days)].copy()
valid = df_recent[df_recent["date"].isin(valid_days)].copy()

print(
    f"Training window: {cutoff.date()} → {max_day.date()}  ({recent_years}y). "
    f"Train rows: {len(train):,}  Valid rows: {len(valid):,}"
)
print("Base rates  train={:.4f}  valid={:.4f}".format(train["y"].mean(), valid["y"].mean()))

# -------------------- build matrices -----------------
feat_cols = [c for c in df_recent.columns if c not in ["hex_id", "date", "y"]]
Xtr, ytr = train[feat_cols].copy(), train["y"].astype(int)
Xva, yva = valid[feat_cols].copy(), valid["y"].astype(int)

# Drop any non-numeric artifacts
for junk in ["cell", "point"]:
    if junk in Xtr.columns:
        print(f"Dropping stray column: {junk}")
        Xtr.drop(columns=[junk], inplace=True, errors="ignore")
        Xva.drop(columns=[junk], inplace=True, errors="ignore")
num_cols = Xtr.select_dtypes(include=["number", "bool"]).columns.tolist()
drop_cols = sorted(set(Xtr.columns) - set(num_cols))
if drop_cols:
    print("Dropping non-numeric columns:", drop_cols)
Xtr = Xtr[num_cols]
Xva = Xva[num_cols]

# ---------------------- model ------------------------
if USE_XGB:
    pos = ytr.sum()
    neg = len(ytr) - pos
    scale_pos = (neg / max(1, pos)) if pos > 0 else 1.0

    model = XGBClassifier(
        n_estimators=1200,
        max_depth=7,
        min_child_weight=6,
        learning_rate=0.05,
        subsample=0.7,
        colsample_bytree=0.7,
        objective="binary:logistic",
        eval_metric="aucpr",
        scale_pos_weight=scale_pos,
        tree_method="hist",
        n_jobs=4,
        random_state=42,
    )

    # ---- Fit with best-available early stopping API; otherwise plain fit
    fitted = False
    # 1) Newer API
    try:
        model.fit(
            Xtr,
            ytr,
            eval_set=[(Xva, yva)],
            early_stopping_rounds=100,
            verbose=False,
        )
        fitted = True
    except TypeError:
        pass
    # 2) Callback API (some mid versions)
    if not fitted:
        try:
            from xgboost.callback import EarlyStopping

            model.fit(
                Xtr,
                ytr,
                eval_set=[(Xva, yva)],
                verbose=False,
                callbacks=[
                    EarlyStopping(rounds=100, metric_name="aucpr", save_best=True, maximize=True)
                ],
            )
            fitted = True
        except Exception:
            pass
    # 3) Fallback: no early stopping
    if not fitted:
        model.fit(Xtr, ytr)
else:
    from sklearn.ensemble import GradientBoostingClassifier

    model = GradientBoostingClassifier(
        n_estimators=600, learning_rate=0.05, subsample=0.8, random_state=42
    )
    model.fit(Xtr, ytr)

# ------------------- validation metrics --------------
if hasattr(model, "predict_proba"):
    pva = model.predict_proba(Xva)[:, 1]
else:
    margin = model.decision_function(Xva)
    pva = 1 / (1 + np.exp(-margin))

auc = roc_auc_score(yva, pva) if yva.nunique() == 2 else float("nan")
prauc = average_precision_score(yva, pva) if yva.nunique() == 2 else float("nan")
print(f"Val ROC-AUC: {auc:.3f}  PR-AUC: {prauc:.3f}")

# ---------------- margin→score scaler (0–1) ----------
if USE_XGB:
    train_margin = model.predict(Xtr, output_margin=True)
    valid_margin = model.predict(Xva, output_margin=True)
else:
    train_margin = model.decision_function(Xtr)
    valid_margin = model.decision_function(Xva)

q_lo, q_hi = np.percentile(train_margin, [1, 99])


def margin_to_score(m):
    return np.clip((m - q_lo) / max(1e-9, (q_hi - q_lo)), 0.0, 1.0)


score_va = margin_to_score(valid_margin)
print(
    "Validation score summary (0–1 scaled): min={:.3f} p50={:.3f} p90={:.3f} max={:.3f}".format(
        score_va.min(), np.median(score_va), np.percentile(score_va, 90), score_va.max()
    )
)

# ---------------------- persist ----------------------
bundle = {
    "model": model,
    "features": num_cols,
    "scaler": {"q_lo": float(q_lo), "q_hi": float(q_hi)},
    "use_xgb": USE_XGB,
    "train_window": {
        "start": str(cutoff.date()),
        "end": str(max_day.date()),
        "recent_years": recent_years,
    },
}
joblib.dump(bundle, ART / "model.pkl")
print("Saved", ART / "model.pkl")
