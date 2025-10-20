SHELL := /bin/bash
.PHONY: help app data-backfill gridmet build-hexgrid firms static-topo human make-features make-labels train predict

help:
	@echo "make data-backfill      # hexes, FIRMS, GridMET, static topo/human, labels, features"
	@echo "make build-hexgrid      # build hexgrid_ca.geojson"
	@echo "make firms              # download FIRMS detections (uses FIRMS_MAP_KEY from .env)"
	@echo "make gridmet            # download GridMET weather (START/END optional)"
	@echo "make static-topo        # compute elevation/slope/aspect per hex (one-time/occasional)"
	@echo "make human              # compute distance-to-roads/urban per hex (one-time/occasional)"
	@echo "make make-labels        # build labels.parquet from FIRMS and hexes"
	@echo "make make-features      # build features.parquet (GridMET + VPD + statics)"
	@echo "make train              # train model (saves src/models/model.pkl)"
	@echo "make predict            # export risk_predictions.geojson + metadata.json"
	@echo "make app                # run streamlit app"

# --- Atomic steps ---
build-hexgrid:
	python src/data/build_hexgrid.py

firms:
	python src/data/download_firms.py

gridmet:
	python src/data/download_gridmet.py

static-topo:
	python src/data/build_topography.py

human:
	python src/data/build_human.py

make-labels:
	python src/data/make_labels.py

make-features:
	python src/data/make_features.py

train:
	python src/data/train.py

predict:
	python src/data/predict.py

app:
	streamlit run src/app/streamlit_app.py

# --- Orchestrated backfill (idempotent; safe to re-run) ---
data-backfill: build-hexgrid firms gridmet static-topo human make-labels make-features
