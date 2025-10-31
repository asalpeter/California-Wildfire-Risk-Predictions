# California Wildfire Risk ML App

A machine learning pipeline and Streamlit web app that predicts and visualizes **wildfire risk across California** using environmental and satellite data.  
The system automates daily retraining and updates through GitHub Actions and deploys interactively via Streamlit Cloud.

---

## Overview

The app combines automated data ingestion, model training, and visualization to produce dynamic wildfire risk forecasts.  
Predictions are shown on an interactive **hex-grid map** built with **Folium**, where each cell represents localized risk levels.

- **Data Sources:** NASA FIRMS (fire detections), GridMET (meteorology), USGS DEM (terrain)  
- **Model:** Gradient-boosted decision trees using **XGBoost**  
- **Performance:** PR-AUC 0.628 (~70× baseline 0.009), ROC-AUC 0.945  
- **Automation:** Daily rebuild via GitHub Actions  
- **Deployment:** Streamlit Cloud auto-updates after each rebuild

---

## Features

- End-to-end ML pipeline (data → features → model → predictions)  
- Daily retraining and prediction refresh through GitHub Actions  
- Interactive geospatial visualization with adjustable risk thresholds  
- Data caching for faster rebuilds and low-latency map rendering  

---

## Tech Stack

| Component | Technologies |
|------------|---------------|
| **Frontend / Visualization** | Streamlit, Folium, streamlit-folium |
| **Modeling** | XGBoost, NumPy, pandas, scikit-learn |
| **Geospatial / Data** | GeoPandas, xarray, PyArrow, Shapely |
| **Automation / CI** | GitHub Actions |
| **Hosting** | Streamlit Cloud |

---

## Quick Start

### 1. Clone and set up
```bash
git clone https://github.com/<your-username>/california-wildfire-risk-predictions.git
cd california-wildfire-risk-predictions
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-pipeline.txt
```
### 2. Run the pipeline
```bash
python src/data/download_firms.py
python src/data/download_gridmet.py
python src/data/make_labels.py
python src/data/make_features.py
python src/data/train.py
python src/data/predict.py
```

### 3. Launch the app
```bash
streamlit run src/app/streamlit_app.py
```

---

## Automation & Deployment

A GitHub Actions workflow (.github/workflows/rebuild.yml) runs daily at 10 UTC to:
 - Refresh datasets
 - Retrain the XGBoost model
 - Export new GeoJSON predictions
 - Commit updated artifacts back to the repo
The connected Streamlit Cloud app automatically redeploys using the latest predictions.

---

## Data Sources
 - NASA FIRMS: Fire detections (MODIS / VIIRS)
 - GridMET: Daily meteorological data
 - USGS DEM: Elevation and terrain features
 - California Open Data Portal: Administrative boundaries
