"""
# src/app/streamlit_app.py
import json
from pathlib import Path

import branca.colormap as cm
import folium
import requests
import shapely.geometry as sg
import streamlit as st
from streamlit_autorefresh import st_autorefresh
from streamlit_folium import st_folium

# --- Paths & setup ---
root = Path(__file__).resolve().parents[2]
geo_path = root / "src" / "features" / "risk_predictions.geojson"
meta_path = root / "src" / "features" / "metadata.json"

st.set_page_config(page_title="CA Wildfire Risk (ML)", layout="wide")
st.title("California Wildfire Risk Map")
st.caption("Demo model on a hex grid. Swap in your own data pipeline for production use.")

# Auto-refresh every 5 minutes so the map stays fresh if predictions update
st_autorefresh(interval=5 * 60 * 1000, key="wf_auto")

# Optional metadata (last updated, model date)
if meta_path.exists():
    try:
        meta = json.loads(meta_path.read_text())
        st.caption(
            f"Last updated: {meta.get('last_updated','?')} "
            f"(for data date {meta.get('date','?')})"
        )
    except Exception:
        pass

# Sidebar controls
with st.sidebar:
    st.header("Controls")
    cmap_name = st.selectbox("Color scale", ["viridis", "plasma", "inferno"], index=0)
    thresh = st.slider("Min risk filter", 0.0, 1.0, 0.0, 0.01)
    st.divider()
    st.write("📦 Data source")
    st.code(str(geo_path))

# Load predictions GeoJSON
if not geo_path.exists():
    st.error(f"Predictions file not found: {geo_path}")
    st.stop()

with open(geo_path) as f:
    gj = json.load(f)

# Restrict hexes to California using a public US-states GeoJSON
US_STATES_URL = (
    "https://raw.githubusercontent.com/PublicaMundi/MappingAPI/master/data/geojson/us-states.json"
)
try:
    states_geo = requests.get(US_STATES_URL, timeout=10).json()
    ca_geom = next(
        sg.shape(feat["geometry"])
        for feat in states_geo["features"]
        if feat["properties"].get("name") == "California"
    )

    def in_ca(feat):
        try:
            return sg.shape(feat["geometry"]).centroid.within(ca_geom)
        except Exception:
            return False

    gj["features"] = [f for f in gj["features"] if in_ca(f)]
except Exception as e:
    st.warning("Could not fetch CA boundary; showing all features instead. Error: {}".format(e))

# Filter by risk threshold (assumes 0..1)
gj["features"] = [f for f in gj["features"] if f["properties"].get("risk", 0.0) >= thresh]

# Build the map
m = folium.Map(
    location=[36.8, -120.0],
    zoom_start=5,
    control_scale=True,
    tiles="cartodbpositron",
)

# Colormap
colormaps = {
    "viridis": cm.linear.viridis,
    "plasma": cm.linear.plasma,
    "inferno": cm.linear.inferno,
}
cmap_obj = colormaps[cmap_name].scale(0, 1).to_step(10)
cmap_obj.caption = "Predicted Risk (0–1)"

# Tooltip fields: only include what exists
# Most minimal predictions have: hex_id, risk, (optionally) date
candidate_fields = ["hex_id", "risk", "date"]
if gj["features"]:
    present_keys = set(gj["features"][0]["properties"].keys())
    tooltip_fields = [c for c in candidate_fields if c in present_keys]
else:
    tooltip_fields = ["hex_id", "risk"]

tooltip_aliases = [f.replace("_", " ").title() for f in tooltip_fields]


def style_fn(feat):
    r = float(feat["properties"].get("risk", 0.0))
    return {
        "fillColor": cmap_obj(r),
        "color": "#555555",
        "weight": 0.3,
        "fillOpacity": 0.80,
    }


folium.GeoJson(
    data=gj,
    name="Wildfire risk",
    style_function=style_fn,
    tooltip=folium.GeoJsonTooltip(
        fields=tooltip_fields,
        aliases=tooltip_aliases,
        sticky=False,
        localize=True,
    ),
    highlight_function=lambda f: {"weight": 1.0, "color": "#222"},
).add_to(m)

cmap_obj.add_to(m)
folium.LayerControl(collapsed=False).add_to(m)

st_folium(m, width=None, height=700, returned_objects=[])

# Download the (filtered) predictions
st.download_button(
    "Download predictions (GeoJSON)",
    data=json.dumps(gj),
    file_name="risk_predictions.geojson",
    mime="application/geo+json",
)
"""

import json
import os
from pathlib import Path

import branca.colormap as cm
import folium
import requests
import shapely.geometry as sg
import streamlit as st
from streamlit_autorefresh import st_autorefresh
from streamlit_folium import st_folium

# --- Paths & setup ---
root = Path(__file__).resolve().parents[2]
default_geo_path = root / "src" / "features" / "risk_predictions.geojson"
default_meta_path = root / "src" / "features" / "metadata.json"

# If set, app reads predictions from this URL instead of the local file
GEOJSON_URL = os.environ.get("GEOJSON_URL", "").strip()

st.set_page_config(page_title="CA Wildfire Risk (ML)", layout="wide")
st.title("California Wildfire Risk Map")
st.caption("Demo model on a hex grid. Swap in your own data pipeline for production use.")

# Auto-refresh every 5 minutes
st_autorefresh(interval=5 * 60 * 1000, key="wf_auto")

# Optional metadata
meta = None
if default_meta_path.exists():
    try:
        meta = json.loads(default_meta_path.read_text())
    except Exception:
        pass
if meta:
    st.caption(
        f"Last updated: {meta.get('last_updated','?')} (for data date {meta.get('date','?')})"
    )

# Sidebar
with st.sidebar:
    st.header("Controls")
    cmap_name = st.selectbox("Color scale", ["viridis", "plasma", "inferno"], index=0)
    thresh = st.slider("Min risk filter", 0.0, 1.0, 0.0, 0.01)
    st.divider()
    st.write("📦 Data source")
    st.code(GEOJSON_URL if GEOJSON_URL else str(default_geo_path))


# Load predictions (file or URL)
def load_geojson():
    if GEOJSON_URL:
        r = requests.get(GEOJSON_URL, timeout=20)
        r.raise_for_status()
        return r.json()
    if not default_geo_path.exists():
        st.error(f"Predictions file not found: {default_geo_path}")
        st.stop()
    return json.loads(default_geo_path.read_text())


gj = load_geojson()

# Restrict to California
US_STATES_URL = (
    "https://raw.githubusercontent.com/PublicaMundi/MappingAPI/master/data/geojson/us-states.json"
)
try:
    states_geo = requests.get(US_STATES_URL, timeout=10).json()
    ca_geom = next(
        sg.shape(feat["geometry"])
        for feat in states_geo["features"]
        if feat["properties"].get("name") == "California"
    )

    def in_ca(feat):
        try:
            return sg.shape(feat["geometry"]).centroid.within(ca_geom)
        except Exception:
            return False

    gj["features"] = [f for f in gj["features"] if in_ca(f)]
except Exception as e:
    st.warning("Could not fetch CA boundary; showing all features instead. Error: {}".format(e))

# Filter by risk
gj["features"] = [f for f in gj["features"] if f["properties"].get("risk", 0.0) >= thresh]

# Map
m = folium.Map(location=[36.8, -120.0], zoom_start=5, control_scale=True, tiles="cartodbpositron")
colormaps = {"viridis": cm.linear.viridis, "plasma": cm.linear.plasma, "inferno": cm.linear.inferno}
cmap_obj = colormaps[cmap_name].scale(0, 1).to_step(10)
cmap_obj.caption = "Predicted Risk (0–1)"

# Tooltip fields (robust)
candidate_fields = ["hex_id", "risk", "date"]
tooltip_fields = candidate_fields
if gj["features"]:
    present_keys = set(gj["features"][0]["properties"].keys())
    tooltip_fields = [c for c in candidate_fields if c in present_keys]
tooltip_aliases = [f.replace("_", " ").title() for f in tooltip_fields]


def style_fn(feat):
    r = float(feat["properties"].get("risk", 0.0))
    return {"fillColor": cmap_obj(r), "color": "#555555", "weight": 0.3, "fillOpacity": 0.80}


folium.GeoJson(
    data=gj,
    name="Wildfire risk",
    style_function=style_fn,
    tooltip=folium.GeoJsonTooltip(
        fields=tooltip_fields, aliases=tooltip_aliases, sticky=False, localize=True
    ),
    highlight_function=lambda f: {"weight": 1.0, "color": "#222"},
).add_to(m)
cmap_obj.add_to(m)
folium.LayerControl(collapsed=False).add_to(m)
st_folium(m, width=None, height=700, returned_objects=[])

# Download (filtered) predictions
st.download_button(
    "Download predictions (GeoJSON)",
    data=json.dumps(gj),
    file_name="risk_predictions.geojson",
    mime="application/geo+json",
)
