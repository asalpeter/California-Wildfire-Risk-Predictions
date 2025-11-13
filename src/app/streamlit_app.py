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

root = Path(__file__).resolve().parents[2]
default_geo_path = root / "src" / "features" / "risk_predictions.geojson"
default_meta_path = root / "src" / "features" / "metadata.json"

GEOJSON_URL = os.environ.get("GEOJSON_URL", "").strip()

st.set_page_config(page_title="California Wildfire Risk", layout="wide")
st.title("🔥 California Wildfire Risk Map")

st_autorefresh(interval=5 * 60 * 1000, key="wf_auto")

meta = None
if default_meta_path.exists():
    try:
        meta = json.loads(default_meta_path.read_text())
    except Exception:
        pass
if meta:
    st.caption(
        f"Last updated: {meta.get('last_updated','?')} | Data date: {meta.get('date','?')}"
    )

with st.sidebar:
    st.header("Map Controls")
    cmap_name = st.selectbox("Color scale", ["viridis", "plasma", "inferno"], index=0)
    thresh = st.slider("Minimum risk threshold", 0.0, 1.0, 0.0, 0.01)
    st.divider()
    st.caption("📊 Data Source")
    st.code(GEOJSON_URL if GEOJSON_URL else str(default_geo_path))


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

US_STATES_URL = (
    "https://raw.githubusercontent.com/PublicaMundi/MappingAPI/master/data/geojson/us-states.json"
)
try:
    states_geo = requests.get(US_STATES_URL, timeout=10).json()
    ca_geom = next(
        sg.shape(f["geometry"])
        for f in states_geo["features"]
        if f["properties"].get("name") == "California"
    )

    def in_ca(f):
        try:
            return sg.shape(f["geometry"]).centroid.within(ca_geom)
        except Exception:
            return False

    gj["features"] = [f for f in gj["features"] if in_ca(f)]
except Exception as e:
    st.warning(f"Could not fetch California boundary — showing all features. ({e})")

gj["features"] = [
    f for f in gj["features"] if float(f["properties"].get("risk", 0.0)) >= thresh
]

if not gj["features"]:
    st.warning(
        "No hexes match the current risk threshold. Try lowering the threshold in the sidebar."
    )
    m = folium.Map(
        location=[36.8, -120.0],
        zoom_start=5,
        control_scale=True,
        tiles="cartodbpositron",
    )
    st_folium(m, width=None, height=750, returned_objects=[])
    st.download_button(
        label="⬇️ Download current predictions (GeoJSON)",
        data=json.dumps(gj),
        file_name="wildfire_risk.geojson",
        mime="application/geo+json",
    )
    st.stop()

m = folium.Map(
    location=[36.8, -120.0], zoom_start=5, control_scale=True, tiles="cartodbpositron"
)
colormaps = {
    "viridis": cm.linear.viridis,
    "plasma": cm.linear.plasma,
    "inferno": cm.linear.inferno,
}
cmap_obj = colormaps[cmap_name].scale(0, 1).to_step(10)
cmap_obj.caption = "Predicted Wildfire Risk (0–1)"

candidate_fields = ["hex_id", "risk", "date"]
present_keys = set(gj["features"][0]["properties"].keys())
tooltip_fields = [c for c in candidate_fields if c in present_keys]
tooltip_aliases = [f.replace("_", " ").title() for f in tooltip_fields]


def style_fn(f):
    r = float(f["properties"].get("risk", 0.0))
    return {
        "fillColor": cmap_obj(r),
        "color": "#555555",
        "weight": 0.3,
        "fillOpacity": 0.85,
    }


folium.GeoJson(
    data=gj,
    name="Wildfire Risk",
    style_function=style_fn,
    tooltip=folium.GeoJsonTooltip(
        fields=tooltip_fields,
        aliases=tooltip_aliases,
        sticky=False,
        localize=True,
    ),
    highlight_function=lambda f: {"weight": 1.2, "color": "#222"},
).add_to(m)
cmap_obj.add_to(m)
folium.LayerControl(collapsed=False).add_to(m)

st_folium(m, width=None, height=750, returned_objects=[])

st.download_button(
    label="⬇️ Download current predictions (GeoJSON)",
    data=json.dumps(gj),
    file_name="wildfire_risk.geojson",
    mime="application/geo+json",
)
