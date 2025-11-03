import streamlit as st
from plot_real import run_constellation_pipeline
import pandas as pd

st.title("Constellation Graph Explorer")
st.set_page_config(
    page_title="Constellation Graph Explorer",
    layout="wide",               # 🟢 makes the page take the full width
    initial_sidebar_state="expanded"
)

# Answers to MVP meeting
st.markdown("---")
st.subheader("Issues")
st.markdown("""
- Visibility filter can empty dataset at certain lat/lon/time combinations.  
- RA 0° wrapping can clip bounding boxes.  
- Two-pass algorithm increases computation time slightly.
""")

st.subheader("Next Steps")
st.markdown("""
- Add all-sky projection view.  
- Optimize runtime using cached constellations.  
- Enable draggable user masks for dynamic reruns.
""")

with st.sidebar:
    st.header("Controls")
    csv = st.file_uploader("Upload star CSV", type="csv")
    top_n = st.number_input("Top N stars", 100, 10000, 1200, step=100)
    lat = st.number_input("Latitude (°)", -90.0, 90.0, 39.77)
    lon = st.number_input("Longitude (°)", -180.0, 180.0, -86.16)
    elev = st.number_input("Elevation (m)", 0, 4000, 220)
    utc_time = st.text_input("UTC Time", "2025-11-06T02:00:00")
    mask_pad = st.slider("Mask padding (°)", 1.0, 15.0, 5.0)
    apply_vis = st.checkbox("Apply visibility filter", value=True)
    render = st.button("Render sky plot")

if render and csv:
    df = pd.read_csv(csv)
    fig, df_keep, masks, stats = run_constellation_pipeline(
        csv_path=df, top_n=top_n, apply_visibility=apply_vis,
        lat=lat, lon=lon, elev_m=elev, utc_time=utc_time,
        mask_pad_deg=mask_pad)

    st.subheader("Sky Plot")
    st.pyplot(fig)

    st.subheader("Summary Statistics")
    st.json(stats)

    st.subheader("Foreground Stars (sample)")
    st.dataframe(df_keep.head())

    st.download_button("Download Kept Stars", df_keep.to_csv(index=False), "kept_stars.csv")
