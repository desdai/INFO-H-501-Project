import streamlit as st
from plot_real import run_constellation_pipeline
import pandas as pd
import os

# ------------------------------------------------------------
# Page Configuration
# ------------------------------------------------------------
st.set_page_config(
    page_title="Constellation Graph Explorer",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ------------------------------------------------------------
# Custom CSS (larger font, nicer spacing)
# ------------------------------------------------------------
st.markdown("""
<style>
/* MAIN CONTENT FONT SIZE */
html, body, .block-container {
    font-size: 19px !important;
    line-height: 1.5 !important;
}

/* TITLES */
h1 {
    font-size: 40px !important;
    font-weight: 800 !important;
}
h2 {
    font-size: 30px !important;
    font-weight: 700 !important;
}
h3 {
    font-size: 24px !important;
    font-weight: 650 !important;
}

/* TAB LABEL FONT */
.stTabs [data-baseweb="tab"] {
    font-size: 20px !important;
    padding: 10px 20px !important;
}

/* TABLE FONT */
.dataframe {
    font-size: 18px !important;
}

/* Sidebar still slightly smaller for hierarchy */
.sidebar .block-container {
    font-size: 17px !important;
}
</style>
""", unsafe_allow_html=True)


# ------------------------------------------------------------
# Title + Description
# ------------------------------------------------------------
st.title("✨ Constellation Graph Explorer")

st.markdown("""
Welcome to the **Constellation Graph Explorer**, an interactive tool designed to help you
visualize the night sky using real astronomical data.  
This app reconstructs **constellation graphs** from star catalogs and allows you to explore
how the visible sky changes with:

- **Location** (latitude, longitude, elevation)  
- **Time** (UTC-based sky rotation)  
- **Brightness filtering**  
- **Visibility constraints** (altitude above horizon)

---

## 🧭 How to Use the Interface

### 1. Upload Your Star CSV  
In the **left sidebar**, upload a CSV containing star information (RA, Dec, RGB, magnitude/size).

### 2. Adjust Viewing Parameters  
Use the sidebar to modify:

- **Latitude / Longitude / Elevation** → simulates observing from different places on Earth  
- **UTC Time** → rotates the sky to the correct sidereal position  
- **Top N stars** → controls visual clutter  
- **Mask padding** → expands/shrinks constellation boundaries  
- **Visibility filter** → hides stars below your horizon

All updates apply **instantly**.

---

## 🖼️ Output Tabs

### 🌌 Sky Plot  
Shows the reconstructed constellation graph using:
- kNN structure  
- length-based pruning  
- bounding-box filtering  
- largest connected component detection  

### 📊 Data & Downloads  
Provides:
- Summary statistics  
- Foreground (kept) stars  
- Downloadable CSV  

### 🎬 Animation Gallery  
Generated time-lapse animations to illustrate:
- Earth rotation  
- Changing latitude  
- Changing longitude  
- Elevation-based brightness effects  

---

This system helps bridge astronomy, data processing, and visualization — making it an excellent tool for learning about real sky structures.
""")

st.markdown("---")

# ------------------------------------------------------------
# Sidebar Controls
# ------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Controls")

    st.markdown("Upload your own star dataset, or download the sample below:")

    # ---- Sample CSV Download ----
    sample_path = "stars_plot_ready.csv"
    st.caption("This sample dataset contains ~2000 real stars.")

    if os.path.exists(sample_path):
        with open(sample_path, "rb") as f:
            st.download_button(
                label="📥 Download sample star catalog (2000 stars)",
                data=f,
                file_name="stars_plot_ready.csv",
                mime="text/csv",
                key="sample_download"
            )
    else:
        st.warning("Sample dataset not found on server.")

    # ---- User CSV Upload ----
    csv = st.file_uploader("Upload a star CSV file", type="csv")

    st.markdown("### Sky Parameters")
    lat = st.number_input("Latitude (°)", -90.0, 90.0, 39.70)
    lon = st.number_input("Longitude (°)", -180.0, 180.0, 30.00)
    elev = st.number_input("Elevation (m)", 0, 4000, 220)
    utc_time = st.text_input("UTC Time", "2025-11-06T02:00:00")

    st.markdown("### Constellation Reconstruction")
    top_n = st.number_input("Top N brightest stars", 100, 2000, 1200, step=100)
    mask_pad = st.slider("Mask padding (°)", 1.0, 15.0, 3.0)

    apply_vis = st.checkbox("Apply visibility (alt > 0°)", value=True)

st.markdown("---")


# ------------------------------------------------------------
# Main Contents
# Tabs: Sky • Data • Gallery
# ------------------------------------------------------------
tabs = st.tabs(["🌌 Sky Plot", "📊 Data & Downloads", "🎬 Animation Gallery"])

# ================
#   If file uploaded
# ================
if csv:
    df = pd.read_csv(csv)

    # Run pipeline
    try:
        fig, df_keep, masks, stats = run_constellation_pipeline(
            csv_path=df,
            top_n=top_n,
            apply_visibility=apply_vis,
            lat=lat,
            lon=lon,
            elev_m=elev,
            utc_time=utc_time,
            draw_masks=False,
            mask_pad_deg=mask_pad
        )
    except Exception as e:
        st.error(f"❌ Pipeline error: {e}")
        st.stop()

    # ===========================================
    # TAB 1 — SKY PLOT
    # ===========================================
    with tabs[0]:
        st.subheader("🌌 Sky Projection (RA/Dec)")
        st.caption("Constellations reconstructed using kNN + bounding-box filtering. RA axis inverted (telescope view).")
        st.pyplot(fig, width="stretch")

    # ===========================================
    # TAB 2 — DATA & DOWNLOADS
    # ===========================================
    with tabs[1]:
        st.subheader("📊 Summary Statistics")
        st.json(stats)

        st.subheader("📁 Foreground Stars (sample)")
        st.dataframe(df_keep.head(), width="stretch")

        st.download_button(
            "⬇️ Download Processed Stars (CSV)",
            df_keep.to_csv(index=False),
            "kept_stars.csv"
        )

    # ===========================================
    # TAB 3 — ANIMATION GALLERY
    # ===========================================
    with tabs[2]:
        st.subheader("🎬 Sky Animations")
        st.caption("Pre-generated using the same reconstruction pipeline.")

        col1, col2 = st.columns(2)
        col3, col4 = st.columns(2)
        col5, col6 = st.columns(2)

        def embed_anim(path, label, column):
            if os.path.exists(path):
                column.markdown(f"**{label}**")
                column.video(path)
            else:
                column.info(f"{label}: animation file missing.")


        embed_anim("frames_time/time_animation.mp4", "Time-Lapse Rotation", col1)
        embed_anim("frames_lat/lat_animation.mp4", "Latitude Sweep", col2)
        embed_anim("frames_lon/lon_animation.mp4", "Longitude Sweep", col3)
        embed_anim("frames_elev/elev_animation.mp4", "Elevation Sweep", col4)
        embed_anim("frames_topN/topN_animation.mp4", "Top-N Bright Stars Sweep", col5)
        embed_anim("frames_maskpad/maskpad_animation.mp4", "Mask Padding Sweep", col6)


else:
    st.info("⬅️ Upload a CSV file in the sidebar to begin.")
