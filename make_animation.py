#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
from datetime import datetime, timedelta
import pandas as pd
from pathlib import Path

import imageio
import imageio.v3 as iio

from plot_real import run_constellation_pipeline


# ============================================================
#  CLASS FOR CONFIGURATION
# ============================================================
class AnimationConfig:
    """
    Holds configuration for animation sweeps.
    """
    def __init__(self, csv_path, top_n=1200, pad=3.0,
                 base_lat=39.7, base_lon=30, base_elev=220,
                 n_frames_general=60, n_frames_elev=20):

        self.csv_path = csv_path
        self.top_n = top_n
        self.pad = pad
        self.base_lat = base_lat
        self.base_lon = base_lon
        self.base_elev = base_elev

        self.n_frames_general = n_frames_general
        self.n_frames_elev = n_frames_elev

    def __repr__(self):
        return (
            f"AnimationConfig(csv={self.csv_path}, top_n={self.top_n}, "
            f"pad={self.pad}, lat={self.base_lat}, lon={self.base_lon}, "
            f"elev={self.base_elev}, frames_general={self.n_frames_general}, "
            f"frames_elev={self.n_frames_elev})"
        )


# ----------------------------------------
# Helper for frame naming
# ----------------------------------------
def frame_name(outdir, i):
    return outdir / f"frame_{i:04d}.png"


# ============================================================
#  TIME SWEEP
# ============================================================
def animate_time(df, cfg: AnimationConfig):
    outdir = Path("frames_time")
    outdir.mkdir(exist_ok=True)

    start = datetime(2025, 11, 6, 18, 0, 0)
    dt = timedelta(minutes=15)

    for i in range(cfg.n_frames_general):
        utc_str = (start + i * dt).strftime("%Y-%m-%dT%H:%M:%S")
        print(f"[time] Frame {i}, utc={utc_str}")

        fig, df_keep, masks, stats = run_constellation_pipeline(
            df,
            top_n=cfg.top_n,
            apply_visibility=True,
            lat=cfg.base_lat,
            lon=cfg.base_lon,
            elev_m=cfg.base_elev,
            utc_time=utc_str,
            mask_pad_deg=cfg.pad,
            draw_masks=False,
        )
        fig.savefig(frame_name(outdir, i), dpi=200, bbox_inches="tight")
        fig.clf()

    build_gif_mp4(outdir, "time")


# ============================================================
#  LATITUDE SWEEP
# ============================================================
def animate_lat(df, cfg: AnimationConfig):
    outdir = Path("frames_lat")
    outdir.mkdir(exist_ok=True)

    lats = np.linspace(-60, 60, cfg.n_frames_general)

    for i, lat in enumerate(lats):
        print(f"[lat] Frame {i}, lat={lat:.2f}")

        fig, df_keep, masks, stats = run_constellation_pipeline(
            df,
            top_n=cfg.top_n,
            apply_visibility=True,
            lat=float(lat),
            lon=cfg.base_lon,
            elev_m=cfg.base_elev,
            utc_time="2025-11-06T02:00:00",
            mask_pad_deg=cfg.pad,
            draw_masks=False,
        )
        fig.savefig(frame_name(outdir, i), dpi=200, bbox_inches="tight")
        fig.clf()

    build_gif_mp4(outdir, "lat")


# ============================================================
#  LONGITUDE SWEEP
# ============================================================
def animate_lon(df, cfg: AnimationConfig):
    outdir = Path("frames_lon")
    outdir.mkdir(exist_ok=True)

    lons = np.linspace(0, 360, cfg.n_frames_general)

    for i, lon in enumerate(lons):
        print(f"[lon] Frame {i}, lon={lon:.2f}")

        fig, df_keep, masks, stats = run_constellation_pipeline(
            df,
            top_n=cfg.top_n,
            apply_visibility=True,
            lat=cfg.base_lat,
            lon=float(lon),
            elev_m=cfg.base_elev,
            utc_time="2025-11-06T02:00:00",
            mask_pad_deg=cfg.pad,
            draw_masks=False,
        )
        fig.savefig(frame_name(outdir, i), dpi=200, bbox_inches="tight")
        fig.clf()

    build_gif_mp4(outdir, "lon")


# ============================================================
#  ELEVATION SWEEP
# ============================================================
def animate_elev(df, cfg: AnimationConfig):
    outdir = Path("frames_elev")
    outdir.mkdir(exist_ok=True)

    elevs = np.linspace(0, 4000, cfg.n_frames_elev)

    for i, el in enumerate(elevs):
        print(f"[elev] Frame {i}, elev={el:.1f}")

        fig, df_keep, masks, stats = run_constellation_pipeline(
            df,
            top_n=cfg.top_n,
            apply_visibility=True,
            lat=cfg.base_lat,
            lon=cfg.base_lon,
            elev_m=float(el),
            utc_time="2025-11-06T02:00:00",
            mask_pad_deg=cfg.pad,
            draw_masks=False,
        )
        fig.savefig(frame_name(outdir, i), dpi=200, bbox_inches="tight")
        fig.clf()

    build_gif_mp4(outdir, "elev")

# ============================================================
#  VISIBLE TOP_N STARS SWEEP
# ============================================================
def animate_topN(df, cfg: AnimationConfig):
    outdir = Path("frames_topN")
    outdir.mkdir(exist_ok=True)

    topNs = np.linspace(100, 2000, cfg.n_frames_general).astype(int)

    for i, nstars in enumerate(topNs):
        print(f"[topN] Frame {i}, top_n={nstars}")

        fig, df_keep, masks, stats = run_constellation_pipeline(
            df,
            top_n=int(nstars),
            apply_visibility=True,
            lat=cfg.base_lat,
            lon=cfg.base_lon,
            elev_m=cfg.base_elev,
            utc_time="2025-11-06T02:00:00",
            mask_pad_deg=cfg.pad,
            draw_masks=False,
        )

        fig.savefig(frame_name(outdir, i), dpi=200, bbox_inches="tight")
        fig.clf()

    build_gif_mp4(outdir, "topN")

# ============================================================
#  MASKPAD SIZE SWEEP
# ============================================================
def animate_maskpad(df, cfg: AnimationConfig):
    outdir = Path("frames_maskpad")
    outdir.mkdir(exist_ok=True)

    pads = np.linspace(1, 15, cfg.n_frames_general)   # degrees of padding

    for i, pad in enumerate(pads):
        print(f"[maskpad] Frame {i}, pad={pad:.2f}°")

        fig, df_keep, masks, stats = run_constellation_pipeline(
            df,
            top_n=cfg.top_n,
            apply_visibility=True,
            lat=cfg.base_lat,
            lon=cfg.base_lon,
            elev_m=cfg.base_elev,
            utc_time="2025-11-06T02:00:00",
            mask_pad_deg=float(pad),
            draw_masks=False,
        )

        fig.savefig(frame_name(outdir, i), dpi=200, bbox_inches="tight")
        fig.clf()

    build_gif_mp4(outdir, "maskpad")


# ============================================================
#  CREATE GIF + MP4
# ============================================================
def build_gif_mp4(outdir, mode):
    print(f"Building GIF/MP4 for mode={mode} ...")

    frame_files = sorted([f for f in outdir.iterdir() if f.suffix == ".png"])

    gif_path = outdir / f"{mode}_animation.gif"
    with imageio.get_writer(gif_path, mode="I", duration=0.08) as writer:
        for f in frame_files:
            writer.append_data(iio.imread(f))

    mp4_path = outdir / f"{mode}_animation.mp4"
    with imageio.get_writer(mp4_path, fps=12, codec="libx264") as writer:
        for f in frame_files:
            writer.append_data(iio.imread(f))

    print("GIF saved:", gif_path)
    print("MP4 saved:", mp4_path)


# ============================================================
#  MAIN
# ============================================================
if __name__ == "__main__":

    cfg = AnimationConfig(
        csv_path="stars_plot_ready.csv",
        top_n=1200,
        pad=3.0,
        base_lat=39.7,
        base_lon=30,
        base_elev=220,
        n_frames_general=60,
        n_frames_elev=20       # To make it easy to see
    )

    print("Using Config:", cfg)

    df = pd.read_csv(cfg.csv_path)
    print(f"Loaded {len(df)} stars")

    animate_time(df, cfg)
    animate_lat(df, cfg)
    animate_lon(df, cfg)
    animate_elev(df, cfg)
    animate_topN(df, cfg)
    animate_maskpad(df, cfg)

    print("\n==== All animations finished ====")
