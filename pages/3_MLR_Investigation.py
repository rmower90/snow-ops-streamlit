"""
MLR Investigation Page

Purpose:
- Recreate MLR preprocessing and cross-validation for a selected
  basin / WY / seasonal / elevation band
- Inspect intermediate artifacts:
    - df_sum_total
    - drop-NaNs table
    - imputed observations
    - cross-validation selections (later)

NOTE:
This page is intentionally lightweight. All heavy lifting should live
in src/snow_ops/mlr/.
"""

from __future__ import annotations

from pathlib import Path
import re
from typing import List 
import glob

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import xarray as xr

from snow_ops.mlr.aso_io import load_aso_metadata
from snow_ops.mlr.preprocess import run_preprocessing_only


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
from typing import List
import glob

DATA_ROOT = Path("data/basins")


def list_basins(root: Path) -> List[str]:
    if not root.exists():
        return []
    return sorted([p.name for p in root.iterdir() if p.is_dir() and not p.name.startswith(".")])


def _extract_wy_from_name(name: str) -> List[int]:
    # matches wy2026, WY2026, wy_2026, etc.
    m = re.findall(r"wy[_-]?(\d{4})", name, flags=re.IGNORECASE)
    return [int(x) for x in m]


def available_test_water_years_for_basin(basin_root: Path, basin: str) -> list[int]:
    patterns = [
        basin_root / "pillows" / "raw" / f"{basin}_insitu_obs_daily_wy_*.nc",
        basin_root / "pillows" / "processed" / f"{basin}_insitu_obs_daily_wy_*.nc",
    ]
    wys = set()
    for pat in patterns:
        for p in pat.parent.glob(pat.name):
            m = re.search(r"wy[_-]?(\d{4})", p.name, flags=re.IGNORECASE)
            if m:
                wys.add(int(m.group(1)))
    return sorted(wys)

def available_wys_from_mlr_outputs(basin_root: Path) -> list[int]:
    """
    Discover available water years based on MLR output files:
      data/basins/{basin}/mlr_prediction/*/prediction_mm_wy{YEAR}.parquet
    Returns sorted unique WYs.
    """
    mlr_root = basin_root / "mlr_prediction"
    if not mlr_root.exists():
        return []

    wys: set[int] = set()
    for p in mlr_root.glob("*/prediction_mm_wy*.parquet"):
        m = re.search(r"prediction_mm_wy(\d{4})\.parquet$", p.name, flags=re.IGNORECASE)
        if m:
            wys.add(int(m.group(1)))

    return sorted(wys)



def prettify_elev_label(label: str) -> str:
    s = str(label).strip()

    # Normalize dash types
    s = s.replace("–", "-").replace("—", "-")

    if s.lower() == "total":
        return "total"

    def to_k(m):
        n = int(m.group(0))
        return f"{n//1000}k"

    s = re.sub(r"\d{4,5}", to_k, s)
    return s


@st.cache_data(show_spinner=False)
def cached_aso_elev_labels(basin: str, config_dir: str) -> list[str]:
    """Fast: read YAML -> open ASO TSERIES only -> return elev labels (truth)."""
    _, paths = load_aso_metadata(basin=basin, config_dir=config_dir)
    tseries_path = paths["aso_tseries_fpath"]

    with xr.open_dataset(tseries_path) as ds:
        if "elev" not in ds.coords:
            raise ValueError(f"ASO tseries is missing 'elev' coord: {tseries_path}")
        return [str(v) for v in ds["elev"].values]


@st.cache_data(show_spinner=False)
def cached_preprocessing(
    basin: str,
    water_year: int,
    seasonal: str,
    band_truth: str,
    config_dir: str,
    pillow_source: str,
    pillow_training_filename: str = "pillow_wy_1980_2025_qa1.nc",
):
    return run_preprocessing_only(
        basin=basin,
        water_year=water_year,
        seasonal=seasonal,
        band=band_truth,
        config_dir=config_dir,
        pillow_source=pillow_source,
        pillow_training_filename=pillow_training_filename,
    )



def plot_aso_vs_insitu_summary(df_sum_total: pd.DataFrame, baseline_cols: list[str] | None = None):
    """Small diagnostic plot: ASO target vs insitu median/mean."""
    df = df_sum_total.copy()
    target_col = "aso_mean_bins_mm"

    if target_col not in df.columns:
        st.warning(f"Expected column '{target_col}' not found in df_sum_total.")
        return

    pillow_cols = [c for c in df.columns if c not in ["time", target_col]]
    use_cols = [c for c in (baseline_cols or []) if c in pillow_cols]
    if len(use_cols) < 3:
        use_cols = pillow_cols

    df["time"] = pd.to_datetime(df["time"])
    df = df.sort_values("time")

    insitu_median = df[use_cols].median(axis=1, skipna=True)
    insitu_mean = df[use_cols].mean(axis=1, skipna=True)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["time"], y=df[target_col], mode="lines", name="ASO (target)", line=dict(dash="solid")))
    fig.add_trace(go.Scatter(x=df["time"], y=insitu_median, mode="lines", name="InSitu median", line=dict(dash="dot")))
    fig.add_trace(go.Scatter(x=df["time"], y=insitu_mean, mode="lines", name="InSitu mean", line=dict(dash="dash")))

    fig.update_layout(
        title=dict(text="ASO target vs InSitu summary", x=0.5, xanchor="center"),
        xaxis_title="Date",
        yaxis_title="SWE [mm]",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="top", y=1.02, xanchor="center", x=0.5),
        margin=dict(l=20, r=20, t=80, b=20),
    )
    st.plotly_chart(fig, use_container_width=True)


# -----------------------------------------------------------------------------
# App configuration
# -----------------------------------------------------------------------------
st.set_page_config(page_title="MLR Investigation", layout="wide")
st.title("MLR Investigation")

CONFIG_DIR = "config/regions_local"  # local dev


# -----------------------------------------------------------------------------
# Sidebar controls (define these first so top controls can depend on basin)
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# Sidebar controls (define these first so top controls can depend on basin)
# -----------------------------------------------------------------------------
with st.sidebar:
    st.header("Basin")
    basins = list_basins(DATA_ROOT)
    if not basins:
        st.error(f"No basins found under {DATA_ROOT}.")
        st.stop()

    default_basin = "USCASJ" if "USCASJ" in basins else basins[0]
    basin = st.selectbox("Basin", basins, index=basins.index(default_basin))

    st.divider()

    st.header("Water Year")
    basin_root = DATA_ROOT / basin

    wys = available_wys_from_mlr_outputs(basin_root)
    if not wys:
        st.error(
            f"No MLR prediction outputs found under: {basin_root / 'mlr_prediction'}\n\n"
            "Expected files like: mlr_prediction/season/prediction_mm_wy2026.parquet"
        )
        st.stop()

    wy = st.selectbox("Water Year", wys, index=len(wys) - 1)

    st.divider()

    st.header("Pillows dataset")
    pillow_source = st.radio(
        "Pillow dataset",
        [
            "Test (raw download)",
            "Test (processed QA/QC)",
            "Training (manual QA/QC multi-year)",
        ],
        index=1,
        key="pillow_source",
    )

    if pillow_source == "Test (raw download)":
        pillows_path = basin_root / "pillows" / "raw" / f"{basin}_insitu_obs_daily_wy_{wy}.nc"
    elif pillow_source == "Test (processed QA/QC)":
        pillows_path = basin_root / "pillows" / "processed" / f"{basin}_insitu_obs_daily_wy_{wy}.nc"
    else:
        pillows_path = basin_root / "pillows" / "processed" / "pillow_wy_1980_2025_qa1.nc"

    with st.expander("Resolved files", expanded=False):
        st.write("Pillows:", pillows_path.as_posix())
        st.write("Exists:", "✅" if pillows_path.exists() else "❌")

    st.divider()

    run_button = st.button("Run MLR Investigation", key="mlr_run_button")



# -----------------------------------------------------------------------------
# Top controls (Elevation from ASO truth -> prettified UI)
# -----------------------------------------------------------------------------
top_left, top_mid, top_right = st.columns([1.2, 1.2, 1.4])

try:
    aso_truth_labels = cached_aso_elev_labels(basin, CONFIG_DIR)
except Exception as e:
    st.error("Could not load ASO elevation labels for UI.")
    st.exception(e)
    st.stop()

ui_labels = [prettify_elev_label(x) for x in aso_truth_labels]

# Make UI labels unique if collisions occur
seen = {}
ui_unique = []
ui_to_truth = {}
for raw, ui in zip(aso_truth_labels, ui_labels):
    if ui in seen:
        seen[ui] += 1
        ui_u = f"{ui} ({seen[ui]})"
    else:
        seen[ui] = 0
        ui_u = ui
    ui_unique.append(ui_u)
    ui_to_truth[ui_u] = raw

default_ui = "total" if "total" in ui_unique else ui_unique[-1]

with top_left:
    band_ui = st.selectbox("Elevation Band", ui_unique, index=ui_unique.index(default_ui))

with top_mid:
    seasonal_choice = st.selectbox("Seasonal", ["season", "accumulation", "melt"], index=0)

with top_right:
    st.selectbox("Units", ["milimeters"], index=0, disabled=True)

band_truth = ui_to_truth[band_ui]


# -----------------------------------------------------------------------------
# Execute
# -----------------------------------------------------------------------------
if run_button:
    try:
        with st.spinner("Running MLR preprocessing..."):
            out = cached_preprocessing(
                basin=basin,
                water_year=wy,
                seasonal=seasonal_choice,
                band_truth=band_truth,
                config_dir=CONFIG_DIR,
                pillow_source=pillow_source, 
            )

        st.subheader("Stage")
        st.write(out.get("_stage"))

        st.subheader("ASO SWE variable")
        st.write(out.get("aso_var_used"))

        st.subheader("Elevation idx used")
        st.write(out.get("elev_idx"))

        if "artifacts" in out:
            st.subheader("Artifacts summary")
            st.json(out["artifacts"])

        if "df_sum_total" in out:
            df = out["df_sum_total"]

            st.subheader("Sanity checks")
            st.write("df_sum_total date range:")
            st.write(df["time"].min(), "→", df["time"].max())

            miss = (df.isna().mean().sort_values(ascending=False) * 100).head(15)
            st.write("Missingness (% NaN) by column (top 15):")
            st.dataframe(miss.to_frame("percent_nan"), use_container_width=True)

            baseline_preview = out.get("artifacts", {}).get("baseline_pils_preview", [])
            plot_aso_vs_insitu_summary(df, baseline_cols=baseline_preview)

        if "drop_na_df" in out:
            st.subheader("drop_na_df (head)")
            st.dataframe(out["drop_na_df"].head(), use_container_width=True)

        if "artifacts" in out:
            st.subheader("Baseline pillows summary")
            st.write(
                {
                    "n_all_pillows": out["artifacts"].get("n_all_pillows"),
                    "n_baseline_pillows": out["artifacts"].get("n_baseline_pillows"),
                    "df_sum_total_shape": out["artifacts"].get("df_sum_total_shape"),
                    "drop_na_df_shape": out["artifacts"].get("drop_na_df_shape"),
                }
            )
            st.write("Baseline pillows (preview):")
            st.write(out["artifacts"].get("baseline_pils_preview", []))

    except Exception as e:
        st.exception(e)


# -----------------------------------------------------------------------------
# Page notes
# -----------------------------------------------------------------------------
with st.expander("Notes on planned functionality", expanded=False):
    st.markdown(
        """
        **This page will eventually allow you to:**
        - Re-run MLR preprocessing for a specific basin / WY / band
        - Inspect `df_sum_total` and NaN handling
        - Reproduce cross-validation station selection
        - Compare predicted SWE vs ASO SWE
        - Visualize residuals and coefficients

        **Design principle:**
        - Streamlit page = UI + orchestration only
        - All heavy computation lives in `src/snow_ops/mlr/`
        """
    )
