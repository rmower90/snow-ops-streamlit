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

import datetime
from pathlib import Path
import re
from typing import List 
import glob

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt

from snow_ops.mlr.aso_io import load_aso_metadata
from snow_ops.mlr.preprocess import (
    run_preprocessing_only,
)
from snow_ops.mlr.steps import (
    dataset_to_list,
    process_daily_qa
)
from snow_ops.mlr.crossval import (
    run_cross_val_selection,
    run_mlr_train_predict,
    run_mlr_inference
)


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

@st.cache_data(show_spinner=False)
def load_mlr_pred_dates(basin_root: Path, seasonal: str, wy: int) -> list[pd.Timestamp]:
    seasonal_dir = {"season": "season", "accumulation": "accum", "melt": "melt"}.get(seasonal, seasonal)
    p = basin_root / f"mlr_prediction/{seasonal_dir}/prediction_mm_wy{wy}.parquet"
    df = pd.read_parquet(p, columns=["Date"])
    df["Date"] = pd.to_datetime(df["Date"])
    dates = sorted(df["Date"].dropna().unique(), reverse=True)
    return dates

# @st.cache_data(show_spinner=False)
# def map_date_to_tidx(time_values, date_selected, mode: str = "nearest"):
#     """
#     Map a selected date (from prediction output) to an index in obs time coord.

#     Parameters
#     ----------
#     time_values : array-like
#         obs_ds["time"].values (or a list/array of datetimes)
#     date_selected : any
#         Selected date from UI (Timestamp/date/datetime/str)
#     mode : {"exact","nearest","ffill","bfill"}
#         - exact: require exact match after normalization
#         - nearest: choose closest obs time
#         - ffill: choose latest obs time <= selected date
#         - bfill: choose earliest obs time >= selected date

#     Returns
#     -------
#     t_idx : int
#     obs_time_used : pd.Timestamp
#     """
#     # Normalize obs time axis -> pandas DatetimeIndex (date-level)
#     obs_times = pd.to_datetime(np.asarray(time_values)).normalize()
#     if len(obs_times) == 0:
#         raise ValueError("obs time coord is empty.")

#     # Normalize selected date -> Timestamp (date-level)
#     sel = pd.to_datetime(date_selected).normalize()

#     # Fast path: exact match after normalization
#     if mode == "exact":
#         hits = np.where(obs_times == sel)[0]
#         if hits.size == 0:
#             raise ValueError(f"Selected date {sel} not found in obs time coord.")
#         idx = int(hits[0])
#         return idx, obs_times[idx]

#     # nearest/ffill/bfill via searchsorted
#     obs_sorted = obs_times  # should already be sorted, but assume it is
#     pos = int(obs_sorted.searchsorted(sel))

#     if mode == "bfill":
#         if pos >= len(obs_sorted):
#             pos = len(obs_sorted) - 1
#         return pos, obs_sorted[pos]

#     if mode == "ffill":
#         if pos == 0:
#             pos = 0
#         else:
#             # if sel is not exactly in obs_sorted, step back
#             if pos < len(obs_sorted) and obs_sorted[pos] != sel:
#                 pos = pos - 1
#             else:
#                 pos = pos
#         return pos, obs_sorted[pos]

#     # mode == "nearest" (default)
#     if pos == 0:
#         return 0, obs_sorted[0]
#     if pos >= len(obs_sorted):
#         return len(obs_sorted) - 1, obs_sorted[-1]

#     before = obs_sorted[pos - 1]
#     after = obs_sorted[pos]
#     if abs((sel - before).days) <= abs((after - sel).days):
#         return pos - 1, before
#     return pos, after

def map_date_to_tidx(time_values, date_selected, mode="nearest"):
    # time_values: numpy datetime64 array
    target = np.datetime64(pd.to_datetime(date_selected).date())
    idx = np.searchsorted(time_values, target)

    if mode == "exact":
        matches = np.where(time_values == target)[0]
        if len(matches) == 0:
            raise ValueError(f"Selected date {target} not found in obs time coord.")
        return int(matches[0]), target

    # nearest
    if idx == 0:
        return 0, time_values[0]
    if idx >= len(time_values):
        return len(time_values) - 1, time_values[-1]

    before = time_values[idx - 1]
    after = time_values[idx]
    chosen = before if (target - before) <= (after - target) else after
    chosen_idx = idx - 1 if chosen == before else idx
    return int(chosen_idx), chosen


@st.cache_data(show_spinner=False)
def mlr_prediction_dates(basin_root: Path, seasonal: str, wy: int) -> list[pd.Timestamp]:
    seasonal_dir = {"season": "season", "accumulation": "accum", "melt": "melt"}.get(seasonal, seasonal)
    p = basin_root / f"mlr_prediction/{seasonal_dir}/prediction_mm_wy{wy}.parquet"
    df = pd.read_parquet(p, columns=["Date"])
    dates = pd.to_datetime(df["Date"]).dropna().unique()
    return sorted(dates, reverse=True)

def date_to_tidx_from_list(obs_list: list[xr.DataArray], date_selected: pd.Timestamp) -> tuple[int, pd.Timestamp, bool]:
    # assumes all pillows share same time coord; use first
    times = pd.to_datetime(obs_list[0]["time"].values)
    date_selected = pd.to_datetime(date_selected)

    # exact match
    m = (times == date_selected)
    if m.any():
        idx = int(np.where(m)[0][0])
        return idx, pd.Timestamp(times[idx]), False

    # nearest fallback
    idx = int(np.argmin(np.abs(times - date_selected)))
    return idx, pd.Timestamp(times[idx]), True


SEASONAL_DIR_MAP = {"season": "season", "accumulation": "accum", "melt": "melt"}

@st.cache_data(show_spinner=False)
def cached_mlr_dates(basin: str, wy: int, seasonal: str) -> pd.DatetimeIndex:
    """
    Read MLR predictions parquet just for Date column and return unique dates (desc).
    """
    seasonal_dir = SEASONAL_DIR_MAP.get(seasonal, seasonal)
    p = Path("data/basins") / basin / "mlr_prediction" / seasonal_dir / f"prediction_mm_wy{wy}.parquet"
    if not p.exists():
        raise FileNotFoundError(f"Missing MLR predictions file: {p}")

    df = pd.read_parquet(p, columns=["Date"])
    d = pd.to_datetime(df["Date"]).dropna().dt.normalize().unique()
    # newest first
    return pd.DatetimeIndex(sorted(d, reverse=True))

@st.cache_data(show_spinner=False)
def cached_pillow_times(pillows_path: str) -> pd.DatetimeIndex:
    """
    Load only the time coordinate from a pillow NetCDF.
    """
    ds = xr.open_dataset(pillows_path)  # lazy-ish
    if "time" not in ds.coords:
        raise ValueError(f"'time' coord not found in pillows dataset: {pillows_path}")
    t = pd.to_datetime(ds["time"].values).normalize()
    return pd.DatetimeIndex(t)

@st.cache_data(show_spinner=False)
def load_aso_tseries(basin: str, config_dir: str) -> xr.Dataset:
    """
    Load ASO 50m SWE time series via basin YAML.
    Small dataset → safe to keep in memory.
    """
    cfg, paths = load_aso_metadata(basin=basin, config_dir=config_dir)
    ds = xr.load_dataset(paths["aso_tseries_fpath"])
    return ds 

@st.cache_data(show_spinner=False)
def load_aso_dem(basin: str, config_dir: str) -> xr.Dataset:
    """
    Load ASO 50m DEM via basin YAML.
    Small dataset → safe to keep in memory.
    """
    cfg, paths = load_aso_metadata(basin=basin, config_dir=config_dir)
    ds = xr.open_dataset(paths["demBin_fpath"], engine = "netcdf4")
    return ds

def date_to_t_idx(pillow_times: pd.DatetimeIndex, selected_date: pd.Timestamp) -> tuple[int, bool]:
    """
    Convert a selected_date -> t_idx on pillow_times.
    Returns (t_idx, exact_match).
    """
    d = pd.to_datetime(selected_date).normalize()

    # exact match preferred
    hits = np.where(pillow_times.values == d.to_datetime64())[0]
    if hits.size > 0:
        return int(hits[0]), True

    # fallback: nearest (should be rare)
    # (pillow_times is sorted; if not, sort it once)
    pillow_times_sorted = pillow_times.sort_values()
    pos = int(np.searchsorted(pillow_times_sorted.values, d.to_datetime64()))
    pos = max(0, min(pos, len(pillow_times_sorted) - 1))
    return int(pos), False


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


def predictions_v_observations(aso_mean_swe,
                               bestFit,
                               validation,
                               max_swe = 1500,
                               features = None,
                               showPlot = True):
    """
        Produces one to one plot of basin mean swe vs. predicted swe 
        Input:
            aso_mean_swe - xarray dataarray of basin mean swe for aso flight dates.
            bestFit - np array of best fit predictions. 
            validation - np array of cross-validation approach.
            max_swe - max swe used for limits on figure.
            features - list of features.
        Output:
            plot of best fit and cross validation.
    """
    ## calculate statistics
    rms_validation = np.sqrt(((aso_mean_swe - validation)**2).mean().values)
    r2_validation = np.corrcoef(aso_mean_swe.values,validation)[0,1]**2
    rms_best = np.sqrt(((aso_mean_swe - bestFit)**2).mean().values)
    r2_best = np.corrcoef(aso_mean_swe.values,bestFit)[0,1]**2

    if not showPlot:
        return r2_validation,rms_validation,r2_best,rms_best, None
    else:
    ## plotting
        fig,ax = plt.subplots(dpi = 150)
        plt.plot(validation, aso_mean_swe,'o', color = 'C1', label="Validation: r$^2$" + f'= {r2_validation:.3f}\nRMS error     = {rms_validation:.2f}')
        plt.plot(bestFit, aso_mean_swe,'x', color = 'C0', label = 'Best Fit: r$^2$' + f'    = {r2_best:.3f}\nRMS error     = {rms_best:.2f}')
        plt.ylabel("Basin Mean SWE [mm]")
        plt.xlabel("Predicted Basin Mean SWE [mm]")
        if features is not None:
            plt.title(f'Predictions \nn = {len(validation)}; k = {len(features)}',fontweight = 'bold')
        plt.legend()
        plt.ylim([0,max_swe])
        plt.xlim([0,max_swe])
        plt.show()
        return r2_validation,rms_validation,r2_best,rms_best, fig
    

def predictions_v_observations_tseries(aso_mean_swe,
                                       validation,
                                       pillow_data,
                                       stations,
                                       start_wy,
                                       end_wy,
                                       aso_site_name,
                                       saveFig = False,
                                       max_swe = 1500):
                              
    """
        Plot time series of pillows that are best predictors for ASO.
        Input:
            aso_mean_swe - xarray dataarray of basin mean swe for aso flight dates.
            validation - np array of cross-validation approach.
            pillow_data - xarray dataset with pillow data.
            stations - list of station integers identifying best predictors.
            start_wy - integer for starting water year.
            end_wy - integer for ending water year.
            max_swe - max swe used for limits on figure.
        Output:
            time series plot.
    """
    
    fig,ax = plt.subplots(dpi=200, figsize=(8,3))

    ax2 = plt.gca()
    ax1 = plt.twinx(ax2)

    ax2.plot(aso_mean_swe.date, aso_mean_swe.data, marker="o", linestyle="", color="C0", label="ASO", zorder=3)
    ax2.plot(aso_mean_swe.date, validation, 'x', color="C1", label="Predicted", zorder=3)
    plt.ylabel("Basin Mean SWE [mm]")

    colors = [f'C{str(i+2)}' for i in range(0,len(stations))]

    for i,c in zip(stations, colors):
        plot_data = np.array(pillow_data[i].data, dtype="f")
        
        ax1.plot(pillow_data[i].time, plot_data, label=pillow_data[i].name, color=c)


    ax1.set_ylabel("Pillow Observed SWE [mm]")
    ax2.set_ylabel("Basin Mean SWE [mm]")
    # ax2.set_ylim(0,max_swe)
    ax2.set_xlim(np.datetime64(f'{start_wy-1}-10-01'),np.datetime64(f'{end_wy}-08-01'))
    ax1.set_xlim(np.datetime64(f'{start_wy-1}-10-01'),np.datetime64(f'{end_wy}-08-01'))
    ax2.set_ylim(0,max_swe)
    ax1.set_ylim(0,3600)
    # plt.xlim(np.datetime64("2012-10-01"),np.datetime64("2019-09-01"))
    ax1.legend(loc="upper left")
    ax2.legend(loc="upper right")

    return fig


def inference_pillow_tseries(pillow_obs_test_list: List[xr.DataArray],
                             mlr_preds_df: pd.DataFrame,
                             selected_station_ids: List[str],
                             t_idx: int) -> pd.DataFrame:
    """
    Extract pillow observations and MLR predictions for selected stations at t_idx.
    Returns a DataFrame with columns: ["station_id", "obs_swe_mm", "pred_swe_mm"].
    """
    records = []
    for da in pillow_obs_test_list:
        station_id = str(da.name)
        if station_id not in selected_station_ids:
            continue
        obs_swe = float(da.isel(time=t_idx).values)
        pred_row = mlr_preds_df[mlr_preds_df["station_id"] == station_id]
        if pred_row.empty:
            pred_swe = np.nan
        else:
            pred_swe = float(pred_row.iloc[0]["predicted_swe_mm"])
        records.append({
            "station_id": station_id,
            "obs_swe_mm": obs_swe,
            "pred_swe_mm": pred_swe,
        })
    return pd.DataFrame.from_records(records)






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

    pillow_source = st.radio(
        "Pillow dataset",
        [
            "Test (raw download)",
            "Test (processed QA/QC)",
            "Training (manual QA/QC multi-year)",
        ],
        index=1,
    )

    if pillow_source == "Test (raw download)":
        pillows_path = (Path("data/basins") / basin / "pillows/raw" /
                        f"{basin}_insitu_obs_daily_wy_{wy}.nc")
    elif pillow_source == "Test (processed QA/QC)":
        pillows_path = (Path("data/basins") / basin / "pillows/processed" /
                        f"{basin}_insitu_obs_daily_wy_{wy}.nc")
    else:
        pillows_path = (Path("data/basins") / basin / "pillows/processed" /
                        "pillow_wy_1980_2025_qa1.nc")


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
    imputation_choice = st.selectbox("Training Infer NaNs", ["drop NaNs", "predict NaNs"], index=0, disabled=False)

band_truth = ui_to_truth[band_ui]


# -----------------------------------------------------------------------------
# Step 3A: Date -> t_idx
# -----------------------------------------------------------------------------
st.subheader("Step 3A: Select date → compute t_idx")

try:
    mlr_dates = cached_mlr_dates(basin=basin, wy=wy, seasonal=seasonal_choice)
except Exception as e:
    st.error("Could not read MLR prediction dates.")
    st.exception(e)
    st.stop()

selected_date = st.selectbox(
    "Prediction date (from MLR output)",
    options=list(mlr_dates),
    index=0,  # newest first
    format_func=lambda d: pd.to_datetime(d).strftime("%Y-%m-%d"),
    key="mlr_date_select",
)

# Load pillow time coord and compute t_idx
try:
    pillow_times = cached_pillow_times(str(pillows_path))
    t_idx, exact = date_to_t_idx(pillow_times, selected_date)
except Exception as e:
    st.error("Could not compute t_idx from pillows time coordinate.")
    st.exception(e)
    st.stop()

# Tiny debug table
debug = pd.DataFrame(
    [{
        "basin": basin,
        "water_year": wy,
        "seasonal": seasonal_choice,
        "pillow_source": pillow_source,
        "pillows_path": str(pillows_path),
        "selected_date": pd.to_datetime(selected_date).strftime("%Y-%m-%d"),
        "t_idx": t_idx,
        "exact_match": bool(exact),
        "pillow_time_at_t_idx": pd.to_datetime(pillow_times[t_idx]).strftime("%Y-%m-%d"),
        "pillow_time_min": pd.to_datetime(pillow_times.min()).strftime("%Y-%m-%d"),
        "pillow_time_max": pd.to_datetime(pillow_times.max()).strftime("%Y-%m-%d"),
        "n_pillow_days": len(pillow_times),
        "n_mlr_dates": len(mlr_dates),
    }]
)
st.dataframe(debug, use_container_width=True, hide_index=True)



# -----------------------------------------------------------------------------
# Execute
# -----------------------------------------------------------------------------
basin_root = Path("data/basins") / basin

# Date selector (most recent first)
dates = load_mlr_pred_dates(basin_root, seasonal_choice, wy)
if not dates:
    st.error("No dates found in MLR predictions for this selection.")
    st.stop()

dates = sorted(dates, reverse=True)
date_selected = st.selectbox(
    "Date",
    dates,
    index=0,
    format_func=lambda d: pd.to_datetime(d).strftime("%Y-%m-%d"),
    key="mlr_inv_date_select",
)

# Run button (give it a key if you ever duplicate buttons)
# run_button already defined in sidebar; just use it here.

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

        # Persist results so changing date doesn't force full preprocessing
        st.session_state["mlr_inv_out"] = out

    except Exception as e:
        st.exception(e)
        st.stop()

# If we have cached output, use it (lets you change dates w/o rerun)
out = st.session_state.get("mlr_inv_out")
if out is None:
    st.info("Click **Run MLR Investigation** to load preprocessing artifacts.")
    st.stop()

# --- Use objects already prepared in preprocess.py ---
# These should exist if you kept your additions:
# out["obs_test_raw_lst"], out["obs_test_proc_lst"], out["baseline_pils"]
obs_data_test_lst_raw = out.get("obs_test_raw_lst")
obs_data_test_lst = out.get("obs_test_proc_lst")

if obs_data_test_lst is None or obs_data_test_lst_raw is None:
    st.error(
        "Missing obs lists in `out`. Make sure preprocess.py sets:\n"
        "- out['obs_test_raw_lst']\n"
        "- out['obs_test_proc_lst']\n"
    )
    st.stop()

if "baseline_pils" not in out:
    st.error("Missing out['baseline_pils']. Make sure preprocess.py stores the full list.")
    st.stop()

# --- Step 3A: date -> t_idx (use obs time as truth; nearest match) ---
# Take time axis from the first pillow DA (all should share same time index)
time_values = obs_data_test_lst[0]["time"].values  # numpy datetime64 array

t_idx, obs_time_used = map_date_to_tidx(time_values, date_selected, mode="nearest")
st.caption(
    f"Selected prediction date: {pd.to_datetime(date_selected).date()} | "
    f"Using obs date: {pd.to_datetime(obs_time_used).date()} (t_idx={t_idx})"
)

# --- Step 3B: daily QA ---
current_vals_df, all_pils_QA, baseline_pils_, df_qa_table = process_daily_qa(
    t_idx=t_idx,
    obs_data_raw=obs_data_test_lst_raw,
    obs_data_qa=obs_data_test_lst,
    baseline_pils=out["baseline_pils"],
    printOutput=False,
)

st.subheader("Daily QA summary")
st.write({
    "n_all_pils": len([i.name for i in obs_data_test_lst_raw]),
    "n_all_pils_QA": len(all_pils_QA),
    "n_baseline_before": len(out["baseline_pils"]),
    "n_baseline_after_daily_QA": len(baseline_pils_),
})

st.subheader("Current day values (raw)")
st.dataframe(current_vals_df, use_container_width=True)

st.subheader("Current day QA flags/table")
st.dataframe(df_qa_table, use_container_width=True)

# --- Existing diagnostics you already had ---
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

    # IMPORTANT: baseline_pils_preview is truncated; prefer full list for summaries
    # For plotting, you can still downselect to a manageable subset if needed.
    baseline_for_plot = out.get("baseline_pils", [])[:25]
    plot_aso_vs_insitu_summary(df, baseline_cols=baseline_for_plot)

if "drop_na_df" in out:
    st.subheader("drop_na_df (head)")
    st.dataframe(out["drop_na_df"].head(), use_container_width=True)

st.subheader("Cross-val inputs (preview)")
st.write({
    "df_sum_total_shape": tuple(out["df_sum_total"].shape),
    "aso_tseries_da_dims": out["aso_tseries_da"].dims,
    "aso_tseries_da_shape": out["aso_tseries_da"].shape,
    "start_wy": out["start_wy"],
    "end_wy": out["end_wy"],
    "n_all_pils_QA_today": len(all_pils_QA),
    "n_baseline_pils_QA_today": len(baseline_pils_),
})

run_cv = st.button("Run cross-validation (debug)")
if run_cv:
    # load aso time series.
    aso_tseries_ds = load_aso_tseries(basin, CONFIG_DIR)
    # load aso dem bin.
    dem_bin = load_aso_dem(basin, CONFIG_DIR)
    # create prediction date datetime object.
    prediction_date = datetime.datetime(pd.to_datetime(obs_time_used).year,pd.to_datetime(obs_time_used).month,pd.to_datetime(obs_time_used).day)
    # create seasonal booleans.
    if seasonal_choice == "season":
        isSplit, isAccum = False, False
    elif seasonal_choice == "accumulation":
        isSplit, isAccum = True, True
    else: #melt
        isSplit, isAccum = True, False

    if imputation_choice == "drop NaNs":
        isImpute = False
    else:
        isImpute = True
    
    preds_bestfit, preds_val, selected_station_ids, selected_station_names, aso_used, obs_used, summary_dict, df_final_train = run_mlr_train_predict(
        aso_tseries_ds.aso_swe,
        out["obs_train_lst"],
        out["elev_idx"],
        out["all_pils"],
        all_pils_QA,
        out["df_sum_total"],
        baseline_pils_,
        out["start_wy"],
        out["end_wy"],
        basin,
        impute_df = out["pillow_imputation_df"],
        melt_thresh_df = out["melt_threshold_df"],
        isSplit=isSplit,
        isAccum=isAccum,
        isImpute=isImpute,
        isMean=False,
        prediction_date=prediction_date,
        modelID=0,
        QA_flag=1,
        model_type = 'MLR',
        showOutput = False,
        isCombination = False,
        saveValidation = False
        )
    
    
    summary_dict = run_mlr_inference(df_final_train,
                                     current_vals_df,
                                     dem_bin.dem_bin,
                                     out["elev_idx"],
                                     selected_station_names,
                                     summary_dict,
                                     add_zeroASO=True,
                                     ModelID = 0
                                     )
    # preds_bestfit, preds_val, selected_station_ids, selected_station_names, aso_used, obs_used = run_cross_val_selection(
    #     obs_data=out["obs_train_lst"],
    #     df_sum_total=out["df_sum_total"],
    #     aso_tseries=out["aso_tseries_da"],
    #     all_pillows=baseline_pils_,
    #     start_wy=out["start_wy"],
    #     end_wy=out["end_wy"],
    #     elev_band=out["elev_idx"],
    #     isCombination=True,
    #     isMelt=(seasonal_choice == "melt"),
    #     showOutput=False,
    # )

    st.write(summary_dict)
    
    st.subheader("MLR Training Results")

    r2v, rmsev, r2b, rmseb, fig1 =predictions_v_observations(aso_used[:,out["elev_idx"]],
                               bestFit = preds_bestfit,
                               validation = preds_val,
                               max_swe = 1500)
    
    st.pyplot(fig1, clear_figure=True)

    fig2 = predictions_v_observations_tseries(aso_used[:,out["elev_idx"]],
                                           preds_val,
                                           obs_used,
                                           selected_station_ids,
                                           out["start_wy"],
                                           out["end_wy"],
                                           basin,
                                           max_swe = 1500)
    st.pyplot(fig2, clear_figure=True)

    # inference_pillow_tseries_df = inference_pillow_tseries(
    #     pillow_obs_test_list=obs_data_test_lst,
    #     mlr_preds_df=pd.DataFrame({
    #         "station_id": selected_station_ids,
    #         "predicted_swe_mm": preds_val,
    #     }),
    #     selected_station_ids=selected_station_ids,
    #     t_idx=0
    # )
    # st.subheader("Inference pillow tseries at selected date")
    # st.dataframe(inference_pillow_tseries_df, use_container_width=True)
    st.subheader("Cross-validation summary")
    st.write({
        "n_candidates_today": len(baseline_pils_),
        "n_selected_stations": len(selected_station_ids),
        "selected_stations": selected_station_names,
        "predictions_len": len(preds_bestfit),
    })

    assert len(preds_bestfit) == aso_used.shape[0], "Prediction/ASO mismatch"





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
