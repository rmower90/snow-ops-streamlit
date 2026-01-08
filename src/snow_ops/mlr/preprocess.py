# src/snow_ops/mlr/preprocess.py
from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Any

import xarray as xr
import numpy as np
import pandas as pd

from snow_ops.mlr.aso_io import load_aso_metadata, load_aso_data
from snow_ops.mlr.steps import (
    dataset_to_list,
    train_test_split,
    combine_aso_insitu,
    generate_drop_NaNs_table,
    load_melt_imputation_tables,
    create_qa_tables,  # imported for later use; not strictly required yet

)



def _repo_root() -> Path:
    # preprocess.py -> src/snow_ops/mlr/preprocess.py -> src -> repo root
    return Path(__file__).resolve().parents[3]


def pick_aso_swe_var(ds: xr.Dataset) -> str:
    """
    Prefer your historical variable name first; fall back to common SWE-ish names.
    """
    preferred = ["aso_swe", "aso_mean_bins_mm", "aso_mean_swe", "swe"]
    for v in preferred:
        if v in ds.data_vars:
            return v

    # Very light heuristic fallback (only if you ever change variable names)
    for v in ds.data_vars:
        if "swe" in v.lower():
            return v

    raise ValueError(
        "Could not identify ASO SWE variable in aso_tseries_ds. "
        f"data_vars={list(ds.data_vars)}"
    )

def band_to_elev_idx_from_aso(aso_elev_values: list[str], band_truth: str) -> int:
    """
    Map ASO truth elev label -> index in ASO tseries elev coordinate.

    Parameters
    ----------
    aso_elev_values : list[str]
        List of ASO elevation labels (from aso_tseries_ds["elev"].values)
    band_truth : str
        Exact label selected from ASO elev (e.g. "<7000", "7000–8000", "total")

    Returns
    -------
    int
        Index into ASO elev dimension.
    """
    if band_truth not in aso_elev_values:
        raise ValueError(
            f"Band '{band_truth}' not found in ASO elev labels: {aso_elev_values}"
        )
    return aso_elev_values.index(band_truth)




def band_to_elev_idx(cfg: dict, band: str) -> int:
    """
    Map Streamlit band label -> ASO tseries elev index using YAML labels.

    Assumptions:
      - YAML contains cfg["elevation"]["bins"]["labels"] like ["<7k","7k-8k",...]
      - "total" is represented by using the last elev entry (idx = -1), consistent
        with your historical approach.
    """
    if band.strip().lower() == "total":
        return -1

    labels = cfg["elevation"]["bins"]["labels"]
    if band not in labels:
        raise ValueError(f"Band '{band}' not in YAML labels: {labels}")
    return labels.index(band)



def run_preprocessing_only(
    basin: str,
    water_year: int,
    seasonal: str,
    band: str,
    *,
    config_dir: str | None = None,
    exclude_pillows: list[str] | None = None,
    pillow_source: str = "Test (processed QA/QC)",
    pillow_training_filename: str = "pillow_wy_1980_2025_qa1.nc",
) -> Dict[str, Any]:

    """
    Phase 2B (drop-in):
      - Resolve repo-relative artifacts (MLR/UASWE/SNODAS + pillow NetCDFs)
      - Load pillow NetCDFs
      - Load ASO datasets via YAML (absolute paths)
      - Build df_sum_total + drop_na_df (up to ~line 286-equivalent behavior)
      - SnowModel zarrs remain optional (opened only if present)

    Returns a dict of metadata + artifacts for Streamlit display.
    """
    exclude_pillows = exclude_pillows or []

    repo_root = _repo_root()
    config_dir_path = Path(config_dir) if config_dir else (repo_root / "config")
    basin_root = repo_root / "data" / "basins" / basin

    seasonal_dir_map = {"season": "season", "accumulation": "accum", "melt": "melt"}
    seasonal_dir = seasonal_dir_map.get(seasonal, seasonal)

    # Repo-relative expected paths (your Streamlit data tree)
    paths = {
        # Artifacts already used by other pages
        "mlr_pred_mm": basin_root / f"mlr_prediction/{seasonal_dir}/prediction_mm_wy{water_year}.parquet",
        "impute_dir": basin_root / "mlr_prediction/imputation/",
        "uaswe_m": basin_root / f"uaswe/mean_swe_uaswe_m_wy{water_year}.parquet",
        "snodas_m": basin_root / f"snodas/mean_swe_snodas_m_wy{water_year}.parquet",

        # Pillow NetCDFs (exist in your repo tree)
        "insitu_processed_train": basin_root / "pillows/processed/pillow_wy_1980_2025_qa1.nc",
        "insitu_test_raw": basin_root / f"pillows/raw/{basin}_insitu_obs_daily_wy_{water_year}.nc",
        "insitu_test_processed": basin_root / f"pillows/processed/{basin}_insitu_obs_daily_wy_{water_year}.nc",

        # Optional SnowModel correlation zarrs (may not exist locally)
        "snowmodel_train_zarr": basin_root / "snowmodel/hrrr_correlated_train_2017_2025_dowy.zarr",
        "snowmodel_test_zarr": basin_root / f"snowmodel/hrrr_correlated_test_{water_year}.zarr",

        # imputation_dir 
        "imputation_dir": basin_root / "mlr_prediction/imputation/",
        "melt_threshold_dir": basin_root / "mlr_prediction/melt_threshold/",
    }
        # Selected pillows dataset (driven by UI)
    if pillow_source == "Test (raw download)":
        paths["insitu_selected"] = basin_root / f"pillows/raw/{basin}_insitu_obs_daily_wy_{water_year}.nc"
        selected_kind = "test_raw"
    elif pillow_source == "Test (processed QA/QC)":
        paths["insitu_selected"] = basin_root / f"pillows/processed/{basin}_insitu_obs_daily_wy_{water_year}.nc"
        selected_kind = "test_processed"
    else:
        paths["insitu_selected"] = basin_root / f"pillows/processed/{pillow_training_filename}"
        selected_kind = "train_manual_multiyear"

    exists = {k: p.exists() for k, p in paths.items()}

    out: Dict[str, Any] = {
        "_stage": "phase2b_init",
        "basin": basin,
        "water_year": water_year,
        "seasonal": seasonal,
        "band": band,
        "exclude_pillows_user": exclude_pillows,
        "repo_root": str(repo_root),
        "config_dir": str(config_dir_path),
        "basin_root": str(basin_root),
        "paths": {k: str(v) for k, v in paths.items()},
        "exists": exists,
    }
    out["pillow_source"] = pillow_source
    out["pillow_selected_kind"] = selected_kind


    # --- Load required pillow NetCDF datasets ---
    required_nc = ["insitu_processed_train", "insitu_test_raw", "insitu_test_processed"]
    missing_nc = [k for k in required_nc if not exists[k]]
    if missing_nc:
        out["_stage"] = "error_missing_netcdf"
        out["error"] = f"Missing required NetCDF files: {missing_nc}"
        return out

    obs_train_ds = xr.load_dataset(str(paths["insitu_processed_train"]))
    obs_test_raw_ds = xr.load_dataset(str(paths["insitu_test_raw"]))
    obs_test_proc_ds = xr.load_dataset(str(paths["insitu_test_processed"]))

    # --- Load selected pillows dataset (for visualization / debugging) ---
    if not exists.get("insitu_selected", False):
        out["_stage"] = "error_missing_selected_pillows"
        out["error"] = f"Missing selected pillows dataset: insitu_selected -> {paths['insitu_selected']}"
        return out

    obs_selected_ds = xr.load_dataset(str(paths["insitu_selected"]))

    # If training multi-year, slice to the selected WY for viewing
    if selected_kind == "train_manual_multiyear":
        start = np.datetime64(f"{water_year-1}-10-01")
        end = np.datetime64(f"{water_year}-10-01")
        if "time" in obs_selected_ds.coords:
            obs_selected_ds = obs_selected_ds.sel(time=slice(start, end))


    # --- YAML-driven ASO loading (absolute paths, matches batch workflow) ---
    cfg, meta_paths = load_aso_metadata(basin=basin, config_dir=config_dir_path)

    # Merge excludes: cfg + user list
    cfg_exclude = (cfg.get("pillow_api", {}) or {}).get("exclude_pillows", []) or []
    exclude_all = sorted(set(exclude_pillows + cfg_exclude))
    out["exclude_pillows_cfg_n"] = len(cfg_exclude)
    out["exclude_pillows_total_n"] = len(exclude_all)
    out["cfg_summary"] = {
        "aso_start_wy": int(cfg["aso_years"]["start"]) if "aso_years" in cfg else None,
        "aso_end_wy": int(cfg["aso_years"]["end"]) if "aso_years" in cfg else None,
    }
    out["aso_meta_paths"] = meta_paths

    if exclude_all:
        obs_train_ds = obs_train_ds.drop_vars(exclude_all, errors="ignore")
        obs_test_raw_ds = obs_test_raw_ds.drop_vars(exclude_all, errors="ignore")
        obs_test_proc_ds = obs_test_proc_ds.drop_vars(exclude_all, errors="ignore")
        obs_selected_ds = obs_selected_ds.drop_vars(exclude_all, errors="ignore")


    aso_spatial_ds, dem_bin_ds, shape_proj_gdf, aso_tseries_ds = load_aso_data(
        aso_spatial_fpath=meta_paths["aso_spatial_fpath"],
        aso_tseries_fpath=meta_paths["aso_tseries_fpath"],
        demBin_fpath=meta_paths["demBin_fpath"],
        shape_fpath=meta_paths["shape_fpath"],
        shape_crs=meta_paths.get("shape_crs"),
    )

    out["datasets_loaded"] = {
        "status": "ok",
        "obs_train_vars_n": len(obs_train_ds.data_vars),
        "obs_test_raw_vars_n": len(obs_test_raw_ds.data_vars),
        "obs_test_proc_vars_n": len(obs_test_proc_ds.data_vars),
        "obs_train_vars_preview": list(obs_train_ds.data_vars)[:10],
        "aso_tseries_vars": list(aso_tseries_ds.data_vars),
        "aso_tseries_coords": list(aso_tseries_ds.coords),
    }
    out["datasets_loaded"]["obs_selected_vars_n"] = len(obs_selected_ds.data_vars)
    out["datasets_loaded"]["obs_selected_vars_preview"] = list(obs_selected_ds.data_vars)[:10]

    # Optional: give a time coverage preview if present
    if "time" in obs_selected_ds.coords and obs_selected_ds.sizes.get("time", 0) > 0:
        t0 = pd.to_datetime(obs_selected_ds["time"].values[0]).date()
        t1 = pd.to_datetime(obs_selected_ds["time"].values[-1]).date()
        out["datasets_loaded"]["obs_selected_time_range"] = f"{t0} → {t1}"
    else:
        out["datasets_loaded"]["obs_selected_time_range"] = None

    if "elev" in aso_tseries_ds.coords:
        out["aso_elev_values"] = [str(v) for v in aso_tseries_ds["elev"].values]
    else:
        out["aso_elev_values"] = None

    # --- Optional SnowModel zarrs (keep for later; do not block preprocessing) ---
    sm_train_ds = None
    sm_test_ds = None
    if exists["snowmodel_train_zarr"]:
        sm_train_ds = xr.open_zarr(str(paths["snowmodel_train_zarr"]), consolidated=False)
    if exists["snowmodel_test_zarr"]:
        sm_test_ds = xr.open_zarr(str(paths["snowmodel_test_zarr"]), consolidated=False)

    out["datasets_loaded"]["sm_train_loaded"] = sm_train_ds is not None
    out["datasets_loaded"]["sm_test_loaded"] = sm_test_ds is not None

    # -----------------------------
    # Phase 2B: Build tables up to ~line 286
    # -----------------------------
    out["_stage"] = "phase2b_build_tables"

    aso_var = pick_aso_swe_var(aso_tseries_ds)
    # elev_idx = band_to_elev_idx(cfg, band)
    elev_values = out["aso_elev_values"]
    elev_idx = band_to_elev_idx_from_aso(elev_values, band)

    out["aso_var_used"] = aso_var
    out["elev_idx"] = elev_idx

    obs_train_lst = dataset_to_list(obs_train_ds)

    (
        aso_spatial_test,
        aso_tseries_test,
        aso_spatial_train,
        aso_tseries_train,
        insitu_train,
        insitu_qa,
    ) = train_test_split(
        aso_spatial_data=aso_spatial_ds,
        aso_tseries_data=aso_tseries_ds,
        insitu_data=obs_train_lst,
        test_water_year=water_year,
    )

    # Note: your combine_aso_insitu expects a DataArray with coords 'date' + 'elev'
    df_sum_total = combine_aso_insitu(
        insitu_data=insitu_train,
        aso_tseries=aso_tseries_ds[aso_var],
        elev_idx=elev_idx,
    )

    drop_na_df, all_pils, baseline_pils = generate_drop_NaNs_table(
        df_summary_table=df_sum_total,
        obs_data=insitu_qa,
    )
    # load melt threshold and imputation csv files 
    melt_threshold_df, pillow_imputation_df, snowmodel_imputation_df = load_melt_imputation_tables(
        melt_threshold_dir=str(paths["melt_threshold_dir"]),
        pillow_imputation_dir=str(paths["imputation_dir"]),
        water_year=water_year
    ) 


    # Keep this available for later plots/debug (not always needed)
    # historic_vals_df = create_qa_tables(insitu_train, missing_stations=[], isQA=False)

    out["_stage"] = "phase2b_tables_built"
    out["artifacts"] = {
        "df_sum_total_shape": tuple(df_sum_total.shape),
        "drop_na_df_shape": tuple(drop_na_df.shape),
        "time_range": (
            df_sum_total["time"].min(),
            df_sum_total["time"].max(),
        ),
        "elev_idx": elev_idx,
        "aso_elev_label": band,
        "n_all_pillows": len(all_pils),
        "n_baseline_pillows": len(baseline_pils),
        "baseline_fraction": len(baseline_pils) / len(all_pils),
        "baseline_pils_preview": baseline_pils[:15],
    }

    out["df_sum_total"] = df_sum_total
    out["drop_na_df"] = drop_na_df

    out["baseline_pils"] = baseline_pils
    out["all_pils"] = all_pils

    out["obs_test_raw_ds"] = obs_test_raw_ds
    out["obs_test_proc_ds"] = obs_test_proc_ds

    out["aso_tseries_ds"] = aso_tseries_ds

    # after: obs_train_lst = dataset_to_list(obs_train_ds)
    out["obs_train_lst"] = obs_train_lst

    # keep these too
    out["aso_tseries_da"] = aso_tseries_ds[aso_var]   # DataArray
    out["start_wy"] = int(cfg["aso_years"]["start"])
    out["end_wy"] = int(cfg["aso_years"]["end"])
    out["baseline_pils"] = baseline_pils
    out["all_pils"] = all_pils

    # for daily QA on a selected date (WY-specific)
    out["obs_test_raw_lst"] = dataset_to_list(obs_test_raw_ds)
    out["obs_test_proc_lst"] = dataset_to_list(obs_test_proc_ds)

    # extra csv paths.
    # out["imputation_dir"] = str(paths["imputation_dir"])
    # out["melt_threshold_dir"] = str(paths["melt_threshold_dir"])

    # imputation and melt threshold tables.
    out["melt_threshold_df"] = melt_threshold_df
    out["pillow_imputation_df"] = pillow_imputation_df
    out["snowmodel_imputation_df"] = snowmodel_imputation_df



    return out
