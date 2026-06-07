

import numpy as np
import pandas as pd
import xarray as xr
import copy
from typing import List, Dict, Tuple, Optional


def build_sm_pillow_training_ds_slice_first(
    df_corr: pd.DataFrame,
    ds_hrrr: xr.Dataset,
    obs_data_hist: List[xr.DataArray],
    wy_start: int = 2017,
    wy_end: int = 2025,
    var_grid: str = "swed",
    resample_daily: bool = True,
    grid_units_to_mm: bool = True,
    interp_grid_time: bool = True,
    out_path: Optional[str] = None,
    out_format: str = "zarr",
) -> xr.Dataset:
    """
    Efficient version: slice relevant grid cells first (time,pil), THEN resample daily.
    Avoids resampling the full 3D grid.
    """

    t0 = np.datetime64(f"{wy_start-1}-10-01")
    t1 = np.datetime64(f"{wy_end}-10-02")

    # only keep variable + time range
    ds = ds_hrrr[[var_grid]].sel(time=slice(t0, t1))

    # Drop conflicting grid lat/lon coords/vars early (safe)
    for nm in ("xlat", "xlon", "lat", "lon"):
        if nm in ds.coords:
            ds = ds.reset_coords(nm, drop=True)
        if nm in ds.variables and nm not in ds.dims:
            ds = ds.drop_vars(nm, errors="ignore")

    def _to_int_or_nan(x):
        try:
            if pd.isna(x):
                return np.nan
            return int(x)
        except Exception:
            return np.nan

    pil_ids = df_corr["pil_id"].astype(str).tolist()

    best_idx   = np.array([_to_int_or_nan(x) for x in df_corr["best_idx"].values], dtype=float)
    best_idy   = np.array([_to_int_or_nan(x) for x in df_corr["best_idy"].values], dtype=float)
    second_idx = np.array([_to_int_or_nan(x) for x in df_corr["second_idx"].values], dtype=float)
    second_idy = np.array([_to_int_or_nan(x) for x in df_corr["second_idy"].values], dtype=float)
    third_idx  = np.array([_to_int_or_nan(x) for x in df_corr["third_idx"].values], dtype=float)
    third_idy  = np.array([_to_int_or_nan(x) for x in df_corr["third_idy"].values], dtype=float)

    good = np.isfinite(best_idx) & np.isfinite(best_idy)
    pil_good = [p for p, g in zip(pil_ids, good) if g]
    if len(pil_good) == 0:
        raise ValueError("No pillows in df_corr have finite best_idx/best_idy.")

    def _isel_points(idx_arr, idy_arr, name: str) -> xr.DataArray:
        idx = idx_arr[good].astype(int)
        idy = idy_arr[good].astype(int)

        da = ds[var_grid].isel(
            east_west=xr.DataArray(idx, dims="pil", coords={"pil": pil_good}),
            south_north=xr.DataArray(idy, dims="pil", coords={"pil": pil_good}),
        ).rename(name)

        # unit conversion after slicing (cheap)
        if grid_units_to_mm:
            da = da * 1000.0
            da.attrs["units"] = "mm"

        return da

    # Slice first (time, pil) — still sub-daily at this point
    swed_best = _isel_points(best_idx, best_idy, "swed_best")

    # For second/third: if missing, use best indices as placeholder then mask to NaN
    swed_second = _isel_points(
        np.where(np.isfinite(second_idx), second_idx, best_idx),
        np.where(np.isfinite(second_idy), second_idy, best_idy),
        "swed_second",
    )
    swed_third = _isel_points(
        np.where(np.isfinite(third_idx), third_idx, best_idx),
        np.where(np.isfinite(third_idy), third_idy, best_idy),
        "swed_third",
    )

    # Mask missing second/third
    sec_missing = ~(np.isfinite(second_idx[good]) & np.isfinite(second_idy[good]))
    thr_missing = ~(np.isfinite(third_idx[good]) & np.isfinite(third_idy[good]))
    if np.any(sec_missing):
        swed_second.loc[dict(pil=np.array(pil_good)[sec_missing])] = np.nan
    if np.any(thr_missing):
        swed_third.loc[dict(pil=np.array(pil_good)[thr_missing])] = np.nan

    # NOW resample daily on small (time,pil) arrays
    if resample_daily:
        swed_best = swed_best.resample(time="1D").mean()
        swed_second = swed_second.resample(time="1D").mean()
        swed_third = swed_third.resample(time="1D").mean()

    # Build obs (time,pil)
    obs_map = {da.name: da for da in obs_data_hist}
    obs_list = []
    for pil in pil_good:
        if pil not in obs_map:
            continue
        da = obs_map[pil].sel(time=slice(t0, t1))
        if resample_daily:
            da = da.resample(time="1D").mean()
        da = da.rename("swe_obs").assign_coords(pil=pil).expand_dims("pil")
        obs_list.append(da)

    if len(obs_list) == 0:
        raise ValueError("No matching pillows found in obs_data_hist for pil_good.")

    swe_obs = xr.concat(obs_list, dim="pil").transpose("time", "pil")
    swe_obs.attrs["units"] = "mm"

    out = xr.Dataset(
        {
            "swed_best": swed_best,
            "swed_second": swed_second,
            "swed_third": swed_third,
            "swe_obs": swe_obs,
        }
    )

    # Align to obs time
    out = out.sel(time=swe_obs.time)

    # Interpolate missing grid values along time (now cheap)
    if interp_grid_time:
        for v in ("swed_best", "swed_second", "swed_third"):
            out[v] = out[v].interpolate_na(dim="time")

    # Save
    if out_path is not None:
        if out_format.lower() == "zarr":
            out.to_zarr(out_path, mode="w")
        elif out_format.lower() in ("nc", "netcdf"):
            out.to_netcdf(out_path)
        else:
            raise ValueError("out_format must be 'zarr' or 'netcdf'.")

    return out


def _drop_grid_xy(da: xr.DataArray) -> xr.DataArray:
    """Drop 2D grid coords/vars that will conflict across points."""
    # coords OR data_vars can both show up depending on how ds was built
    for name in ("xlat", "xlon", "lat", "lon"):
        if name in da.coords:
            da = da.reset_coords(name, drop=True)
        if name in da.variables and name not in da.dims:
            da = da.drop_vars(name, errors="ignore")
    return da


# -------------------------
# Helpers
# -------------------------

def _fmt(x, nd=1):
    return f"{x:.{nd}f}" if np.isfinite(x) else "nan"


def _ols_pred_sepred(
    X_train: np.ndarray,   # (n, k) features
    y_train: np.ndarray,   # (n,)
    x0: np.ndarray,        # (k,)
) -> Tuple[float, float, float, int]:
    """
    OLS y ~ 1 + X
    Returns:
      yhat, sepred (for a NEW observation), sigma, ntrain
    """
    # drop any rows with non-finite values
    m = np.isfinite(y_train) & np.all(np.isfinite(X_train), axis=1)
    X = X_train[m]
    y = y_train[m]
    n = X.shape[0]
    if n < 10:
        return np.nan, np.nan, np.nan, n

    # intercept
    Xi = np.c_[np.ones(n), X]     # (n, p)
    x0i = np.r_[1.0, x0]          # (p,)

    # solve
    beta, *_ = np.linalg.lstsq(Xi, y, rcond=None)
    yhat = float(x0i @ beta)

    resid = y - (Xi @ beta)
    sse = float((resid ** 2).sum())
    p = Xi.shape[1]
    dof = n - p
    if dof <= 0:
        return yhat, np.nan, np.nan, n

    sigma = np.sqrt(sse / dof)

    try:
        XtX_inv = np.linalg.inv(Xi.T @ Xi)
        h0 = float(x0i @ XtX_inv @ x0i.T)
        sepred = sigma * np.sqrt(1.0 + h0)  # new observation prediction SE
    except np.linalg.LinAlgError:
        sepred = np.nan

    return yhat, sepred, sigma, n


def create_qa_tables(obs_data, missing_stations, isQA=True):
    """
    Your same helper (copied for drop-in convenience).
    Returns:
      all_df (time index), qa_df (same shape, zeros) if isQA else all_df only
    """
    count = 0
    for i in range(0, len(obs_data)):
        df = obs_data[i].to_dataframe().reset_index()
        df["time"] = df["time"].dt.date
        df = df.set_index("time")
        if count == 0:
            all_df = copy.deepcopy(df)
        else:
            all_df = pd.merge(all_df, df, how="left", on="time")
        count += 1

    all_df = all_df.reset_index()
    all_df["time"] = pd.to_datetime(all_df["time"])
    all_df = all_df.set_index("time")

    if isQA:
        qa_df = copy.deepcopy(all_df)
        for col in all_df.columns:
            qa_df[col] = 0
        for pil in missing_stations:
            qa_df[pil] = 0
            all_df[pil] = np.nan
        return all_df, qa_df
    else:
        return all_df


# -------------------------
# SnowModel daily predictions + PI
# -------------------------

def daily_predictions_pi_snowmodel(
    tstamp: np.datetime64,
    train_ds: xr.Dataset,          # (time, pil), vars swed_best/second/third, swe_obs
    pil_list: List[str],
    features: Tuple[str, str, str] = ("swed_best", "swed_second", "swed_third"),
    target: str = "swe_obs",
    min_train: int = 60,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Returns 1-row DataFrames keyed by pillow:
      df_yhat    : predicted SWE (mm)
      df_sepred  : prediction SE for NEW obs (mm)
      df_sigma   : residual sigma (mm)
      df_ntrain  : training sample count
    """
    t0 = pd.to_datetime(tstamp)

    yhat_out, sepred_out, sigma_out, n_out = {}, {}, {}, {}

    # Ensure we can access time
    if "time" not in train_ds.dims:
        raise ValueError("train_ds must have a 'time' dimension.")
    if "pil" not in train_ds.dims:
        raise ValueError("train_ds must have a 'pil' dimension.")

    for pil in pil_list:
        yhat_out[pil] = np.nan
        sepred_out[pil] = np.nan
        sigma_out[pil] = np.nan
        n_out[pil] = 0

        if pil not in train_ds["pil"].values:
            continue

        # pull (time,) series for this pil
        ds_p = train_ds.sel(pil=pil)

        # require x0 exists at t0 for all features
        if t0 not in pd.to_datetime(ds_p["time"].values):
            continue

        # x0
        x0 = []
        x0_bad = False
        for f in features:
            v = ds_p[f].sel(time=t0).values
            if not np.isfinite(v):
                x0_bad = True
                break
            x0.append(float(v))
        if x0_bad:
            continue
        x0 = np.array(x0, dtype=float)

        # training excludes today
        # build aligned arrays
        df = pd.DataFrame(
            {
                "y": ds_p[target].to_series(),
                features[0]: ds_p[features[0]].to_series(),
                features[1]: ds_p[features[1]].to_series(),
                features[2]: ds_p[features[2]].to_series(),
            }
        )

        df = df.dropna()
        df = df.loc[df.index != t0]

        ntrain = len(df)
        n_out[pil] = ntrain
        if ntrain < max(min_train, 10):
            continue

        X_train = df[[features[0], features[1], features[2]]].to_numpy(float)
        y_train = df["y"].to_numpy(float)

        yhat, sepred, sigma, n_eff = _ols_pred_sepred(X_train, y_train, x0)
        if np.isfinite(yhat) and yhat < 0:
            yhat = 0.0

        yhat_out[pil] = yhat
        sepred_out[pil] = sepred
        sigma_out[pil] = sigma
        n_out[pil] = n_eff

    df_yhat = pd.DataFrame([yhat_out])
    df_sepred = pd.DataFrame([sepred_out])
    df_sigma = pd.DataFrame([sigma_out])
    df_ntrain = pd.DataFrame([n_out])

    return df_yhat, df_sepred, df_sigma, df_ntrain


# -------------------------
# QA flagging (v2-style)
# -------------------------

def daily_qa_flag_snowmodel_v2(
    historic_vals_df: pd.DataFrame,     # (time, pil) QA'd historic SWE (mm)
    current_vals_df: pd.DataFrame,      # 1-row (time index not required), columns pil
    previous_vals_df: pd.DataFrame,     # 1-row
    df_sepred: pd.DataFrame,            # 1-row, columns pil
    df_yhat: pd.DataFrame,              # 1-row, columns pil
    tstamp: np.datetime64,
    printOutput: bool = True,
    # dswe envelope
    q_lo: float = 0.005,
    q_hi: float = 0.995,
    engineering_buff: float = 1.5,
    # PI threshold controls
    z: float = 2.0,          # ~95%
    se_buff: float = 1.2,
    min_abs_mm: float = 25.0,
    min_rel: float = 0.05,
    # hard checks
    max_swe_mm: float = 3000.0,
    # melt-gate: when snowmodel predicts <snow_present_mm, suppress the pred residual check
    # to avoid flagging lingering pillow snow as a disagreement with near-zero predictions.
    snow_present_mm: float = None,
) -> pd.DataFrame:
    """
    Returns 1-row df with time + per-pillow QA flag {0,1}.
    QA=0 OK, QA=1 suspect.
    Print output matches your qa_static style pretty closely.
    """
    qa = {"time": tstamp}

    hist_diff = historic_vals_df.diff()

    for pil in current_vals_df.columns:
        cur_val = float(current_vals_df[pil].values) if pil in current_vals_df else np.nan
        prev_val = float(previous_vals_df[pil].values) if pil in previous_vals_df else np.nan
        dcur = cur_val - prev_val

        # Hard checks
        missing_today_flag = np.isnan(cur_val)
        phys_flag = (np.isfinite(cur_val) and ((cur_val < 0.0) or (cur_val > max_swe_mm)))

        # NOTE: nan_transition and dswe-envelope checks were removed from snowmodel —
        # those generic data-integrity checks are now owned by qa_static only so that the
        # 2-of-3 majority requires real corroboration rather than deterministic agreement
        # on shared reasons.

        # prediction disagreement using PI-style threshold
        yhat = float(df_yhat[pil].values) if pil in df_yhat else np.nan
        se_pred = float(df_sepred[pil].values) if pil in df_sepred else np.nan

        pred_flag = False
        thr = np.nan

        # compute threshold whenever we have prediction uncertainty
        if np.isfinite(yhat) and np.isfinite(se_pred):
            thr = max(z * se_buff * se_pred, min_abs_mm, min_rel * max(yhat, 1.0))

        # only evaluate pred_flag if we also have an observation today.
        # melt-gate: skip the pred check when snowmodel predicts <snow_present_mm so we
        # don't flag lingering pillow snow as disagreement with near-zero predictions.
        if (not missing_today_flag) and np.isfinite(thr):
            if (snow_present_mm is not None) and np.isfinite(yhat) and (yhat < float(snow_present_mm)):
                pred_flag = False
            else:
                pred_flag = (abs(cur_val - yhat) > thr)


        suspect = bool(
            missing_today_flag or phys_flag or pred_flag
        )
        qa[pil] = 1 if suspect else 0

        if printOutput:
            # match your qa_static-ish output
            print(f"{pil}: QUALITYSCORE {qa[pil]}")
            print(f"  current {_fmt(cur_val)} mm; yhat {_fmt(yhat)} mm; thr {_fmt(thr)} mm")
            print(f"  previous {_fmt(prev_val)} mm; diff {_fmt(dcur)} mm")
            print(f"  hard_checks: missing_today_flag={missing_today_flag} phys_flag={phys_flag} pred_flag={pred_flag} (max_swe_mm={max_swe_mm:.0f})")
            print("")

    df_qa = pd.DataFrame([qa])
    df_qa["time"] = pd.to_datetime(df_qa["time"])
    return df_qa


# -------------------------
# Main entrypoint (drop-in style)
# -------------------------

def run_snowmodel_qa(
    t_idx: int,
    obs_data_test: List,          # list[xr.DataArray], SWE mm
    obs_data_qa: List,            # list[xr.DataArray], QA'd historic SWE mm
    user_qa_level: int,           # ignored but kept for signature compatibility
    baseline_pils: List[str],
    pil_corr: Dict[str, List[str]],  # ignored but kept for compatibility
    train_ds: xr.Dataset,         # snowmodel training DS (time,pil)
    engineering_buff: float = 1.5,
    se_buff: float = 1.2,
    printOutput: bool = False,
    # envelope knobs
    q_lo: float = 0.005,
    q_hi: float = 0.995,
    # PI knobs
    z: float = 2.0,
    min_abs_mm: float = 25.0,
    min_rel: float = 0.05,
    # hard checks
    max_swe_mm: float = 3000.0,
    # training
    min_train: int = 60,
) -> Tuple[pd.DataFrame, List[str], List[str], pd.DataFrame]:
    """
    Signature/output matches your old usage pattern closely.

    Returns:
      current_vals_df, all_pils_QA, baseline_pils_, df_qa_table
    where df_qa_table has QA flags {0,1} and time.
    """
    # identify missing stations today
    obs_data_current_day = [da.isel({"time": t_idx}) for da in obs_data_test]
    missing_stations = [da.name for da in obs_data_current_day if da.isnull()]

    wateryear_vals_df, wateryear_lookup_df = create_qa_tables(obs_data_test, missing_stations, isQA=True)
    current_vals_df = pd.DataFrame(wateryear_vals_df.iloc[t_idx]).T
    previous_vals_df = pd.DataFrame(wateryear_vals_df.iloc[t_idx - 1]).T if t_idx > 0 else current_vals_df.copy()

    historic_vals_df = create_qa_tables(obs_data_qa, missing_stations, isQA=False)

    # consistent columns
    sorted_pillows = previous_vals_df.columns.to_list()
    for df in (current_vals_df, previous_vals_df):
        for c in sorted_pillows:
            if c not in df.columns:
                df[c] = np.nan
        df[:] = df[sorted_pillows].values

    # timestamp
    tstamp = obs_data_current_day[0].time.values

    # predictions from snowmodel
    df_yhat, df_sepred, df_sigma, df_ntrain = daily_predictions_pi_snowmodel(
        tstamp=tstamp,
        train_ds=train_ds,
        pil_list=sorted_pillows,
        min_train=min_train,
    )

    # QA flagging
    df_qa_table = daily_qa_flag_snowmodel_v2(
        historic_vals_df=historic_vals_df,
        current_vals_df=current_vals_df,
        previous_vals_df=previous_vals_df,
        df_sepred=df_sepred,
        df_yhat=df_yhat,
        tstamp=tstamp,
        printOutput=printOutput,
        q_lo=q_lo,
        q_hi=q_hi,
        engineering_buff=engineering_buff,
        z=z,
        se_buff=se_buff,
        min_abs_mm=min_abs_mm,
        min_rel=min_rel,
        max_swe_mm=max_swe_mm,
    )

    # OK pillows for inference: numeric today + QA=0
    pillows_w_numeric_vals = current_vals_df.dropna(axis=1).columns.to_list()
    all_pils_QA = [p for p in sorted_pillows if (p in pillows_w_numeric_vals) and (df_qa_table.at[0, p] == 0)]

    baseline_pils_ = [p for p in baseline_pils if p in all_pils_QA]

    return current_vals_df, all_pils_QA, baseline_pils_, df_qa_table

# ============================================================================
# PATCH (2025-12-27): ops-ready SnowModel QA with test_ds + pred_method
# - train_ds: historical dataset used for fitting
# - test_ds : inference dataset used to supply "today" predictors (isel by t_idx)
# - pred_method: "base" | "seasonal" | "detrend"
# - season_period: period for sin/cos water_doy encoding
# - snow_present_mm: optional gate to suppress transient melt-out false positives
# - return_long: return (df_simple, df_detail) in addition to legacy df_qa_table
# ============================================================================

def _water_doy_sincos(t: pd.Timestamp, period: float = 365.25) -> tuple:
    """Return (sin, cos) for water day-of-year (WY starts Oct 1)."""
    y = t.year if t.month >= 10 else (t.year - 1)
    wy0 = pd.Timestamp(f"{y}-10-01")
    wdoy = int((t.normalize() - wy0).days) + 1
    ang = 2.0 * np.pi * (float(wdoy) / float(period))
    return float(np.sin(ang)), float(np.cos(ang))

def _ols_pred_sepred(
    X_train: np.ndarray,   # (n, k)
    y_train: np.ndarray,   # (n,)
    x0: np.ndarray,        # (k,)
) -> Tuple[float, float, float, int, float]:
    """
    Numerically stable OLS y ~ 1 + X.
    Returns:
      yhat, sepred(new obs), sigma, ntrain, h0
    Notes:
      - Uses lstsq + pinv -> avoids "Singular matrix" crashes.
    """
    m = np.isfinite(y_train) & np.all(np.isfinite(X_train), axis=1)
    X = np.asarray(X_train[m], dtype=float)
    y = np.asarray(y_train[m], dtype=float)
    n = int(X.shape[0])
    if n < 10:
        return np.nan, np.nan, np.nan, n, np.nan

    # design
    Xi = np.column_stack([np.ones(n, dtype=float), X])  # (n, p)
    x0i = np.concatenate([[1.0], np.asarray(x0, dtype=float).ravel()])  # (p,)
    p = int(Xi.shape[1])

    # fit
    try:
        beta, *_ = np.linalg.lstsq(Xi, y, rcond=None)
        yhat = float(x0i @ beta)
    except Exception:
        return np.nan, np.nan, np.nan, n, np.nan

    # residual sigma
    yhat_all = Xi @ beta
    resid = y - yhat_all
    sse = float(np.sum(resid**2))
    dof = n - p
    if dof <= 0:
        return yhat, np.nan, np.nan, n, np.nan

    sigma = float(np.sqrt(sse / dof))

    # leverage + prediction SE
    try:
        XtX_inv = np.linalg.pinv(Xi.T @ Xi)
        h0 = float(x0i @ XtX_inv @ x0i.T)
        sepred = float(sigma * np.sqrt(1.0 + h0))
    except Exception:
        h0 = np.nan
        sepred = np.nan

    return yhat, sepred, sigma, n, h0

def daily_predictions_pi_snowmodel(
    tstamp: np.datetime64,
    train_ds: xr.Dataset,                 # fit DS (time,pil)
    pil_list: List[str],
    features: Tuple[str, str, str] = ("swed_best", "swed_second", "swed_third"),
    target: str = "swe_obs",
    min_train: int = 60,
    # NEW:
    pred_ds: Optional[xr.Dataset] = None, # if provided, supplies x0 at t_idx
    t_idx: Optional[int] = None,
    pred_method: str = "base",            # base|seasonal|detrend
    season_period: float = 365.25,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Returns 1-row DataFrames keyed by pillow:
      df_yhat    : predicted SWE (mm)
      df_sepred  : prediction SE for NEW obs (mm)
      df_sigma   : residual sigma (mm)
      df_ntrain  : training sample count
      df_h0      : leverage (dimensionless)
    """
    t0 = pd.to_datetime(tstamp)

    yhat_out = {}
    sepred_out = {}
    sigma_out = {}
    n_out = {}
    h0_out = {}

    # choose predictor dataset for x0
    if pred_ds is None:
        pred_ds = train_ds

    for pil in pil_list:
        pil = str(pil).strip()
        if "pil" not in train_ds.dims:
            continue
        if pil not in train_ds["pil"].values.astype(str):
            continue

        # pull training series
        ds_tr = train_ds.sel(pil=pil)

        # build training df
        df = pd.DataFrame({"y": ds_tr[target].to_series()})
        for f in features:
            df[f] = ds_tr[f].to_series()

        # seasonal columns
        if pred_method in ("seasonal", "detrend"):
            tt = pd.to_datetime(df.index)
            sincos = np.array([_water_doy_sincos(pd.Timestamp(t), period=season_period) for t in tt], dtype=float)
            df["wdoy_sin"] = sincos[:, 0]
            df["wdoy_cos"] = sincos[:, 1]

        # exclude any overlap day if t0 exists in train
        if t0 in df.index:
            df_tr = df.drop(index=t0)
        else:
            df_tr = df

        # minimum training
        df_tr = df_tr.dropna()
        if df_tr.shape[0] < min_train:
            continue

        # build x0 from pred_ds
        ds_pr = pred_ds.sel(pil=pil)
        try:
            if t_idx is not None and "time" in ds_pr.dims:
                # t_idx is authoritative in ops
                x0_time = pd.to_datetime(ds_pr["time"].values[int(t_idx)])
                x0_row = ds_pr.isel(time=int(t_idx))
            else:
                x0_time = t0
                x0_row = ds_pr.sel(time=t0)
        except Exception:
            continue

        # predictors at x0
        try:
            x0_base = np.array([float(x0_row[f].values) for f in features], dtype=float)
        except Exception:
            continue
        if not np.all(np.isfinite(x0_base)):
            continue

        # ----- pred_method handling -----
        if pred_method == "base":
            X_train = df_tr[list(features)].to_numpy(dtype=float)
            y_train = df_tr["y"].to_numpy(dtype=float)
            x0 = x0_base

            yhat, sepred, sigma, n_eff, h0 = _ols_pred_sepred(X_train, y_train, x0)

        elif pred_method == "seasonal":
            X_train = df_tr[list(features) + ["wdoy_sin", "wdoy_cos"]].to_numpy(dtype=float)
            y_train = df_tr["y"].to_numpy(dtype=float)

            sin0, cos0 = _water_doy_sincos(pd.Timestamp(x0_time), period=season_period)
            x0 = np.concatenate([x0_base, [sin0, cos0]]).astype(float)

            yhat, sepred, sigma, n_eff, h0 = _ols_pred_sepred(X_train, y_train, x0)

        elif pred_method == "detrend":
            # 1) Fit seasonal baseline on TRAIN: y ~ sin/cos
            Xs = df_tr[["wdoy_sin", "wdoy_cos"]].to_numpy(dtype=float)
            y = df_tr["y"].to_numpy(dtype=float)
            # season fit
            try:
                beta_s, *_ = np.linalg.lstsq(np.column_stack([np.ones(len(y)), Xs]), y, rcond=None)
            except Exception:
                continue

            yhat_season_tr = (np.column_stack([np.ones(len(y)), Xs]) @ beta_s)
            y_resid = y - yhat_season_tr

            # 2) Residualize predictors on TRAIN: each x ~ sin/cos
            X_pred_res = []
            for f in features:
                xf = df_tr[f].to_numpy(dtype=float)
                try:
                    beta_x, *_ = np.linalg.lstsq(np.column_stack([np.ones(len(xf)), Xs]), xf, rcond=None)
                    xf_hat = (np.column_stack([np.ones(len(xf)), Xs]) @ beta_x)
                    X_pred_res.append(xf - xf_hat)
                except Exception:
                    X_pred_res.append(np.full_like(xf, np.nan))
            X_pred_res = np.column_stack(X_pred_res)

            # 3) Fit residual model: y_resid ~ X_pred_res
            m = np.isfinite(y_resid) & np.all(np.isfinite(X_pred_res), axis=1)
            if m.sum() < min_train:
                continue
            X_train = X_pred_res[m]
            y_train = y_resid[m]

            # Build x0 residual predictors using TRAIN seasonal betas for each x
            # Compute seasonal terms at x0
            sin0, cos0 = _water_doy_sincos(pd.Timestamp(x0_time), period=season_period)
            Xs0 = np.array([sin0, cos0], dtype=float)
            x0_res = []
            for j, f in enumerate(features):
                # refit beta_x on full TRAIN for this predictor (cheap, stable)
                xf = df_tr[f].to_numpy(dtype=float)
                beta_x, *_ = np.linalg.lstsq(np.column_stack([np.ones(len(xf)), Xs]), xf, rcond=None)
                xf0_hat = float(np.dot(np.array([1.0, Xs0[0], Xs0[1]]), beta_x))
                x0_res.append(float(x0_base[j] - xf0_hat))
            x0 = np.array(x0_res, dtype=float)

            # Residual prediction + PI
            yhat_resid, sepred, sigma, n_eff, h0 = _ols_pred_sepred(X_train, y_train, x0)

            # Add back seasonal baseline for y at x0
            y_season0 = float(np.dot(np.array([1.0, Xs0[0], Xs0[1]]), beta_s))
            yhat = y_season0 + (yhat_resid if np.isfinite(yhat_resid) else np.nan)

        else:
            raise ValueError("pred_method must be 'base', 'seasonal', or 'detrend'")

        if np.isfinite(yhat) and yhat < 0.0:
            yhat = 0.0

        yhat_out[pil] = yhat
        sepred_out[pil] = sepred
        sigma_out[pil] = sigma
        n_out[pil] = n_eff
        h0_out[pil] = h0

    df_yhat = pd.DataFrame([yhat_out])
    df_sepred = pd.DataFrame([sepred_out])
    df_sigma = pd.DataFrame([sigma_out])
    df_ntrain = pd.DataFrame([n_out])
    df_h0 = pd.DataFrame([h0_out])

    return df_yhat, df_sepred, df_sigma, df_ntrain, df_h0


def run_snowmodel_qa(
    t_idx: int,
    obs_data_test: List,
    obs_data_qa: List,
    user_qa_level: int,
    baseline_pils: List[str],
    pil_corr: Dict[str, List[str]],
    train_ds: xr.Dataset,
    # NEW:
    test_ds: Optional[xr.Dataset] = None,
    pred_method: str = "base",
    season_period: float = 365.25,
    engineering_buff: float = 1.5,
    se_buff: float = 1.2,
    printOutput: bool = False,
    q_lo: float = 0.005,
    q_hi: float = 0.995,
    z: float = 2.0,
    min_abs_mm: float = 25.0,
    min_rel: float = 0.05,
    max_swe_mm: float = 3000.0,
    min_train: int = 60,
    snow_present_mm: Optional[float] = None,
    return_long: bool = False,
) -> Tuple[pd.DataFrame, List[str], List[str], pd.DataFrame, Optional[pd.DataFrame]]:
    """
    Ops-ready SnowModel QA.

    Legacy return (return_long=False):
      current_vals_df, all_pils_QA, baseline_pils_, df_qa_table (wide)

    New (return_long=True):
      current_vals_df, all_pils_QA, baseline_pils_, df_simple, df_detail
    """
    # identify missing stations today
    obs_data_current_day = [da.isel({"time": t_idx}) for da in obs_data_test]
    missing_stations = [da.name for da in obs_data_current_day if da.isnull()]

    wateryear_vals_df, _ = create_qa_tables(obs_data_test, missing_stations, isQA=True)
    current_vals_df = pd.DataFrame(wateryear_vals_df.iloc[t_idx]).T
    previous_vals_df = pd.DataFrame(wateryear_vals_df.iloc[t_idx - 1]).T if t_idx > 0 else current_vals_df.copy()

    historic_vals_df = create_qa_tables(obs_data_qa, missing_stations, isQA=False)

    # consistent columns
    sorted_pillows = previous_vals_df.columns.to_list()
    for df in (current_vals_df, previous_vals_df):
        for c in sorted_pillows:
            if c not in df.columns:
                df[c] = np.nan
        df[:] = df[sorted_pillows].values

    # timestamp from obs list
    tstamp = obs_data_current_day[0].time.values

    # predictions: fit on train_ds, x0 from test_ds if provided
    pred_ds = test_ds if test_ds is not None else train_ds
    df_yhat, df_sepred, df_sigma, df_ntrain, df_h0 = daily_predictions_pi_snowmodel(
        tstamp=tstamp,
        train_ds=train_ds,
        pred_ds=pred_ds,
        t_idx=t_idx,
        pil_list=sorted_pillows,
        min_train=min_train,
        pred_method=pred_method,
        season_period=season_period,
    )

    # wide QA table using the original logic (no melt-gating)
    df_qa_wide = daily_qa_flag_snowmodel_v2(
        historic_vals_df=historic_vals_df,
        current_vals_df=current_vals_df,
        previous_vals_df=previous_vals_df,
        df_sepred=df_sepred,
        df_yhat=df_yhat,
        tstamp=tstamp,
        printOutput=printOutput,
        q_lo=q_lo,
        q_hi=q_hi,
        engineering_buff=engineering_buff,
        z=z,
        se_buff=se_buff,
        min_abs_mm=min_abs_mm,
        min_rel=min_rel,
        max_swe_mm=max_swe_mm,
    )

    # optional melt gating: adjust flags in a long detail table, and regenerate simple/wide flags
    if return_long:
        t0 = pd.to_datetime(tstamp)
        hist_diff = historic_vals_df.diff()

        rows = []
        for pil in sorted_pillows:
            pil_s = str(pil).strip()
            cur = float(current_vals_df[pil].values) if pil in current_vals_df else np.nan
            prev = float(previous_vals_df[pil].values) if pil in previous_vals_df else np.nan
            dcur = (cur - prev) if (np.isfinite(cur) and np.isfinite(prev)) else np.nan

            missing_today_flag = bool(np.isnan(cur))
            phys_flag = bool(np.isfinite(cur) and ((cur < 0.0) or (cur > max_swe_mm)))
            nan_transition_flag = bool(np.isnan(prev) and (not np.isnan(cur)))

            # dswe envelope
            dswe_flag = False
            dq_lo = dq_hi = np.nan
            if pil in hist_diff.columns:
                d = hist_diff[pil].to_numpy(dtype=float)
                d = d[np.isfinite(d)]
                if d.size >= 30 and np.isfinite(dcur):
                    dq_lo = float(np.quantile(d, q_lo))
                    dq_hi = float(np.quantile(d, q_hi))
                    dswe_flag = bool((dcur < dq_lo * engineering_buff) or (dcur > dq_hi * engineering_buff))

            yhat = float(df_yhat.get(pil, np.nan).values[0]) if pil in df_yhat.columns else np.nan
            sepred = float(df_sepred.get(pil, np.nan).values[0]) if pil in df_sepred.columns else np.nan
            sigma = float(df_sigma.get(pil, np.nan).values[0]) if pil in df_sigma.columns else np.nan
            ntrain = float(df_ntrain.get(pil, np.nan).values[0]) if pil in df_ntrain.columns else np.nan
            h0 = float(df_h0.get(pil, np.nan).values[0]) if pil in df_h0.columns else np.nan

            resid = (cur - yhat) if (np.isfinite(cur) and np.isfinite(yhat)) else np.nan

            thr = np.nan
            pred_flag = False
            pred_check_skipped = False
            pred_skip_reason = ""
            if np.isfinite(cur) and np.isfinite(yhat) and np.isfinite(sepred):
                thr = max(z * se_buff * sepred, min_abs_mm, min_rel * max(abs(yhat), 1.0))
                pred_flag = bool(abs(resid) > thr)
            else:
                pred_check_skipped = True
                if not np.isfinite(yhat):
                    pred_skip_reason = "missing_yhat"
                elif not np.isfinite(sepred):
                    pred_skip_reason = "missing_se_pred"
                else:
                    pred_skip_reason = "missing_cur"
            # melt gate (optional): suppress dswe/pred transient flags when snow not present
            melt_gate = False
            if snow_present_mm is not None and np.isfinite(cur) and np.isfinite(yhat):
                melt_gate = (cur < snow_present_mm) and (yhat < snow_present_mm)

            if melt_gate:
                dswe_flag = False
                pred_flag = False

            flag = int(
                missing_today_flag or phys_flag or nan_transition_flag or dswe_flag or pred_flag
            )

            reason_code = "OK"
            reason_detail = ""
            if missing_today_flag:
                reason_code = "MISSING_TODAY"
                reason_detail = "current SWE is NaN"
            elif phys_flag:
                reason_code = "PHYS_RANGE"
                reason_detail = f"cur_mm={cur:.1f} outside [0,{max_swe_mm:.0f}]"
            elif nan_transition_flag:
                reason_code = "NAN_TO_VALUE"
                reason_detail = "previous was NaN, current is finite"
            elif dswe_flag:
                reason_code = "DSWE_ENVELOPE"
                if np.isfinite(dcur) and np.isfinite(dq_lo) and np.isfinite(dq_hi):
                    reason_detail = f"diff_mm={dcur:.1f} outside [{dq_lo:.1f},{dq_hi:.1f}] * buff={engineering_buff:.2f}"
                else:
                    reason_detail = "diff_mm outside historic envelope"
            elif pred_flag:
                reason_code = "PRED_RESID_TOO_LARGE"
                if np.isfinite(resid) and np.isfinite(thr):
                    reason_detail = f"|resid|={abs(resid):.1f} > thr={thr:.1f}"
                else:
                    reason_detail = "prediction residual exceeded threshold"
            elif pred_check_skipped:
                reason_code = "PRED_CHECK_SKIPPED_LOW_CONFIDENCE"
                reason_detail = pred_skip_reason

            if melt_gate and flag == 0:
                # optional breadcrumb
                reason_code = "OK_MELT_GATE"
                reason_detail = "melt gate suppressed transient flag"
            rows.append(
                dict(
                    time=t0,
                    pillow=pil_s,
                    method="snowmodel",
                    flag=flag,
                    severity=int(flag),
                    QUALITYSCORE=int(flag),
                    # values
                    cur_mm=cur,
                    prev_mm=prev,
                    diff_mm=dcur,
                    # prediction
                    yhat_mm=yhat,
                    resid_mm=resid,
                    thr_mm=thr,
                    sigma_mm=sigma,
                    se_pred_mm=sepred,
                    ntrain=ntrain,
                    h0=h0,
                    # envelope
                    dq_lo=dq_lo,
                    dq_hi=dq_hi,
                    # flags
                    missing_today_flag=int(missing_today_flag),
                    phys_flag=int(phys_flag),
                    nan_transition_flag=int(nan_transition_flag),
                    dswe_flag=int(dswe_flag),
                    pred_flag=int(pred_flag),
                    pred_check_skipped=int(pred_check_skipped),
                    pred_skip_reason=pred_skip_reason,

                    melt_gate=int(bool(melt_gate)),
                    pred_method=str(pred_method),
                    reason_code=reason_code,
                    reason_detail=reason_detail,

                )
            )

        df_detail = pd.DataFrame(rows)
        df_simple = df_detail[["time", "pillow", "method", "flag"]].copy()

        # Determine which pillows are OK for inference (numeric today + flag==0)
        pillows_w_numeric_vals = current_vals_df.dropna(axis=1).columns.to_list()
        ok_today = df_detail.loc[df_detail["flag"] == 0, "pillow"].tolist()
        all_pils_QA = [p for p in sorted_pillows if (p in pillows_w_numeric_vals) and (p in ok_today)]
        baseline_pils_ = [p for p in baseline_pils if p in all_pils_QA]

        return current_vals_df, all_pils_QA, baseline_pils_, df_simple, df_detail

    # legacy path
    pillows_w_numeric_vals = current_vals_df.dropna(axis=1).columns.to_list()
    all_pils_QA = [p for p in sorted_pillows if (p in pillows_w_numeric_vals) and (df_qa_wide.at[0, p] == 0)]
    baseline_pils_ = [p for p in baseline_pils if p in all_pils_QA]

    return current_vals_df, all_pils_QA, baseline_pils_, df_qa_wide



# ==========================
# DROP-IN PATCH: seasonal-gated SnowModel predictions + long outputs
# (2026-01-22)
# - Adds test_ds, pred_method, season_period, snow_present_mm, return_long
# - For pred_method='seasonal', computes both base and seasonal fits, then gates
#   seasonal term off when local SnowModel predictors are flat/decreasing.
# - Adds df_simple/df_detail outputs when return_long=True.
# ==========================

from typing import Optional

def _water_doy_sincos(t: pd.Timestamp, period: float = 365.25) -> tuple:
    """Return (sin, cos) for water day-of-year (WY starts Oct 1)."""
    y = t.year if t.month >= 10 else (t.year - 1)
    wy0 = pd.Timestamp(f"{y}-10-01")
    wdoy = int((t.normalize() - wy0).days) + 1
    ang = 2.0 * np.pi * (float(wdoy) / float(period))
    return float(np.sin(ang)), float(np.cos(ang))

def _gate_seasonal_from_predictors(
    x_hist: np.ndarray,
    eps_mm: float = 0.0,
) -> bool:
    """
    Decide whether to ALLOW seasonal term based on recent predictor trend.
    Returns True if net accumulation (mean delta > eps_mm), else False.
    x_hist: (k_days, n_features) predictors history (best/second/third).
    """
    if x_hist.ndim != 2 or x_hist.shape[0] < 2:
        return True  # not enough info -> don't block
    d = np.diff(x_hist, axis=0)
    d_mean = float(np.nanmean(d))
    if not np.isfinite(d_mean):
        return True
    return d_mean > eps_mm

def daily_predictions_pi_snowmodel(
    tstamp: np.datetime64,
    train_ds: xr.Dataset,          # (time, pil)
    pil_list: List[str],
    features: Tuple[str, str, str] = ("swed_best", "swed_second", "swed_third"),
    target: str = "swe_obs",
    min_train: int = 60,
    *,
    test_ds: Optional[xr.Dataset] = None,
    pred_method: str = "base",          # "base" | "seasonal" | "detrend"
    season_period: float = 365.25,
    seasonal_gate: bool = True,
    seasonal_gate_k_days: int = 3,
    seasonal_gate_eps_mm: float = 0.0,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Extended version (overrides earlier def).

    Returns 1-row DataFrames keyed by pillow:
      df_yhat    : predicted SWE (mm)
      df_sepred  : prediction SE for NEW obs (mm)
      df_sigma   : residual sigma (mm)
      df_ntrain  : training sample count
      df_h0      : leverage for x0 (for diagnostics)
    """
    t0 = pd.to_datetime(tstamp)

    # if no test_ds provided, fall back to train_ds for x0 (legacy behavior)
    test_ds = train_ds if test_ds is None else test_ds

    yhat_out, sepred_out, sigma_out, n_out, h0_out = {}, {}, {}, {}, {}

    if "time" not in train_ds.dims or "pil" not in train_ds.dims:
        raise ValueError("train_ds must have dims ('time','pil').")
    if "time" not in test_ds.dims or "pil" not in test_ds.dims:
        raise ValueError("test_ds must have dims ('time','pil').")

    for pil in pil_list:
        yhat_out[pil] = np.nan
        sepred_out[pil] = np.nan
        sigma_out[pil] = np.nan
        n_out[pil] = 0
        h0_out[pil] = np.nan

        if pil not in train_ds["pil"].values or pil not in test_ds["pil"].values:
            continue

        ds_tr = train_ds.sel(pil=pil)
        ds_te = test_ds.sel(pil=pil)

        # require x0 exists at t0 for all base features
        te_times = pd.to_datetime(ds_te["time"].values)
        if t0 not in te_times:
            continue

        # x0 base
        x0_base = []
        for f in features:
            v = ds_te[f].sel(time=t0).values
            if not np.isfinite(v):
                x0_base = None
                break
            x0_base.append(float(v))
        if x0_base is None:
            continue
        x0_base = np.asarray(x0_base, dtype=float)

        # Build training DF (exclude today)
        df = pd.DataFrame(
            {
                "y": ds_tr[target].to_series(),
                features[0]: ds_tr[features[0]].to_series(),
                features[1]: ds_tr[features[1]].to_series(),
                features[2]: ds_tr[features[2]].to_series(),
            }
        ).dropna()
        df = df.loc[df.index != t0]
        ntrain = int(len(df))
        n_out[pil] = ntrain
        if ntrain < max(min_train, 10):
            continue

        X_base = df[[features[0], features[1], features[2]]].to_numpy(float)
        y_train = df["y"].to_numpy(float)

        # --- base fit
        yhat_base, sepred_base, sigma_base, n_eff_base, h0_base = _ols_pred_sepred(X_base, y_train, x0_base)

        if pred_method.lower() == "seasonal":
            sin0, cos0 = _water_doy_sincos(t0, period=season_period)
            x0_seas = np.concatenate([x0_base, [sin0, cos0]]).astype(float)

            idx = pd.to_datetime(df.index)
            sc = np.array([_water_doy_sincos(pd.Timestamp(tt), period=season_period) for tt in idx], dtype=float)
            X_seas = np.column_stack([X_base, sc[:, 0], sc[:, 1]])

            yhat_seas, sepred_seas, sigma_seas, n_eff_seas, h0_seas = _ols_pred_sepred(X_seas, y_train, x0_seas)

            # seasonal gating based on recent predictor trend from test_ds
            use_seasonal = True
            if seasonal_gate:
                k = max(2, int(seasonal_gate_k_days))
                te_df = pd.DataFrame(
                    {
                        features[0]: ds_te[features[0]].to_series(),
                        features[1]: ds_te[features[1]].to_series(),
                        features[2]: ds_te[features[2]].to_series(),
                    }
                )
                te_df = te_df.loc[:t0].tail(k).dropna()
                if len(te_df) >= 2:
                    use_seasonal = _gate_seasonal_from_predictors(te_df.to_numpy(float), eps_mm=seasonal_gate_eps_mm)

            if use_seasonal and np.isfinite(yhat_seas) and np.isfinite(sepred_seas):
                yhat, sepred, sigma, n_eff, h0 = yhat_seas, sepred_seas, sigma_seas, n_eff_seas, h0_seas
            else:
                yhat, sepred, sigma, n_eff, h0 = yhat_base, sepred_base, sigma_base, n_eff_base, h0_base

        elif pred_method.lower() == "detrend":
            X_mean = np.nanmean(X_base, axis=0)
            y_mean = float(np.nanmean(y_train))
            Xb = X_base - X_mean
            x0b = x0_base - X_mean
            yb = y_train - y_mean
            yhat_b, sepred_b, sigma_b, n_eff_b, h0_b = _ols_pred_sepred(Xb, yb, x0b)
            if np.isfinite(yhat_b):
                yhat_b = yhat_b + y_mean
            yhat, sepred, sigma, n_eff, h0 = yhat_b, sepred_b, sigma_b, n_eff_b, h0_b
        else:
            yhat, sepred, sigma, n_eff, h0 = yhat_base, sepred_base, sigma_base, n_eff_base, h0_base

        if np.isfinite(yhat) and yhat < 0:
            yhat = 0.0

        yhat_out[pil] = yhat
        sepred_out[pil] = sepred
        sigma_out[pil] = sigma
        n_out[pil] = n_eff
        h0_out[pil] = h0

    df_yhat = pd.DataFrame([yhat_out])
    df_sepred = pd.DataFrame([sepred_out])
    df_sigma = pd.DataFrame([sigma_out])
    df_ntrain = pd.DataFrame([n_out])
    df_h0 = pd.DataFrame([h0_out])

    return df_yhat, df_sepred, df_sigma, df_ntrain, df_h0


def run_snowmodel_qa(
    t_idx: int,
    obs_data_test: List,
    obs_data_qa: List,
    user_qa_level: int,
    baseline_pils: List[str],
    pil_corr: Dict[str, List[str]],
    train_ds: xr.Dataset,
    test_ds: Optional[xr.Dataset] = None,
    pred_method: str = "base",
    season_period: float = 365.25,
    engineering_buff: float = 1.5,
    se_buff: float = 1.2,
    printOutput: bool = False,
    q_lo: float = 0.005,
    q_hi: float = 0.995,
    z: float = 2.0,
    min_abs_mm: float = 25.0,
    min_rel: float = 0.05,
    max_swe_mm: float = 3000.0,
    min_train: int = 60,
    snow_present_mm: Optional[float] = None,
    return_long: bool = False,
    seasonal_gate: bool = True,
    seasonal_gate_k_days: int = 3,
    seasonal_gate_eps_mm: float = 0.0,
):
    """
    Extended signature to match insitu_qa.py usage.

    Returns:
      current_vals_df, all_pils_QA, baseline_pils_, df_qa_table, df_simple, df_detail

    If return_long=False, df_simple/df_detail are None.
    """
    obs_data_current_day = [da.isel({"time": t_idx}) for da in obs_data_test]
    missing_stations = [da.name for da in obs_data_current_day if bool(da.isnull().values)]

    wateryear_vals_df, wateryear_lookup_df = create_qa_tables(obs_data_test, missing_stations, isQA=True)
    current_vals_df = pd.DataFrame(wateryear_vals_df.iloc[t_idx]).T
    previous_vals_df = pd.DataFrame(wateryear_vals_df.iloc[t_idx - 1]).T if t_idx > 0 else current_vals_df.copy()
    historic_vals_df = create_qa_tables(obs_data_qa, missing_stations, isQA=False)

    sorted_pillows = previous_vals_df.columns.to_list()
    for df in (current_vals_df, previous_vals_df):
        for c in sorted_pillows:
            if c not in df.columns:
                df[c] = np.nan
        df[:] = df[sorted_pillows].values

    tstamp = obs_data_current_day[0].time.values

    df_yhat, df_sepred, df_sigma, df_ntrain, df_h0 = daily_predictions_pi_snowmodel(
        tstamp=tstamp,
        train_ds=train_ds,
        test_ds=test_ds,
        pil_list=sorted_pillows,
        min_train=min_train,
        pred_method=pred_method,
        season_period=season_period,
        seasonal_gate=seasonal_gate,
        seasonal_gate_k_days=seasonal_gate_k_days,
        seasonal_gate_eps_mm=seasonal_gate_eps_mm,
    )

    df_qa_table = daily_qa_flag_snowmodel_v2(
        historic_vals_df=historic_vals_df,
        current_vals_df=current_vals_df,
        previous_vals_df=previous_vals_df,
        df_sepred=df_sepred,
        df_yhat=df_yhat,
        tstamp=tstamp,
        printOutput=printOutput,
        q_lo=q_lo,
        q_hi=q_hi,
        engineering_buff=engineering_buff,
        z=z,
        se_buff=se_buff,
        min_abs_mm=min_abs_mm,
        min_rel=min_rel,
        max_swe_mm=max_swe_mm,
        snow_present_mm=snow_present_mm,
    )

    df_simple = None
    df_detail = None

    if return_long:
        rows_simple = []
        rows_detail = []
        time_pd = pd.to_datetime(tstamp)
        hist_diff = historic_vals_df.diff()

        for pil in sorted_pillows:
            cur_val = float(current_vals_df[pil].values) if pil in current_vals_df else np.nan
            prev_val = float(previous_vals_df[pil].values) if pil in previous_vals_df else np.nan
            dcur = cur_val - prev_val if (np.isfinite(cur_val) and np.isfinite(prev_val)) else np.nan

            yhat = float(df_yhat[pil].values) if pil in df_yhat else np.nan
            se_pred = float(df_sepred[pil].values) if pil in df_sepred else np.nan

            thr = np.nan
            if np.isfinite(yhat) and np.isfinite(se_pred):
                thr = max(z * se_buff * se_pred, min_abs_mm, min_rel * max(yhat, 1.0))

            missing_today_flag = np.isnan(cur_val)
            phys_flag = (np.isfinite(cur_val) and ((cur_val < 0.0) or (cur_val > max_swe_mm)))

            # NOTE: nan_transition and dswe-envelope checks were previously computed here too.
            # those generic data-integrity checks are now owned by qa_static only — keeping
            # them duplicated in voting/snowmodel made the 2-of-3 majority vote deterministic
            # on shared reasons. snowmodel now only acts on its unique pred-residual signal.

            pred_flag = False
            if (not missing_today_flag) and np.isfinite(thr):
                if (snow_present_mm is not None) and np.isfinite(yhat) and (yhat < float(snow_present_mm)):
                    pred_flag = False
                else:
                    pred_flag = (abs(cur_val - yhat) > thr)

            if missing_today_flag:
                reason_code = "MISSING_TODAY"; reason_detail = "current SWE is NaN"
            elif phys_flag:
                reason_code = "PHYS_LIMIT"; reason_detail = f"cur_mm={cur_val:.1f} outside [0,{max_swe_mm:.0f}]"
            elif pred_flag:
                resid = abs(cur_val - yhat) if (np.isfinite(cur_val) and np.isfinite(yhat)) else np.nan
                reason_code = "PRED_RESID_TOO_LARGE"
                reason_detail = f"|resid|={resid:.1f} > thr={thr:.1f}" if (np.isfinite(resid) and np.isfinite(thr)) else "prediction disagreement"
            else:
                reason_code = "OK"; reason_detail = np.nan

            flag = int(df_qa_table[pil].values[0]) if pil in df_qa_table else int(
                bool(missing_today_flag or phys_flag or pred_flag)
            )

            pred_used = pred_method
            if pred_method.lower() == "seasonal" and seasonal_gate and (test_ds is not None) and (pil in test_ds["pil"].values):
                ds_te = test_ds.sel(pil=pil)
                te_df = pd.DataFrame(
                    {
                        "swed_best": ds_te["swed_best"].to_series(),
                        "swed_second": ds_te["swed_second"].to_series(),
                        "swed_third": ds_te["swed_third"].to_series(),
                    }
                )
                te_df = te_df.loc[:time_pd].tail(max(2, int(seasonal_gate_k_days))).dropna()
                if len(te_df) >= 2:
                    allow = _gate_seasonal_from_predictors(te_df.to_numpy(float), eps_mm=seasonal_gate_eps_mm)
                    pred_used = "seasonal" if allow else "base"

            rows_simple.append(
                {
                    "time": time_pd,
                    "pillow": pil,
                    "method": "snowmodel",
                    "flag": flag,
                    "reason_code": reason_code,
                    "reason_detail": reason_detail,
                }
            )
            rows_detail.append(
                {
                    "time": time_pd,
                    "pillow": pil,
                    "method": "snowmodel",
                    "pred_method": pred_method,
                    "pred_method_used": pred_used,
                    "cur_mm": cur_val,
                    "prev_mm": prev_val,
                    "dswe_mm": dcur,
                    "yhat_mm": yhat,
                    "thr_mm": thr,
                    "se_pred_mm": se_pred,
                    "sigma_mm": float(df_sigma[pil].values) if pil in df_sigma else np.nan,
                    "h0": float(df_h0[pil].values) if pil in df_h0 else np.nan,
                    "missing_today_flag": bool(missing_today_flag),
                    "phys_flag": bool(phys_flag),
                    "pred_flag": bool(pred_flag),
                    "flag": flag,
                    "reason_code": reason_code,
                    "reason_detail": reason_detail,
                }
            )

        df_simple = pd.DataFrame(rows_simple)
        df_detail = pd.DataFrame(rows_detail)

    all_pils_QA = sorted_pillows
    baseline_pils_ = baseline_pils

    if return_long:
        return current_vals_df, all_pils_QA, baseline_pils_, df_simple, df_detail
    return current_vals_df, all_pils_QA, baseline_pils_, df_qa_table
