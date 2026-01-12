"""qa_static.py (peer-median static QA)

Drop-in replacement for your qa_static module.

Goal: make "static" flags stable and explainable.

Key behavior
------------
- Uses **absolute SWE** (mm) and **peer median** from correlated pillows (no regression).
- Keeps your envelope check on **dSWE/day** (mm/day) from historic quantiles.
- Optional melt gating via snow_present_mm to avoid near-zero noise flags.
- Returns the same first 4 outputs as your original run_old_qa.
- If return_long=True, returns (df_simple, df_detail) as 5th and 6th outputs.

This file intentionally does NOT depend on rioxarray.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import copy
import numpy as np
import pandas as pd


# -----------------------------
# Small utilities
# -----------------------------

def _robust_sigma_mad(x: np.ndarray) -> float:
    """Robust sigma estimate from MAD (scaled ~sigma for Normal)."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size < 5:
        return np.nan
    med = np.median(x)
    mad = np.median(np.abs(x - med))
    if not np.isfinite(mad) or mad == 0:
        return np.nan
    return float(1.4826 * mad)


def _safe_quantile_bounds(x: np.ndarray, q_lo: float, q_hi: float) -> Tuple[float, float]:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size < 10:
        return (np.nan, np.nan)
    return (float(np.quantile(x, q_lo)), float(np.quantile(x, q_hi)))


def create_qa_tables(
    obs_data: List,
    missing_stations: List[str],
    isQA: bool = True,
) -> pd.DataFrame:
    """Minimal helper used throughout your QA stack.

    Returns a (time x pillow) DataFrame of SWE values (mm).

    Notes
    -----
    - obs_data is list[xr.DataArray] with name == pillow id.
    - We *do not* try to carry lookup tables here (your old code did).
    """
    # Build a dict of series
    cols = {}
    time_index = None
    for da in obs_data:
        pil = str(da.name).strip()
        # drop "missing stations" entirely? (keep column but will be NaN)
        s = da.to_series()
        if time_index is None:
            time_index = s.index
        cols[pil] = s

    df = pd.DataFrame(cols)
    df.index.name = "time"
    df = df.sort_index()

    # if a station is missing today, keep it as column but values may be NaN
    for pil in missing_stations:
        if pil not in df.columns:
            df[pil] = np.nan

    return df


# -----------------------------
# Main entrypoint
# -----------------------------

def run_old_qa(
    t_idx: int,
    obs_data_test: List,
    obs_data_qa: List,
    user_qa_level: int,
    baseline_pils: List,
    pil_corr: Dict[str, List[str]],
    engineering_buff_: float = 1.5,
    se_buff_: float = 1.2,
    printOutput: bool = False,
    isv1: bool = False,  # kept for signature compatibility; ignored here
    return_long: bool = False,
    # envelope knobs
    q_lo: float = 0.005,
    q_hi: float = 0.995,
    # peer check knobs
    top_k: int = 5,
    k_mad: float = 3.0,
    min_abs_mm: float = 25.0,
    min_rel: float = 0.05,
    # hard checks
    max_swe_mm: float = 3000.0,
    # melt gating
    snow_present_mm: Optional[float] = None,
) -> Tuple[pd.DataFrame, List[str], List[str], pd.DataFrame, Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """Static QA.

    Returns
    -------
    current_vals_df : 1-row df (today) with pillow columns (mm)
    all_pils_QA     : pillows OK for inference (numeric today and severity<=user_qa_level)
    baseline_pils_  : baseline pillows filtered to all_pils_QA
    df_bad_data     : 1-row df with per-pillow severity {0,1,2}

    If return_long=True additionally returns:
      df_simple : columns [time, pillow, method, flag]
      df_detail : explainable diagnostics per pillow
    """

    # current day slices to identify missing stations
    obs_data_current_day = [da.isel({"time": t_idx}) for da in obs_data_test]
    missing_stations = [str(da.name).strip() for da in obs_data_current_day if da.isnull()]

    wateryear_vals_df = create_qa_tables(obs_data_test, missing_stations, isQA=True)
    current_vals_df = pd.DataFrame(wateryear_vals_df.iloc[t_idx]).T
    previous_vals_df = pd.DataFrame(wateryear_vals_df.iloc[t_idx - 1]).T if t_idx > 0 else current_vals_df.copy()

    historic_vals_df = create_qa_tables(obs_data_qa, missing_stations, isQA=False)

    # consistent columns
    sorted_pillows = previous_vals_df.columns.to_list()
    for df in (current_vals_df, previous_vals_df, historic_vals_df):
        for c in sorted_pillows:
            if c not in df.columns:
                df[c] = np.nan
        df[:] = df[sorted_pillows].values

    # dSWE for today
    diff_today = (current_vals_df.values - previous_vals_df.values).ravel()
    diff_df = pd.DataFrame([diff_today], columns=sorted_pillows)

    # historic diffs for envelope
    hist_diff = historic_vals_df.diff()

    # timestamp
    tstamp = pd.to_datetime(obs_data_current_day[0].time.values)

    # output containers
    qa_sev = {"time": tstamp}
    rows_detail = []

    for pil in sorted_pillows:
        prev_val = float(previous_vals_df[pil].values) if pil in previous_vals_df else np.nan
        cur_val = float(current_vals_df[pil].values) if pil in current_vals_df else np.nan
        diff_val = float(diff_df[pil].values) if pil in diff_df else np.nan

        # --- hard checks ---
        missing_today_flag = bool(np.isnan(cur_val))
        phys_flag = bool(np.isfinite(cur_val) and ((cur_val < 0.0) or (cur_val > max_swe_mm)))
        nan_transition_flag = bool(np.isnan(prev_val) and (not np.isnan(cur_val)))

        # --- dswe envelope ---
        dswe_flag = False
        dq_lo = dq_hi = np.nan
        if pil in hist_diff.columns and np.isfinite(diff_val):
            lo, hi = _safe_quantile_bounds(hist_diff[pil].to_numpy(dtype=float), q_lo=q_lo, q_hi=q_hi)
            dq_lo, dq_hi = lo, hi
            if np.isfinite(lo) and np.isfinite(hi):
                dswe_flag = bool((diff_val < lo * engineering_buff_) or (diff_val > hi * engineering_buff_))

        # --- peer-median absolute SWE check ---
        peer_used: List[str] = []
        peer_med = np.nan
        resid = np.nan
        sigma = np.nan
        thr = np.nan
        pred_flag = False
        pred_check_skipped = False
        pred_skip_reason = ""

        # melt gating: skip peer check if there's basically no snow
        if snow_present_mm is not None:
            if (np.isfinite(cur_val) and cur_val < snow_present_mm):
                pred_check_skipped = True
                pred_skip_reason = f"cur<{snow_present_mm:g}"

        if (not pred_check_skipped) and (not missing_today_flag) and (not phys_flag):
            peers = [p for p in pil_corr.get(pil, []) if p in sorted_pillows and p != pil]
            peers = peers[: max(1, int(top_k))]

            # require peers with finite values today
            peer_vals_today = []
            for p in peers:
                v = float(current_vals_df[p].values)
                if np.isfinite(v):
                    peer_vals_today.append(v)
                    peer_used.append(p)

            if len(peer_vals_today) == 0:
                pred_check_skipped = True
                pred_skip_reason = "no_peers_today"
            else:
                peer_med = float(np.median(peer_vals_today))

                # if melt gating: also require peer median above threshold
                if snow_present_mm is not None and peer_med < snow_present_mm:
                    pred_check_skipped = True
                    pred_skip_reason = f"peer_med<{snow_present_mm:g}"
                else:
                    resid = cur_val - peer_med

                    # build historic residual series using same peers
                    # (median across peers each day, then resid vs target)
                    hist_peer = historic_vals_df[peer_used].median(axis=1, skipna=True)
                    hist_resid = historic_vals_df[pil] - hist_peer
                    sigma = _robust_sigma_mad(hist_resid.to_numpy(dtype=float))

                    # If sigma not finite, skip rather than flag
                    if not np.isfinite(sigma):
                        pred_check_skipped = True
                        pred_skip_reason = "sigma_nan"
                    else:
                        thr = max(k_mad * sigma * se_buff_, min_abs_mm)
                        thr = max(thr, min_rel * max(abs(peer_med), 1.0))
                        pred_flag = bool(abs(resid) > thr)

        # --- combine to severity ---
        reason_code = "OK"
        reason_detail = ""

        if missing_today_flag:
            sev = 2
            reason_code = "MISSING_TODAY"
        elif phys_flag:
            sev = 2
            reason_code = "PHYS_RANGE"
        elif nan_transition_flag:
            # keep this as severity 1 (soft) so user_qa_level can choose
            sev = 1
            reason_code = "NAN_TO_VALUE"
        elif dswe_flag:
            sev = 2
            reason_code = "DSWE_ENVELOPE"
            reason_detail = f"diff={diff_val:.1f} outside [{dq_lo:.1f},{dq_hi:.1f}] * buff"
        elif pred_flag:
            sev = 1
            reason_code = "PEER_RESID_TOO_LARGE"
            reason_detail = f"|resid|={abs(resid):.1f} > thr={thr:.1f} (sigma={sigma:.1f})"
        elif pred_check_skipped:
            sev = 0
            reason_code = "PEER_CHECK_SKIPPED"
            reason_detail = pred_skip_reason
        else:
            sev = 0

        qa_sev[pil] = int(sev)

        if printOutput:
            print(f"{pil}: severity={sev} reason={reason_code} {reason_detail}")

        # long detail row
        rows_detail.append(
            {
                "time": tstamp,
                "pillow": str(pil).strip(),
                "method": "static",
                "QUALITYSCORE": int(sev),
                "severity": int(sev),
                "flag": 1 if sev > 0 else 0,
                "reason_code": reason_code,
                "reason_detail": reason_detail,
                "cur_mm": cur_val,
                "prev_mm": prev_val,
                "diff_mm": diff_val,
                "peer_median_mm": peer_med,
                "resid_mm": resid,
                "thr_mm": thr,
                "sigma_mm": sigma,
                "n_peers_used": len(peer_used),
                "peers_used": ",".join(peer_used) if len(peer_used) else "",
                "missing_today_flag": int(missing_today_flag),
                "phys_flag": int(phys_flag),
                "nan_transition_flag": int(nan_transition_flag),
                "dswe_flag": int(dswe_flag),
                "pred_flag": int(pred_flag),
                "pred_check_skipped": int(pred_check_skipped),
                "pred_skip_reason": pred_skip_reason,
                "dq_lo": dq_lo,
                "dq_hi": dq_hi,
                "snow_present_mm": snow_present_mm if snow_present_mm is not None else np.nan,
            }
        )

    df_bad_data = pd.DataFrame([qa_sev])
    df_bad_data["time"] = pd.to_datetime(df_bad_data["time"])

    # QA selection lists
    pillows_w_numeric_vals = current_vals_df.dropna(axis=1).columns.to_list()
    pils_qa_level = [p for p in sorted_pillows if (p in pillows_w_numeric_vals) and (df_bad_data.at[0, p] <= user_qa_level)]
    all_pils_QA = pils_qa_level
    baseline_pils_ = [p for p in baseline_pils if p in all_pils_QA]

    if not return_long:
        # Legacy behavior: 4-tuple with the wide per-day QA table.
        return current_vals_df, all_pils_QA, baseline_pils_, df_bad_data

    # New behavior for your analysis/plots: return long outputs only
    # (keeps notebook/script calls consistent: expect 5 values).
    df_detail = pd.DataFrame(rows_detail)
    df_simple = df_detail[["time", "pillow", "method", "flag"]].copy()
    return current_vals_df, all_pils_QA, baseline_pils_, df_simple, df_detail

