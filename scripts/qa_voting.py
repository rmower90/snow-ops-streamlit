# qa_voting.py
import copy
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional

# ---------------------------
# Helpers
# ---------------------------

def _robust_sigma_mad(x: np.ndarray) -> float:
    """Robust scale estimate using MAD (returns ~sigma for normal residuals)."""
    x = x[np.isfinite(x)]
    if x.size < 10:
        return np.nan
    med = np.median(x)
    mad = np.median(np.abs(x - med))
    if mad == 0:
        return 0.0
    return 1.4826 * mad

def _safe_corr_r2(a: np.ndarray, b: np.ndarray) -> float:
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 10:
        return np.nan
    r = np.corrcoef(a[m], b[m])[0, 1]
    if not np.isfinite(r):
        return np.nan
    return float(r * r)

def create_qa_tables(obs_data, missing_stations, isQA=True):
    """
    Same as your existing helper (copied in for drop-in convenience).
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


# ---------------------------
# Core algorithm
# ---------------------------

def run_voting_qa(
    t_idx: int,
    obs_data_test: List,
    obs_data_qa: List,
    user_qa_level: int,
    baseline_pils: List[str],
    pil_corr: Dict[str, List[str]],
    engineering_buff: float = 1.5,
    se_buff: float = 1.2,
    printOutput: bool = False,
    max_swe_mm: float = 3000.0,
    min_voters: int = 3,
    ok_frac_thresh: float = 0.60,
    trim_frac: float = 0.20,
    q_lo: float = 0.005,
    q_hi: float = 0.995,
    k_mad: float = 3.0,
    min_abs_mm: float = 25.0,
    min_rel: float = 0.05,
    use_delta: bool = True,
    dynamic_voters: bool = True,
    n_dynamic_voters: int = 5,
    corr_rank: Optional[Dict[str, List[str]]] = None,
    snow_present_mm: Optional[float] = None,
    return_long: bool = False,
) -> Tuple[pd.DataFrame, List[str], List[str], pd.DataFrame, pd.DataFrame, pd.DataFrame]:

    """
    Returns:
      current_vals_df : 1-row df (today) with columns pillow IDs (mm)
      all_pils_QA     : pillows considered OK for inference (QA=0 and numeric today)
      baseline_pils_  : baseline pillows filtered to all_pils_QA
      df_qa_table     : 1-row df with time + per-pillow QA flag {0,1}
      If return_long=True:
        df_simple     : long df (time,pillow,method,flag)
        df_detail     : long df with explainable diagnostics + reason_code
    """
    # Current day slices to identify missing stations
    obs_data_current_day = [da.isel({"time": t_idx}) for da in obs_data_test]
    missing_stations = [da.name for da in obs_data_current_day if da.isnull()]

    # Build daily tables (test)
    wateryear_vals_df, wateryear_lookup_df = create_qa_tables(obs_data_test, missing_stations, isQA=True)
    current_vals_df = pd.DataFrame(wateryear_vals_df.iloc[t_idx]).T
    previous_vals_df = pd.DataFrame(wateryear_vals_df.iloc[t_idx - 1]).T if t_idx > 0 else current_vals_df.copy()

    historic_vals_df = create_qa_tables(obs_data_qa, missing_stations, isQA=False)

    # only build if not provided
    if corr_rank is None:
        corr_rank = build_corr_rank(historic_vals_df, use_delta=use_delta, min_overlap=200, method="r2")

    # safer reorder
    sorted_pillows = previous_vals_df.columns.to_list()
    current_vals_df = current_vals_df.reindex(columns=sorted_pillows)
    previous_vals_df = previous_vals_df.reindex(columns=sorted_pillows)

    # Ensure consistent columns
    sorted_pillows = previous_vals_df.columns.to_list()
    for df in (current_vals_df, previous_vals_df):
        for c in sorted_pillows:
            if c not in df.columns:
                df[c] = np.nan
        # reorder
        df[:] = df[sorted_pillows].values

    # Compute diffs (ΔSWE) in mm/day
    diff_today = (current_vals_df.values - previous_vals_df.values).ravel()
    diff_df = pd.DataFrame([diff_today], columns=sorted_pillows)

    # Time stamp
    tstamp = obs_data_current_day[0].time.values

    # Output QA dict
    qa = {"time": tstamp}
    rows = []  # long-form detail rows for explainability

    # Precompute historic diffs for envelope + voting
    hist_diff = historic_vals_df.diff()

    for pil in sorted_pillows:
        prev_val = float(previous_vals_df[pil].values) if pil in previous_vals_df else np.nan
        cur_val  = float(current_vals_df[pil].values) if pil in current_vals_df else np.nan
        dcur     = float(diff_df[pil].values) if pil in diff_df else np.nan

        # Melt-out gating: when snow is essentially gone, avoid noisy false positives
        melt_gate = False
        if snow_present_mm is not None:
            try:
                if np.isfinite(cur_val) and np.isfinite(prev_val):
                    melt_gate = (cur_val < float(snow_present_mm)) and (prev_val < float(snow_present_mm))
            except Exception:
                melt_gate = False

        dswe_flag_raw = False
        voting_flag_raw = False

        # ---------- HARD CHECK 0: missing today ----------
        missing_today_flag = np.isnan(cur_val)

        # ---------- HARD CHECK 1: physical SWE range ----------
        # (only applies if cur_val is finite)
        phys_flag = (np.isfinite(cur_val) and ((cur_val < 0.0) or (cur_val > max_swe_mm)))

        # ---------- Hard flag A: NaN -> value transition ----------
        nan_transition_flag = (np.isnan(prev_val) and (not np.isnan(cur_val)))

        # ---------- Hard flag B: dswe/dt envelope (robust quantiles) ----------
        dswe_flag = False
        lo = hi = np.nan
        if pil in hist_diff.columns:
            d = hist_diff[pil].to_numpy(dtype=float)
            d = d[np.isfinite(d)]
            if d.size >= 30 and np.isfinite(dcur):
                lo = float(np.quantile(d, q_lo))
                hi = float(np.quantile(d, q_hi))
                dswe_flag_raw = (dcur < lo * engineering_buff) or (dcur > hi * engineering_buff)
                dswe_flag = bool(dswe_flag_raw and (not melt_gate))

        # ---------- Voting: peer consistency on ΔSWE (default) ----------
        # ---------- Voting: choose top-K correlated available voters ----------
        if dynamic_voters:
            # corr_rank[pil] must be an ordered list (best -> worse)
            ranked = (corr_rank.get(pil, []) if corr_rank is not None else pil_corr.get(pil, []))
        
            voters = []
            for v in ranked:
                if v == pil:
                    continue
                if v not in sorted_pillows:
                    continue

                v_today = float(current_vals_df[v].values)
                if np.isnan(v_today):
                    continue

                if use_delta:
                    v_prev = float(previous_vals_df[v].values)
                    if np.isnan(v_prev):
                        continue

                voters.append(v)
                if len(voters) >= n_dynamic_voters:
                    break
        else:
            voters = pil_corr.get(pil, [])
            voters = [v for v in voters if v in sorted_pillows and v != pil]


        vote_records = []  # (abs_resid, ok_bool, weight)

        # IMPORTANT: only attempt voting if we have a usable current value
        if (not missing_today_flag) and (len(voters) > 0):
            y_hist = (hist_diff[pil] if use_delta else historic_vals_df[pil]).astype(float)

            for v in voters:
                v_today = float(current_vals_df[v].values)
                if np.isnan(v_today):
                    continue
                if use_delta:
                    v_prev = float(previous_vals_df[v].values)
                    if np.isnan(v_prev):
                        continue
                    x0 = float(diff_df[v].values)
                else:
                    x0 = v_today

                x_hist = (hist_diff[v] if use_delta else historic_vals_df[v]).astype(float)
                m = np.isfinite(x_hist) & np.isfinite(y_hist)
                if m.sum() < 40:
                    continue

                x = x_hist[m].to_numpy()
                y = y_hist[m].to_numpy()

                x_mean = x.mean()
                y_mean = y.mean()
                denom = np.sum((x - x_mean) ** 2)
                if denom == 0:
                    continue
                b = np.sum((x - x_mean) * (y - y_mean)) / denom
                a = y_mean - b * x_mean

                yhat = a + b * x0
                resid = (dcur - yhat) if use_delta else (cur_val - yhat)

                resid_hist = y - (a + b * x)
                sigma = _robust_sigma_mad(resid_hist)
                if not np.isfinite(sigma):
                    continue

                thr = max(k_mad * sigma * se_buff, min_abs_mm)
                thr = max(thr, min_rel * max(abs(yhat), 1.0))

                ok = (abs(resid) <= thr)

                r2 = _safe_corr_r2(x, y)
                w = r2 if np.isfinite(r2) else 0.5
                vote_records.append((abs(resid), ok, w))

        voting_flag = False
        vote_ok_frac = np.nan
        n_used = 0
        n_dropped = 0

        if len(vote_records) >= min_voters:
            vote_records.sort(key=lambda t: t[0])  # ascending abs_resid
            if trim_frac > 0:
                kdrop = int(np.floor(trim_frac * len(vote_records)))
                kdrop = min(kdrop, max(0, len(vote_records) - min_voters))
            else:
                kdrop = 0

            trimmed = vote_records[: len(vote_records) - kdrop]
            n_used = len(trimmed)
            n_dropped = kdrop

            w_total = sum(w for _, _, w in trimmed)
            w_ok = sum(w for _, ok, w in trimmed if ok)

            if w_total > 0:
                vote_ok_frac = w_ok / w_total
                voting_flag_raw = (vote_ok_frac < ok_frac_thresh)
                voting_flag = bool(voting_flag_raw and (not melt_gate))
            else:
                voting_flag = False

        # ---------- Combine flags into final binary QA ----------
        # Any hard check wins. Voting is only advisory on top of those.
        suspect = bool(
            missing_today_flag
            or phys_flag
            or dswe_flag
            or nan_transition_flag
            or voting_flag
        )
        qa[pil] = 1 if suspect else 0
        # ------------------------------
        # Explainability (detail row)
        # ------------------------------
        reason_code = "OK"
        reason_detail = ""
        if missing_today_flag:
            reason_code = "MISSING_TODAY"
            reason_detail = "current value is NaN"
        elif phys_flag:
            reason_code = "PHYS_RANGE"
            reason_detail = f"cur={cur_val:.1f} mm outside [0,{max_swe_mm:.0f}]" if np.isfinite(cur_val) else "physical range"
        elif nan_transition_flag:
            reason_code = "NAN_TO_VALUE"
            reason_detail = "previous was NaN, current became finite"
        elif melt_gate and (dswe_flag_raw or voting_flag_raw) and (not (missing_today_flag or phys_flag or nan_transition_flag)):
            reason_code = "MELT_GATE_SKIP"
            reason_detail = f"cur/prev < {float(snow_present_mm):.1f} mm; suppressed: " + ("DSWE " if dswe_flag_raw else "") + ("VOTING" if voting_flag_raw else "")
        elif dswe_flag:
            reason_code = "DSWE_ENVELOPE"
            if np.isfinite(lo) and np.isfinite(hi) and np.isfinite(dcur):
                reason_detail = f"dSWE={dcur:.1f} outside [{lo:.1f},{hi:.1f}]*buff({engineering_buff:.2f})"
            else:
                reason_detail = "dSWE outside envelope"
        elif voting_flag:
            reason_code = "VOTING_INCONSISTENT"
            if np.isfinite(vote_ok_frac):
                reason_detail = f"ok_frac={vote_ok_frac:.2f} < {ok_frac_thresh:.2f} (used={n_used}, dropped={n_dropped})"
            else:
                reason_detail = f"insufficient voters (<{min_voters}) or unusable history"

        rows.append({
            "time": pd.to_datetime(tstamp),
            "pillow": str(pil).strip(),
            "method": "voting",
            "QUALITYSCORE": int(qa[pil]),
            "severity": int(qa[pil]),
            "flag": int(qa[pil]),
            "reason_code": reason_code,
            "reason_detail": reason_detail,
            "cur_mm": cur_val,
            "prev_mm": prev_val,
            "diff_mm": dcur,
            "dq_lo": lo,
            "dq_hi": hi,
            "missing_today_flag": int(missing_today_flag),
            "phys_flag": int(phys_flag),
            "nan_transition_flag": int(nan_transition_flag),
            "dswe_flag": int(dswe_flag),
            "melt_gate": int(melt_gate),
            "pred_flag": 0,
            "pred_check_skipped": 1 if (missing_today_flag or not np.isfinite(dcur)) else 0,
            "pred_skip_reason": "not_applicable",
            "vote_ok_frac": vote_ok_frac,
            "ok_frac_thresh": ok_frac_thresh,
            "voters_used": int(n_used),
            "voters_dropped": int(n_dropped),
            "n_vote_records": int(len(vote_records)),
            "dynamic_voters": int(dynamic_voters),
            "n_dynamic_voters": int(n_dynamic_voters),
            "use_delta": int(use_delta),
            "voters_list": ",".join([str(v) for v in voters]) if isinstance(voters, list) else "",
            "vote_check_skipped": int(melt_gate),
            "vote_skip_reason": f"melt_gate<{float(snow_present_mm):.1f}mm" if melt_gate and (snow_present_mm is not None) else "",
        })

        if printOutput:
            print(f"{pil}: QA={qa[pil]}")
            print(f"  cur={cur_val if np.isfinite(cur_val) else np.nan:.1f} mm prev={prev_val if np.isfinite(prev_val) else np.nan:.1f} mm dSWE={dcur if np.isfinite(dcur) else np.nan:.1f} mm/day")
            print(f"  hard_checks: missing_today_flag={missing_today_flag} phys_flag={phys_flag} (max_swe_mm={max_swe_mm:.0f})")
            if np.isfinite(lo) and np.isfinite(hi):
                print(f"  envelope[{q_lo:.3f},{q_hi:.3f}]={lo:.1f},{hi:.1f} (buff={engineering_buff:.2f}) dswe_flag={dswe_flag}")
            else:
                print(f"  envelope: n/a dswe_flag={dswe_flag}")
            print(f"  nan_transition_flag={nan_transition_flag}")
            if np.isfinite(vote_ok_frac):
                print(f"  voting: ok_frac={vote_ok_frac:.2f} (thresh={ok_frac_thresh:.2f}) voters_used={n_used} dropped={n_dropped} voting_flag={voting_flag}")
            else:
                print(f"  voting: n/a (need >= {min_voters} voters with usable history)")
            print("")

    df_qa_table = pd.DataFrame([qa])
    df_qa_table["time"] = pd.to_datetime(df_qa_table["time"])

    # Determine which pillows are OK for inference (numeric today + QA==0)
    pillows_w_numeric_vals = current_vals_df.dropna(axis=1).columns.to_list()
    all_pils_QA = [p for p in sorted_pillows if (p in pillows_w_numeric_vals) and (df_qa_table.at[0, p] == 0)]

    baseline_pils_ = [p for p in baseline_pils if p in all_pils_QA]

    if not return_long:
        return current_vals_df, all_pils_QA, baseline_pils_, df_qa_table

    df_detail = pd.DataFrame(rows)
    if not df_detail.empty:
        df_detail["time"] = pd.to_datetime(df_detail["time"])
        df_detail["pillow"] = df_detail["pillow"].astype(str).str.strip()

    # Simple comparison table
    df_simple = df_detail[["time", "pillow", "method", "flag"]].copy() if not df_detail.empty else pd.DataFrame(columns=["time","pillow","method","flag"])

    return current_vals_df, all_pils_QA, baseline_pils_, df_simple, df_detail


def build_corr_rank(
    historic_vals_df: pd.DataFrame,
    use_delta: bool = True,
    min_overlap: int = 200,
    method: str = "r2",   # "r" or "r2"
) -> Dict[str, List[str]]:
    """
    Build ordered correlation rankings from historic QA'd data.

    Returns:
      corr_rank[pil] = [v1, v2, v3, ...]  # best -> worse
    """
    df = historic_vals_df.copy()
    if use_delta:
        df = df.diff()

    cols = list(df.columns)
    corr_rank: Dict[str, List[str]] = {}

    # Preconvert to arrays for speed
    arr = df.to_numpy(float)
    col_idx = {c: i for i, c in enumerate(cols)}

    for pil in cols:
        i = col_idx[pil]
        y = arr[:, i]

        scores = []
        for v in cols:
            if v == pil:
                continue
            j = col_idx[v]
            x = arr[:, j]

            m = np.isfinite(x) & np.isfinite(y)
            if m.sum() < min_overlap:
                continue

            r = np.corrcoef(x[m], y[m])[0, 1]
            if not np.isfinite(r):
                continue

            score = (r * r) if method == "r2" else abs(r)
            scores.append((score, v))

        scores.sort(reverse=True)  # best first
        corr_rank[pil] = [v for _, v in scores]

    return corr_rank

