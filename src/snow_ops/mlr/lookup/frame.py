"""
Canonical predict-NaNs historical training matrix for the pretrained MLR lookup tables.

WHAT THIS PRODUCES

    X   (n_flights, n_pillows)   complete pillow SWE, mm -- observed where available,
                                 imputed elsewhere by the EXISTING three-donor method
    y   (n_flights,)             ASO basin-mean SWE, mm, for one elevation band
    plus a per-cell boolean mask recording which entries were imputed.

Every retained pillow has a value on every retained flight, so every pillow combination
is fit against identical dates and an identical y. That is the whole point: adjusted R2
compared across combinations is only meaningful if the support is the same.

WHY IT CANNOT REUSE THE PRODUCTION PATHWAY AS-IS

`lm_model.impute_pillow_prediction` is called from `run_mlr_train_predict` with
`all_pils_QA` -- the set of pillows that passed QA TODAY -- as its `all_pils` argument.
That single argument makes four things day-dependent:

  1. the retained universe            pils_removed is a subset of all_pils_QA
  2. the imputed values themselves    the donor search draws candidates from
                                      df_dropped_pils.columns, i.e. the retained set
  3. the surviving flight count       run_cross_val_selection2's missing_times filter
                                      runs over all_pillows = pils_removed
  4. the on-disk cache                keyed only on (site, WY, suffix); its COLUMNS are
                                      frozen at whatever availability the first day of
                                      the run happened to have

The imputation METHOD is not the problem and is not reimplemented here. The fix is to
hand the same function a fixed historical universe and a caller-owned output path. Both
are one argument each.

CONVENTIONS (V4-V6 fidelity, confirmed by A/B reconstruction against the published
FRIANT v4/v5/v6 outputs)

    fit_intercept = False, add_zero_aso = True.

Under fit_intercept=False a row of zeros contributes nothing to X'X or X'y, so the
synthetic row is EXACTLY coefficient- and SSE-inert (measured: max|dcoef| = 0.0e+00).
It is not statistic-inert: it moves n and ybar, inflating R2 by ~0.06. So the row is
added at fit time and excluded from every reported statistic --
`stats_exclude_synthetic_row = True`. It is deliberately kept OUT of the imputation
procedure, which sees only real observations.

The synthetic row is NOT added here. This module returns the real historical matrix;
`fit.py` augments per combination.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
import xarray as xr

# ---------------------------------------------------------------------------
# The batch pipeline lives in scripts/ and is imported by the drivers through the
# /home/rossamower/bin/scripts symlink, which resolves back into this same checkout.
# Reuse those modules rather than reimplementing them -- see aso_io.py for the same
# pattern applied to the config loader.
# ---------------------------------------------------------------------------
_SCRIPTS_DIR = Path(__file__).resolve().parents[4] / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(1, str(_SCRIPTS_DIR))

import lm_model  # noqa: E402
import metadata  # noqa: E402
import preprocessing  # noqa: E402

# ---------------------------------------------------------------------------
# Schema / methodology constants. These participate in the input hash, so bumping one
# forces a rebuild rather than silently reusing an incompatible artifact.
# ---------------------------------------------------------------------------
FRAME_SCHEMA_VERSION = "canonical_frame_v1"
IMPUTE_METHOD = "lm_model.impute_pillow_prediction:three_donor_adjr2_forward"
IMPUTE_DONOR_WINDOW_START_YEAR = 2013  # hardcoded inside the imputation search
MAX_PILLOWS = 5
DEFAULT_OBS_THRESHOLD = 0.50
DEFAULT_SCORE_RULE = "adj_r2"
TIE_BREAK = ("score_desc", "n_predictors_asc", "pillow_key_asc")
SUPPORT_POLICY = "canonical_complete_imputed"

_ASO_COL = "aso_mean_bins_mm"
_IMPUTED_SUFFIX = "_is_imputed"


# ===========================================================================
# provenance helpers
# ===========================================================================

def _sha256_16(path: str | Path) -> str | None:
    """Short content digest. None if the file is unreadable rather than raising --
    provenance must never be the thing that breaks a build."""
    try:
        h = hashlib.sha256()
        with open(path, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()[:16]
    except OSError:
        return None


def _file_provenance(path: str | Path) -> dict:
    p = Path(path)
    resolved = p.resolve() if p.exists() else p
    out = {
        "path": str(p),
        "resolved_path": str(resolved),
        "is_symlink": p.is_symlink(),
        "sha256_16": _sha256_16(p),
    }
    try:
        st = p.stat()
        out["size_bytes"] = st.st_size
        out["mtime_utc"] = datetime.fromtimestamp(st.st_mtime, timezone.utc).isoformat()
    except OSError:
        out["size_bytes"] = None
        out["mtime_utc"] = None
    return out


def _git_provenance(repo_root: Path) -> dict:
    """Branch, commit and dirtiness. `dirty` and `dirty_files` are recorded because the
    V4-V6 archaeology turned on exactly this: three published runs were produced from an
    uncommitted working tree and nothing on disk said so."""
    def _run(*args: str, strip: bool = True) -> str | None:
        try:
            out = subprocess.run(args, cwd=repo_root, capture_output=True,
                                 text=True, timeout=15, check=True).stdout
            return out.strip() if strip else out
        except (subprocess.SubprocessError, OSError):
            return None

    # NOT stripped: `git status --porcelain` encodes the staged/unstaged state in the
    # first TWO columns, so an unstaged modification is " M path" with a leading space.
    # Stripping the whole output eats that space on the first line only, after which a
    # fixed l[3:] slice silently truncates one character off that one path -- observed as
    # "cripts/lm_model.py". Slice past the 2-column code and strip per line instead.
    status = _run("git", "status", "--porcelain", strip=False)
    dirty_files = sorted(l[2:].strip() for l in (status or "").splitlines() if l.strip())
    return {
        "repo_root": str(repo_root),
        "branch": _run("git", "rev-parse", "--abbrev-ref", "HEAD"),
        "commit": _run("git", "rev-parse", "HEAD"),
        "commit_short": _run("git", "rev-parse", "--short", "HEAD"),
        "commit_date": _run("git", "log", "-1", "--format=%cI"),
        "dirty": bool(dirty_files),
        "dirty_files": dirty_files,
    }


def _water_year(ts: pd.Timestamp) -> int:
    return int(ts.year) + 1 if int(ts.month) >= 10 else int(ts.year)


# ===========================================================================
# the frame
# ===========================================================================

@dataclass(frozen=True)
class CanonicalFrame:
    """Immutable canonical training matrix plus everything needed to reproduce it."""

    y: np.ndarray                       # (n,)   ASO basin-mean SWE, mm
    X: np.ndarray                       # (n, p) complete pillow SWE, mm
    imputed: np.ndarray                 # (n, p) bool, True where X was imputed
    X_raw: np.ndarray                   # (n, p) pre-imputation, NaN where imputed
    universe: tuple[str, ...]           # retained pillows, sorted -- BITMASK BIT ORDER
    dates: pd.DatetimeIndex             # (n,)   ASO flight dates
    water_years: np.ndarray             # (n,)   int
    table: pd.DataFrame                 # tidy on-disk form
    manifest: dict = field(default_factory=dict)

    @property
    def n_obs(self) -> int:
        return int(self.X.shape[0])

    @property
    def n_pillows(self) -> int:
        return int(self.X.shape[1])

    def frac_imputed_per_pillow(self) -> pd.Series:
        return pd.Series(self.imputed.mean(axis=0), index=list(self.universe))

    def imputation_burden(self, pillows: Sequence[str]) -> dict:
        """Model-level imputation burden for one combination. B1: this is what lets a
        model be filtered on imputation burden LATER without rebuilding the library --
        it is a function of the frame, not of the fit, so fit.py can materialise it as
        stored columns for every combination."""
        idx = [self.universe.index(p) for p in pillows]
        per = self.imputed[:, idx].mean(axis=0)
        cells = int(self.imputed[:, idx].sum())
        return {
            "n_imputed_cells": cells,
            "frac_imputed": float(cells / max(self.imputed[:, idx].size, 1)),
            "max_pillow_frac_imputed": float(per.max()) if len(per) else 0.0,
            "mean_pillow_frac_imputed": float(per.mean()) if len(per) else 0.0,
        }


# ===========================================================================
# paths
# ===========================================================================

def library_root(mlr_pred_dir: str | Path) -> Path:
    """`mlrPred_dir` from the basin YAML is already basin-scoped
    (.../mlr_prediction/FRIANT/), so the basin level is not repeated here."""
    return Path(mlr_pred_dir) / "model_library"


def frame_paths(mlr_pred_dir: str | Path, frame_id: str) -> dict[str, Path]:
    root = library_root(mlr_pred_dir)
    return {
        "root": root,
        "frame": root / "canonical_frames" / f"{frame_id}.parquet",
        "frame_csv": root / "canonical_frames" / f"{frame_id}.csv",
        "impute_cache": root / "canonical_frames" / "impute_cache" / f"{frame_id}_impute.csv",
        "manifest": root / "manifests" / f"{frame_id}.json",
        "libraries": root / "libraries",
    }


def _frame_id(basin: str, mode: str, elev_band: str, library_id: str,
              qa_tag: str, inputs_hash: str) -> str:
    safe_band = elev_band.replace("<", "lt").replace(">", "gt").replace("/", "_")
    return f"{basin}_{mode}_{safe_band}_{library_id}_{qa_tag}_{inputs_hash}"


# ===========================================================================
# build
# ===========================================================================

def build_canonical_frame(
    basin: str,
    *,
    elev_band: str = "total",
    mode: str = "season",
    obs_qa_file: str = "pillow_wy_1980_2025_qa6.nc",
    obs_threshold: float = DEFAULT_OBS_THRESHOLD,
    excluded_years: Iterable[int] = (),
    library_id: str = "all_years",
    fit_intercept: bool = False,
    add_zero_aso: bool = True,
    config_dir: str | Path = "/home/rossamower/work/aso/configs/",
    rebuild_imputation: bool = False,
    verbose: bool = True,
) -> CanonicalFrame:
    """
    Build the canonical predict-NaNs training matrix for one
    (basin, elevation band, mode, library) and return it with a full manifest.

    Nothing about a prediction date or a daily QA outcome enters this function.

    `excluded_years` is water years to omit from the TARGET rows; it is the mechanism
    for the later leave-one-water-year sensitivity libraries. Empty for the primary
    full-history library (D5).
    """
    excluded_years = tuple(sorted(int(y) for y in excluded_years))
    if mode != "season":
        raise NotImplementedError(
            f"mode={mode!r}: accumulation/melt need process_melt_accum_thresh, "
            "which is milestone 2. Only 'season' is supported."
        )

    log = (lambda *a: print(*a)) if verbose else (lambda *a: None)
    repo_root = Path(__file__).resolve().parents[4]

    # ---- config -----------------------------------------------------------
    cfg_path = Path(config_dir) / "regions" / f"{basin}.yaml"
    cfg = metadata.load_yaml(cfg_path)
    fp = cfg["data_filepaths"]
    exclude_pillows = sorted((cfg.get("pillow_api", {}) or {}).get("exclude_pillows") or [])
    obs_qa_path = Path(fp["insitu_dir"]) / "processed" / obs_qa_file
    qa_tag = Path(obs_qa_file).stem.split("_")[-1]

    log(f"CANONICAL FRAME  basin={basin} band={elev_band} mode={mode} "
        f"library={library_id} qa={qa_tag}")

    # ---- load -------------------------------------------------------------
    aso_spatial_ds = xr.open_dataset(fp["aso_spatial"], engine="netcdf4")
    aso_tseries_ds = xr.open_dataset(fp["aso_temporal"], engine="netcdf4")
    obs_ds = xr.load_dataset(str(obs_qa_path))
    if exclude_pillows:
        obs_ds = obs_ds.drop_vars(exclude_pillows, errors="ignore")

    elev_labels = [str(v) for v in aso_tseries_ds["elev"].values]
    if elev_band not in elev_labels:
        raise ValueError(f"elev_band={elev_band!r} not in ASO elev labels {elev_labels}")
    elev_idx = elev_labels.index(elev_band)
    # 'total' is the last ASO elev entry; both drivers address it as -1, so normalise to
    # the negative index they use to keep combine_aso_insitu's slice identical.
    if elev_idx == len(elev_labels) - 1:
        elev_idx = -1

    obs_lst = [obs_ds[p].rename(p) for p in obs_ds.data_vars]
    candidate_pillows = tuple(sorted(str(p) for p in obs_ds.data_vars))

    # ---- no-holdout split -------------------------------------------------
    # D5: the primary library uses every available flight. train_test_split is reused
    # rather than bypassed because it also produces the >=2013 `insitu_qa` window that
    # the imputation donor regressions train on. Handing it a test water year one past
    # the end of the record makes the test side empty, so aso_tseries_train is the full
    # flight list. The sentinel is DERIVED from the data, not a magic constant.
    flight_dates_all = pd.DatetimeIndex(pd.to_datetime(aso_tseries_ds["date"].values))
    holdout_sentinel_wy = max(_water_year(d) for d in flight_dates_all) + 1

    (_, aso_tseries_test, _, aso_tseries_train,
     obs_data_train, obs_data_qa) = preprocessing.train_test_split(
        aso_spatial_ds, aso_tseries_ds, obs_lst, holdout_sentinel_wy,
        isFriant=True, aso_holdout_mode="wy")

    n_test = 0 if aso_tseries_test is None else int(aso_tseries_test.date.size)
    if n_test:
        raise RuntimeError(
            f"holdout sentinel WY{holdout_sentinel_wy} still held out {n_test} flight(s); "
            "the primary library must use the full record")

    # ---- raw flight frame -------------------------------------------------
    df_sum_total = preprocessing.combine_aso_insitu(
        obs_data_train, aso_tseries_train["aso_swe"], elev_idx=elev_idx)

    if excluded_years:
        wy = pd.to_datetime(df_sum_total["time"]).map(_water_year)
        df_sum_total = df_sum_total[~wy.isin(excluded_years)].reset_index(drop=True)

    # The imputation loop mixes positional .iloc[row] with label .loc[row, pil], so a
    # 0..n-1 RangeIndex is a correctness precondition, not a formality.
    df_sum_total = df_sum_total.reset_index(drop=True)
    assert list(df_sum_total.index) == list(range(len(df_sum_total))), \
        "df_sum_total must carry a 0..n-1 RangeIndex for impute_pillow_prediction"

    all_pils = [c for c in df_sum_total.columns if c not in ("time", _ASO_COL)]
    n_flights = len(df_sum_total)
    log(f"  df_sum_total {df_sum_total.shape}  n_flights={n_flights} "
        f"n_candidate_pillows={len(all_pils)}")

    # ---- retention (recomputed here only to assert against the function) --
    frac_missing_all = (df_sum_total[all_pils].isnull().sum(axis=0) / n_flights)
    expected_universe = tuple(sorted(frac_missing_all[frac_missing_all < obs_threshold].index))
    dropped = tuple(sorted(set(all_pils) - set(expected_universe)))
    log(f"  retention frac_missing < {obs_threshold}: "
        f"retained={len(expected_universe)} dropped={len(dropped)}")

    # ---- input hash / paths ----------------------------------------------
    hash_payload = {
        "frame_schema_version": FRAME_SCHEMA_VERSION,
        "impute_method": IMPUTE_METHOD,
        "impute_donor_window_start_year": IMPUTE_DONOR_WINDOW_START_YEAR,
        "basin": basin, "mode": mode, "elev_band": elev_band, "elev_idx": elev_idx,
        "library_id": library_id, "excluded_years": excluded_years,
        "obs_threshold": obs_threshold,
        "candidate_pillows": candidate_pillows,
        "exclude_pillows": exclude_pillows,
        "flight_dates": [str(d.date()) for d in pd.to_datetime(df_sum_total["time"])],
        "aso_temporal_sha256_16": _sha256_16(fp["aso_temporal"]),
        "obs_qa_sha256_16": _sha256_16(obs_qa_path),
    }
    inputs_hash = hashlib.sha256(
        json.dumps(hash_payload, sort_keys=True, default=str).encode()
    ).hexdigest()[:8]
    frame_id = _frame_id(basin, mode, elev_band, library_id, qa_tag, inputs_hash)
    paths = frame_paths(fp["mlrPred_dir"], frame_id)
    for key in ("frame", "impute_cache", "manifest"):
        paths[key].parent.mkdir(parents=True, exist_ok=True)
    paths["libraries"].mkdir(parents=True, exist_ok=True)

    # A changed input changes `inputs_hash`, hence the filename, hence a cache miss.
    # This is the structural fix for the stale-cache class of defect the CV audit found
    # (v2-v6 silently reused a qa1-derived imputation table under qa6 data).
    if rebuild_imputation and paths["impute_cache"].exists():
        log(f"  --rebuild-imputation: removing {paths['impute_cache'].name}")
        paths["impute_cache"].unlink()
    cache_hit = paths["impute_cache"].exists()
    log(f"  impute cache {'HIT ' if cache_hit else 'MISS'} {paths['impute_cache']}")

    # ---- imputation (existing method, fixed universe, owned path) ---------
    # all_pils is the FIXED historical candidate list, not today's all_pils_QA. That one
    # substitution is what makes the result availability-independent.
    #
    # prediction_date is inert once impute_fpath is supplied -- it is used nowhere else
    # in the function -- but it is a positional parameter, so the last real flight date
    # is passed rather than inventing a water year.
    last_flight = pd.to_datetime(df_sum_total["time"]).max()
    sentinel_date = datetime(last_flight.year, last_flight.month, last_flight.day)

    obs_imputed_lst, retained, df_imputed = lm_model.impute_pillow_prediction(
        df_sum_total,
        all_pils,
        obs_data_qa,
        basin,
        sentinel_date,
        obs_threshold=obs_threshold,
        saveImputeCSV=True,
        impute_fpath=str(paths["impute_cache"]),
    )
    universe = tuple(sorted(str(p) for p in retained))

    if universe != expected_universe:
        raise RuntimeError(
            "retained universe disagrees with the independently recomputed rule:\n"
            f"  function : {universe}\n  recomputed: {expected_universe}")

    # ---- assemble ---------------------------------------------------------
    df_imputed = df_imputed.copy()
    df_imputed["time"] = pd.to_datetime(df_imputed["time"])
    df_imputed = df_imputed.sort_values("time").reset_index(drop=True)

    dates = pd.DatetimeIndex(df_imputed["time"])
    raw = (df_sum_total.assign(time=pd.to_datetime(df_sum_total["time"]))
           .set_index("time").reindex(dates))
    X_raw = raw[list(universe)].to_numpy(dtype=float)
    imputed_mask = np.isnan(X_raw)

    X = df_imputed[list(universe)].to_numpy(dtype=float)

    # Imputation must FILL NaNs, not perturb observations. lm_model writes its table to
    # CSV and reads it straight back (lm_model.py:1455-1462), and that round-trip moves
    # already-observed values by up to ~2.3e-13 mm -- one ULP at 1000 mm. Harmless
    # numerically, but it would make the canonical matrix a function of pandas' CSV float
    # formatting and would break any bit-exactness claim about the observed data. So
    # observed cells are restored from the raw frame, confining CSV precision to the
    # imputed cells, which are regression predictions where one ULP is meaningless.
    _obs = ~imputed_mask
    n_restored = int((X[_obs] != X_raw[_obs]).sum())
    max_restore_delta = float(np.abs(X[_obs] - X_raw[_obs]).max()) if _obs.any() else 0.0
    X[_obs] = X_raw[_obs]

    # y is the ASO target. It is never imputed, so it comes from the raw frame directly
    # rather than through the same CSV round-trip.
    y = raw[_ASO_COL].to_numpy(dtype=float)
    y_csv = df_imputed[_ASO_COL].to_numpy(dtype=float)
    max_y_delta = float(np.nanmax(np.abs(y - y_csv))) if len(y) else 0.0
    if max_y_delta > 1e-6:
        raise RuntimeError(
            f"ASO target disagrees between the raw frame and the imputation table by "
            f"{max_y_delta:g} mm; alignment is wrong, not a rounding artefact")

    # ---- residual NaN handling -------------------------------------------
    # The donor search leaves a NaN in place when a flight row has no valid donor at all
    # (`if not first_corr: continue`). Such flights cannot be used, and dropping them
    # here -- explicitly and counted -- is what keeps "every retained pillow has a value
    # on every retained flight" true rather than merely intended.
    bad = ~np.isfinite(X).all(axis=1) | ~np.isfinite(y)
    n_dropped_residual = int(bad.sum())
    if n_dropped_residual:
        log(f"  WARNING dropping {n_dropped_residual} flight(s) with residual NaN: "
            f"{[str(d.date()) for d in dates[bad]]}")
        keep = ~bad
        X, y, imputed_mask, dates = X[keep], y[keep], imputed_mask[keep], dates[keep]
        X_raw = X_raw[keep]
        df_imputed = df_imputed[keep].reset_index(drop=True)

    water_years = np.array([_water_year(d) for d in dates], dtype=int)

    table = pd.DataFrame({"time": dates, "water_year": water_years, _ASO_COL: y})
    for j, pil in enumerate(universe):
        table[pil] = X[:, j]
    for j, pil in enumerate(universe):
        table[f"{pil}{_IMPUTED_SUFFIX}"] = imputed_mask[:, j]

    per_pillow_frac = {p: float(v) for p, v in
                       zip(universe, imputed_mask.mean(axis=0))}

    # ---- manifest ---------------------------------------------------------
    manifest = {
        # identity
        "frame_id": frame_id,
        "frame_schema_version": FRAME_SCHEMA_VERSION,
        "inputs_hash": inputs_hash,
        "library_id": library_id,
        "built_at_utc": datetime.now(timezone.utc).isoformat(),
        "git": _git_provenance(repo_root),
        "python": sys.version.split()[0],
        # target
        "basin": basin,
        "elev_band": elev_band,
        "elev_idx": elev_idx,
        "elev_labels": elev_labels,
        "mode": mode,
        "aso_variable": "aso_swe",
        "target_column": _ASO_COL,
        "units": "mm",
        # provenance
        "aso_spatial": _file_provenance(fp["aso_spatial"]),
        "aso_temporal": _file_provenance(fp["aso_temporal"]),
        "obs_qa_file": obs_qa_file,
        "qa_tag": qa_tag,
        "obs_qa": _file_provenance(obs_qa_path),
        "config": _file_provenance(cfg_path),
        # universe / eligibility
        "candidate_pillows": list(candidate_pillows),
        "n_candidate_pillows": len(candidate_pillows),
        "exclude_pillows_config": exclude_pillows,
        "obs_threshold": obs_threshold,
        "eligibility_rule": ("frac_missing_at_aso_flight_dates < obs_threshold; "
                             "denominator = n_flights; strict less-than"),
        "retained_universe": list(universe),
        "n_retained": len(universe),
        "dropped_pillows": {p: float(frac_missing_all[p]) for p in dropped},
        "bitmask_bit_order": list(universe),
        # imputation
        "impute_method": IMPUTE_METHOD,
        "impute_donor_window_start_year": IMPUTE_DONOR_WINDOW_START_YEAR,
        "impute_donor_max": 3,
        "impute_negative_clipped_to_zero": True,
        "impute_cache_path": str(paths["impute_cache"]),
        "impute_cache_hit": cache_hit,
        "impute_excludes_synthetic_zero_row": True,
        "observed_cells_restored_from_raw": n_restored,
        "observed_cell_max_csv_delta_mm": max_restore_delta,
        "target_max_csv_delta_mm": max_y_delta,
        "n_imputed_cells": int(imputed_mask.sum()),
        "n_cells": int(imputed_mask.size),
        "frac_imputed_overall": float(imputed_mask.mean()),
        "frac_imputed_per_pillow": per_pillow_frac,
        "max_pillow_frac_imputed": float(max(per_pillow_frac.values())),
        "mean_pillow_frac_imputed": float(np.mean(list(per_pillow_frac.values()))),
        # training record
        "excluded_years": list(excluded_years),
        "holdout_sentinel_wy": holdout_sentinel_wy,
        "n_flights_raw": n_flights,
        "n_flights_dropped_residual_nan": n_dropped_residual,
        "flight_dates": [str(d.date()) for d in dates],
        "water_years": sorted(set(int(w) for w in water_years)),
        "flights_per_water_year": {str(k): int(v) for k, v in
                                   pd.Series(water_years).value_counts().sort_index().items()},
        "train_date_min": str(dates.min().date()),
        "train_date_max": str(dates.max().date()),
        # regression convention (D4/B3)
        "fit_intercept": fit_intercept,
        "add_zero_aso": add_zero_aso,
        "stats_exclude_synthetic_row": True,
        "stats_n_obs": int(len(y)),
        "fit_n_obs": int(len(y)) + (1 if add_zero_aso else 0),
        "synthetic_zero_row_index": "1900-01-01",
        "synthetic_zero_row_added_at": "fit_time",
        # selection policy (D1/D7/B2)
        "max_pillows": MAX_PILLOWS,
        "default_score_rule": DEFAULT_SCORE_RULE,
        "score_is_in_sample": True,
        "score_intended_use": "model_selection",
        "score_rebuild_required_to_change": False,
        "tie_break": list(TIE_BREAK),
        "support_policy": SUPPORT_POLICY,
        # shape
        "shape_X": [int(X.shape[0]), int(X.shape[1])],
        "shape_y": [int(y.shape[0])],
        # paths
        "paths": {k: str(v) for k, v in paths.items()},
    }

    frame = CanonicalFrame(y=y, X=X, imputed=imputed_mask, X_raw=X_raw,
                           universe=universe, dates=dates, water_years=water_years,
                           table=table, manifest=manifest)

    problems = validate_canonical_frame(frame)
    manifest["validation"] = {"passed": not problems, "problems": problems}
    if problems:
        raise RuntimeError("canonical frame failed validation:\n  - " +
                           "\n  - ".join(problems))
    log(f"  built X={X.shape} y={y.shape} imputed_cells={int(imputed_mask.sum())} "
        f"({imputed_mask.mean():.2%})")
    return frame


# ===========================================================================
# validation
# ===========================================================================

def validate_canonical_frame(frame: CanonicalFrame) -> list[str]:
    """Return a list of problems; empty means valid. These are the properties the
    lookup architecture depends on, asserted rather than assumed."""
    p: list[str] = []
    X, y, m, u = frame.X, frame.y, frame.imputed, frame.universe

    if X.ndim != 2:
        return [f"X is not 2-D: shape={X.shape}"]
    if y.shape != (X.shape[0],):
        p.append(f"y shape {y.shape} does not match X rows {X.shape[0]}")
    if m.shape != X.shape:
        p.append(f"imputed mask shape {m.shape} != X shape {X.shape}")
    if X.shape[1] != len(u):
        p.append(f"X has {X.shape[1]} columns but universe has {len(u)}")

    # imputation must FILL NaNs and nothing else. These two are the load-bearing
    # invariants of the whole canonical-frame idea, so they are checked on every build
    # rather than left to an external probe.
    if frame.X_raw.shape == X.shape:
        obs = ~m
        if not np.array_equal(X[obs], frame.X_raw[obs]):
            bad_n = int((X[obs] != frame.X_raw[obs]).sum())
            worst = float(np.abs(X[obs] - frame.X_raw[obs]).max())
            p.append(f"imputation altered {bad_n} OBSERVED cells "
                     f"(max delta {worst:g} mm); it must only fill NaNs")
        if not np.array_equal(np.isnan(frame.X_raw), m):
            p.append("imputed mask does not match the pre-imputation NaN pattern")
        if m.any() and not np.isfinite(X[m]).all():
            p.append("some cells flagged imputed are still non-finite")
    else:
        p.append(f"X_raw shape {frame.X_raw.shape} != X shape {X.shape}")

    # completeness: the defining property of the canonical frame
    if not np.isfinite(X).all():
        p.append(f"X contains {int((~np.isfinite(X)).sum())} non-finite cells")
    if not np.isfinite(y).all():
        p.append(f"y contains {int((~np.isfinite(y)).sum())} non-finite values")

    if list(u) != sorted(u):
        p.append("universe is not sorted; bitmask bit order would be unstable")
    if len(set(u)) != len(u):
        p.append("universe contains duplicates")

    if len(frame.dates) != X.shape[0]:
        p.append(f"dates length {len(frame.dates)} != X rows {X.shape[0]}")
    if frame.dates.has_duplicates:
        p.append("duplicate ASO flight dates")
    if not frame.dates.is_monotonic_increasing:
        p.append("dates are not sorted ascending")

    if (X < 0).any():
        p.append(f"X contains {int((X < 0).sum())} negative SWE values")
    if (y < 0).any():
        p.append(f"y contains {int((y < 0).sum())} negative SWE values")

    if X.shape[0] <= MAX_PILLOWS + 1:
        p.append(f"n_obs={X.shape[0]} too small for k={MAX_PILLOWS} "
                 f"(need n > k+1 for adjusted R2)")

    # the synthetic zero row must NOT be here; it belongs to fit time
    if pd.Timestamp("1900-01-01") in frame.dates:
        p.append("synthetic 1900-01-01 zero row present in the canonical frame")
    if (X == 0).all(axis=1).any():
        rows = np.where((X == 0).all(axis=1))[0].tolist()
        p.append(f"all-zero predictor row(s) at {rows} look like a synthetic anchor")

    return p


# ===========================================================================
# persistence
# ===========================================================================

def save_canonical_frame(frame: CanonicalFrame, *, write_csv: bool = True) -> dict[str, Path]:
    """Write the canonical matrix (Parquet, authoritative) and manifest (JSON).
    The CSV is a debug convenience only -- never read back, because a float round-trip
    through CSV is exactly the precision problem the A/B tooling had to fight."""
    paths = {k: Path(v) for k, v in frame.manifest["paths"].items()}
    paths["frame"].parent.mkdir(parents=True, exist_ok=True)
    paths["manifest"].parent.mkdir(parents=True, exist_ok=True)

    frame.table.to_parquet(paths["frame"], index=False)
    written = {"frame": paths["frame"]}
    if write_csv:
        frame.table.to_csv(paths["frame_csv"], index=False, float_format="%.17g")
        written["frame_csv"] = paths["frame_csv"]
    paths["manifest"].write_text(
        json.dumps(frame.manifest, indent=2, sort_keys=True, default=str) + "\n")
    written["manifest"] = paths["manifest"]
    written["impute_cache"] = paths["impute_cache"]
    return written


def load_canonical_frame(manifest_path: str | Path) -> CanonicalFrame:
    """Reload a frame from its manifest, verifying the Parquet still matches."""
    manifest = json.loads(Path(manifest_path).read_text())
    table = pd.read_parquet(manifest["paths"]["frame"])
    universe = tuple(manifest["retained_universe"])

    dates = pd.DatetimeIndex(pd.to_datetime(table["time"]))
    X = table[list(universe)].to_numpy(dtype=float)
    y = table[manifest["target_column"]].to_numpy(dtype=float)
    imputed = table[[f"{p}{_IMPUTED_SUFFIX}" for p in universe]].to_numpy(dtype=bool)

    if [X.shape[0], X.shape[1]] != manifest["shape_X"]:
        raise RuntimeError(f"shape mismatch: parquet {list(X.shape)} vs "
                           f"manifest {manifest['shape_X']}")
    # X_raw is recoverable rather than stored: it is X with NaN reinstated wherever the
    # mask says the cell was imputed.
    X_raw = X.copy()
    X_raw[imputed] = np.nan
    return CanonicalFrame(y=y, X=X, imputed=imputed, X_raw=X_raw, universe=universe,
                          dates=dates,
                          water_years=table["water_year"].to_numpy(dtype=int),
                          table=table, manifest=manifest)
