"""
time_impute_cache.py

Measure the cost of the imputation-cache coverage guard added in step 3.

The guard rebuilds the imputation table whenever the cached columns do not cover the
current day's pils_removed. Rebuilds overwrite rather than accumulate, so with variable
pillow availability the cache degrades toward recompute-on-change. Correctness is fine;
the open question is what that costs at real scale:

    FRIANT historic   46 concurrent jobs, 1 water year each
    USCASJ daily      6 concurrent jobs, 36 distinct pillow sets per season

This times impute_pillow_prediction() directly in three states, which isolates the
per-rebuild cost from everything else in a harness run:

    cold    no table on disk           -> full triple-nested correlation search
    warm    table present, covers      -> read_csv only (the intended fast path)
    miss    table present, misses      -> guard fires, full rebuild

Usage (conda env: bor):
    python time_impute_cache.py FRIANT 2017
"""

import argparse
import os
import time
from pathlib import Path

import pandas as pd

from mlr_ab_harness import replay_preprocessing  # same preprocessing replay
import lm_model as lm_model
import preprocessing as preprocessing

SUFFIX = "_timing"


def impute_path(site, wy, suffix):
    return Path(f"/home/rossamower/work/aso/data/mlr_prediction/{site}/imputation/"
                f"pillow_impute_threePils_wy{wy}{suffix}.csv")


def timed(fn):
    t0 = time.perf_counter()
    out = fn()
    return time.perf_counter() - t0, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("aso_site_name")
    ap.add_argument("water_year", type=int)
    ap.add_argument("--qa-file", default="pillow_wy_1980_2025_qa6.nc")
    args = ap.parse_args()

    site, wy = args.aso_site_name, args.water_year
    print(f"TIMING imputation cache  site={site} wy={wy}\n")

    P = replay_preprocessing(site, wy, args.qa_file, "wy", True)

    # two availability sets from the real run: a narrow early-season one and the full set.
    # the narrow one written first is what makes the wide one a coverage miss.
    obs = P["obs_test_lst"]
    _, narrow_QA, narrow_base, _ = preprocessing.process_daily_qa(
        1, obs, obs, P["baseline_pils"], P["exclude_pillows"], printOutput=False)
    wide_QA = list(P["all_pils"])

    from datetime import datetime
    pdate = datetime(wy - 1, 10, 2)
    fpath = impute_path(site, wy, SUFFIX)
    fpath.unlink(missing_ok=True)

    call = lambda pils: lm_model.impute_pillow_prediction(
        P["df_sum_total"], pils, P["obs_data_qa"], site, pdate,
        obs_threshold=0.50, cache_suffix=SUFFIX)

    print(f"  narrow availability : {len(narrow_QA)} pillows")
    print(f"  wide availability   : {len(wide_QA)} pillows\n")

    t_cold, _ = timed(lambda: call(narrow_QA))
    n_cold = len(pd.read_csv(fpath, nrows=0).columns) - 2
    print(f"  cold  (build from nothing, {n_cold} cols) : {t_cold:8.2f} s")

    t_warm, _ = timed(lambda: call(narrow_QA))
    print(f"  warm  (covers, read_csv only)            : {t_warm:8.2f} s")

    t_miss, _ = timed(lambda: call(wide_QA))
    n_miss = len(pd.read_csv(fpath, nrows=0).columns) - 2
    print(f"  miss  (guard fires, rebuild {n_miss} cols)  : {t_miss:8.2f} s")

    fpath.unlink(missing_ok=True)

    print(f"\n  rebuild penalty vs warm read: {t_miss - t_warm:.2f} s "
          f"({t_miss / max(t_warm, 1e-9):.0f}x)")
    print(f"\n  extrapolation:")
    print(f"    FRIANT historic, 46 jobs x 2-3 rebuilds : "
          f"{46 * 2.5 * (t_miss - t_warm) / 60:6.1f} min of added CPU (concurrent, 7 nodes)")
    print(f"    USCASJ daily,  6 jobs x up to 36 sets   : "
          f"{6 * 36 * (t_miss - t_warm) / 60:6.1f} min worst case if every set misses")


if __name__ == "__main__":
    main()
