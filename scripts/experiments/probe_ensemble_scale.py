"""
probe_ensemble_scale.py

Size the ensemble work before implementing it. Two unknowns drive every estimate:

  1. |station_lst| in identify_best_stations -- the union of pillows selected across CV#1
     folds. combinations(station_lst, 1..5) is enumerated exhaustively, so this sets how
     many rows the ranked table has. C(8,1..5)=218 but C(15,1..5)=4943, a 20x spread.

  2. cost of one run_daily_prediction call -- predict_with_cached_training does
     2 (units) x 16 (isImpute x elev_band) = 32 of them per day today. With n_ensemble=N
     that becomes 32N, so this number sets the prediction-side cost.

Measures both without editing lm_model.py: wraps identify_best_stations to record what it
was given, and times run_daily_prediction on real cached training state.

Usage (conda env: bor):
    python probe_ensemble_scale.py FRIANT 2017
"""

import argparse
import time
from datetime import datetime
from itertools import combinations
from math import comb

import pandas as pd

from mlr_ab_harness import replay_preprocessing
import lm_model as lm_model
import preprocessing as preprocessing


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("aso_site_name")
    ap.add_argument("water_year", type=int)
    ap.add_argument("--qa-file", default="pillow_wy_1980_2025_qa6.nc")
    ap.add_argument("--t-idx", type=int, default=100)
    args = ap.parse_args()

    site, wy = args.aso_site_name, args.water_year
    P = replay_preprocessing(site, wy, args.qa_file, "wy", True)
    obs = P["obs_test_lst"]

    # --- instrument identify_best_stations without touching lm_model.py ---
    observed = []
    real = lm_model.identify_best_stations

    def spy(best_stations, aso_tseries_2, elev_band, summary_data_total,
            isCombination=True, showOutput=True):
        station_lst = list(set(sum(best_stations, [])))
        n = len(station_lst)
        n_combos = sum(comb(n, i) for i in range(1, min(5, n) + 1))
        t0 = time.perf_counter()
        out = real(best_stations, aso_tseries_2, elev_band, summary_data_total,
                   isCombination, showOutput)
        observed.append({"n_folds": len(best_stations), "n_station_lst": n,
                         "n_combos_enumerated": n_combos,
                         "seconds": round(time.perf_counter() - t0, 3),
                         "n_selected": len(out)})
        return out

    lm_model.identify_best_stations = spy

    t_np = obs[0].time[args.t_idx].values
    ts = pd.to_datetime(t_np)
    current_date = datetime(ts.year, ts.month, ts.day)
    cur_vals, all_pils_QA, baseline_pils_, _ = preprocessing.process_daily_qa(
        args.t_idx, obs, obs, P["baseline_pils"], P["exclude_pillows"], printOutput=False)

    print(f"\nPROBE  {site} wy{wy}  t_idx={args.t_idx} ({current_date.date()})")
    print(f"  available pillows (all_pils_QA) : {len(all_pils_QA)}")
    print(f"  baseline pillows                : {len(baseline_pils_)}\n")

    t0 = time.perf_counter()
    cache = lm_model.train_all_mlr_models(
        P["aso_tseries_train"].aso_swe, P["obs_data_qa"], site, P["all_pils"],
        all_pils_QA, P["df_sum_total"], baseline_pils_, P["start_wy"], P["end_wy"],
        False, False, P["mlrPred_dir"], current_date, P["dem_bin"].dem_bin,
        QA_flag=1, modelNUM=0, isMean=False, showOutput=False, saveValidation=False,
        isCombination_=True, pillowImputation_=True, ds_snowmodel_=None,
        impute_cache_suffix="_qa6")
    t_train = time.perf_counter() - t0
    lm_model.identify_best_stations = real

    d = pd.DataFrame(observed)
    print(f"=== identify_best_stations: {len(d)} calls (= 2 isImpute x 8 elev_band) ===")
    print(d.to_string(index=False))
    print(f"\n  |station_lst|        min={d.n_station_lst.min()}  "
          f"median={int(d.n_station_lst.median())}  max={d.n_station_lst.max()}")
    print(f"  combos enumerated    min={d.n_combos_enumerated.min()}  "
          f"median={int(d.n_combos_enumerated.median())}  max={d.n_combos_enumerated.max()}"
          f"   TOTAL={d.n_combos_enumerated.sum()}")
    print(f"  selection time       total={d.seconds.sum():.1f}s of {t_train:.1f}s training")
    print(f"\n  => ranked table rows per availability set = {d.n_combos_enumerated.sum()}"
          f" (already computed today, currently discarded)")

    # --- cost of one run_daily_prediction call ---
    c = cache[0]
    cv = cur_vals.reset_index(names='time')
    n_rep = 25
    t0 = time.perf_counter()
    for _ in range(n_rep):
        lm_model.run_daily_prediction(c['df_split'], cv, c['selected_pils'],
                                      c['modelID'], conversion=0, area_m2=c['area_m2'],
                                      outdir=None, add_zeroASO=True, fit_intercept=False)
    per_call = (time.perf_counter() - t0) / n_rep
    print(f"\n=== run_daily_prediction ===")
    print(f"  per call                  : {per_call*1000:7.2f} ms")
    print(f"  today  32/day  x 365 days : {per_call*32*365:7.1f} s per job")
    for N in (5, 10, 20):
        print(f"  n_ensemble={N:<3d} {32*N}/day     : {per_call*32*N*365:7.1f} s per job"
              f"   (+{per_call*32*(N-1)*365/60:.1f} min)")


if __name__ == "__main__":
    main()
