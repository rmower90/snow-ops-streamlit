"""
pillow_selection_table.py

Dump EVERY pillow combination the selector considered on a single date, with its
cross-validated adjusted R2 and its basin-SWE prediction.

Why this exists. The n_ensemble=10 band answers "which marginal pillow should we add?" --
every top-10 member shares the dominant stations, so the spread measures refinement, not
risk. What we actually want to characterise is "what if a station we lean on is missing or
wrong?" Answering that needs the whole adj_r2/prediction landscape, not the top of it, so
that alternative constructions can be evaluated against real numbers:

  sequential exclusion  rank2 excludes rank1's leading pillow, rank3 excludes ranks 1-2's, ...
  station jackknife     for each available pillow p, the best combination excluding p
  leave-two-out         same, for every pair -- catches correlated failure between twins

All three are just different queries over this table.

Two enumeration widths, and the choice matters:

  narrow (default)  combinations(station_lst, 1..5), where station_lst is the union of
                    pillows the cross-validation folds nominated -- typically 3-10 of them.
                    This is exactly what production already computes and discards, so it is
                    nearly free. But if excluding a dominant pillow pushes the best
                    alternative OUTSIDE what CV nominated, this width cannot see it.

  --wide            combinations(all available pillows, 1..5). C(25,1..5) = 68,405, roughly
                    15 min for one date/band/mode. Needed if the narrow table shows the good
                    alternatives clustering at its edge.

Run narrow first on a data-rich and a data-poor date; the narrow table tells you whether
wide is necessary.

Usage (conda env: bor):
    python pillow_selection_table.py FRIANT 2025 2025-04-01
    python pillow_selection_table.py FRIANT 1983 1983-04-01
    python pillow_selection_table.py FRIANT 2025 2025-04-01 --wide
    python pillow_selection_table.py USCASJ 2026 2026-02-25 --site-kind daily
"""

import argparse
import sys
import time
from datetime import datetime
from itertools import combinations
from math import comb
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(1, str(Path(__file__).resolve().parent))
sys.path.insert(1, '/home/rossamower/bin/scripts/')

from mlr_ab_harness import replay_preprocessing
import lm_model as lm_model
import preprocessing as preprocessing

DEFAULT_OUT = "/home/rossamower/work/aso/data_test/pillow_selection/"


def widen_selector():
    """
    Force identify_best_stations to enumerate over ALL pillows rather than the CV-nominated
    subset. station_lst is derived inside the function as list(set(sum(best_stations, []))),
    so handing it one singleton list per pillow index makes that the full index range without
    touching lm_model.py.
    """
    real = lm_model.identify_best_stations

    def wide(best_stations, aso_tseries_2, elev_band, summary_data_total,
             isCombination=True, showOutput=True, ranked_sink=None):
        n_pillows = summary_data_total.shape[0] - 1     # last row is the ASO series
        return real([[i] for i in range(n_pillows)], aso_tseries_2, elev_band,
                    summary_data_total, isCombination, showOutput, ranked_sink)

    lm_model.identify_best_stations = wide
    return real


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("aso_site_name")
    ap.add_argument("water_year", type=int)
    ap.add_argument("date", help="YYYY-MM-DD")
    ap.add_argument("--elev-band", default="total")
    ap.add_argument("--model", default="predict NaNs",
                    choices=["drop NaNs", "predict NaNs"])
    ap.add_argument("--wide", action="store_true",
                    help="enumerate all available pillows, not just the CV-nominated subset")
    ap.add_argument("--top-k", type=int, default=200000, help="retain at most this many combos")
    ap.add_argument("--qa-file", default="pillow_wy_1980_2025_qa6.nc")
    ap.add_argument("--impute-cache-suffix", default="_qa6")
    ap.add_argument("--fit-intercept", type=int, default=0)
    ap.add_argument("--out-dir", default=DEFAULT_OUT)
    args = ap.parse_args()

    d0 = pd.Timestamp(args.date)
    want_impute = (args.model == "predict NaNs")
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    print(f"PILLOW SELECTION TABLE  {args.aso_site_name} WY{args.water_year} {d0.date()}")
    print(f"  band={args.elev_band}  model={args.model}  "
          f"enumeration={'WIDE (all pillows)' if args.wide else 'narrow (CV-nominated)'}")

    t0 = time.perf_counter()
    P = replay_preprocessing(args.aso_site_name, args.water_year, args.qa_file, "wy", True)
    print(f"  preprocessing replay: {time.perf_counter()-t0:.1f}s")

    # locate the requested date in the test-side series
    obs = P["obs_test_lst"]
    times = pd.DatetimeIndex(pd.to_datetime(obs[0].time.values)).normalize()
    hits = np.where(times == d0)[0]
    if not len(hits):
        raise SystemExit(f"{d0.date()} not in WY{args.water_year} "
                         f"({times.min().date()}..{times.max().date()})")
    t_idx = int(hits[0])

    cur_vals, all_pils_QA, baseline_pils_, _ = preprocessing.process_daily_qa(
        t_idx, obs, obs, P["baseline_pils"], P["exclude_pillows"], printOutput=False)
    print(f"  t_idx={t_idx}  n_QA={len(all_pils_QA)}  n_baseline={len(baseline_pils_)}")
    if not baseline_pils_ or not all_pils_QA:
        raise SystemExit("0 usable pillows on this date")

    if args.wide:
        n = len(baseline_pils_ if not want_impute else all_pils_QA)
        print(f"  wide enumeration will score ~{sum(comb(n, i) for i in range(1, 6)):,} "
              f"combinations -- expect minutes")
        widen_selector()

    # capture_ranked=True also populates frames_sink, which is what lets each combination be
    # predicted: cached['df_split'] carries only rank 1's pillow columns.
    t0 = time.perf_counter()
    cache = lm_model.train_all_mlr_models(
        P["aso_tseries_train"].aso_swe, P["obs_data_qa"], args.aso_site_name,
        P["all_pils"], all_pils_QA, P["df_sum_total"], baseline_pils_,
        P["start_wy"], P["end_wy"], False, False, P["mlrPred_dir"],
        datetime(d0.year, d0.month, d0.day), P["dem_bin"].dem_bin,
        QA_flag=1, modelNUM=0, isMean=False, showOutput=False, saveValidation=False,
        isCombination_=True, pillowImputation_=True, ds_snowmodel_=None,
        impute_cache_suffix=args.impute_cache_suffix,
        capture_ranked=True, ranked_top_k=args.top_k)
    print(f"  selection + frames: {time.perf_counter()-t0:.1f}s")

    # pick the (isImpute, elev_band) cache entry we asked for
    entry = None
    for c in cache:
        mf = c['summary_dict_model'][c['modelID']]['model_features']
        if bool(mf.get('isImpute')) == want_impute and str(mf.get('elevation_band')) == args.elev_band:
            entry = c; break
    if entry is None:
        avail = sorted({str(c['summary_dict_model'][c['modelID']]['model_features'].get('elevation_band'))
                        for c in cache})
        raise SystemExit(f"no cache entry for band={args.elev_band}; available: {avail}")

    ranked = entry.get('ranked_combos') or []
    frames = entry.get('ranked_frames') or []
    print(f"  combinations retained: {len(ranked):,}")

    cv = cur_vals.reset_index(names='time')
    rows = []
    t0 = time.perf_counter()
    for i, (adj_r2, pils) in enumerate(ranked):
        pred = err = None
        if i < len(frames):
            try:
                pred, _, _ = lm_model.run_daily_prediction(
                    frames[i].copy(), cv, list(pils), entry['modelID'], conversion=0,
                    area_m2=entry['area_m2'], outdir=None, add_zeroASO=True,
                    fit_intercept=bool(args.fit_intercept))
                pred = float(pred)
            except Exception as exc:
                err = f'{type(exc).__name__}: {str(exc)[:120]}'
        rows.append({'rank': i + 1, 'adj_r2': float(adj_r2), 'n_pillows': len(pils),
                     'pillows': ','.join(pils), 'pred_mm': pred, 'error': err})
    print(f"  predictions: {time.perf_counter()-t0:.1f}s")

    df = pd.DataFrame(rows)
    df['date'] = str(d0.date())
    df['water_year'] = args.water_year
    df['elev_band'] = args.elev_band
    df['model'] = args.model
    df['n_QA'] = len(all_pils_QA)
    df['n_baseline'] = len(baseline_pils_)
    df['pillows_QA'] = ','.join(sorted(map(str, all_pils_QA)))
    df['enumeration'] = 'wide' if args.wide else 'narrow'

    tag = 'wide' if args.wide else 'narrow'
    mtag = args.model.replace(' ', '')
    fp = out_dir / (f"{args.aso_site_name}_wy{args.water_year}_{d0.date()}_"
                    f"{args.elev_band.replace('<','lt').replace('>','gt')}_{mtag}_{tag}.csv")
    df.to_csv(fp, index=False)

    # the question the table exists to answer: when the leading pillow of rank 1 is removed,
    # how far does the best surviving combination fall, and is it reachable at this width?
    r1 = df.iloc[0]
    lead = r1.pillows.split(',')
    print(f"\n  rank 1: {r1.pillows}  adj_r2={r1.adj_r2:.4f}  pred={r1.pred_mm}")
    print(f"  {'pillow excluded':<18} {'best adj_r2':>11} {'d adj_r2':>9} {'pred_mm':>9} {'d pred':>8}  best combo")
    universe = sorted(set(p for s in df.pillows for p in s.split(',')))
    for p in universe:
        sub = df[~df.pillows.str.split(',').apply(lambda x: p in x)]
        if sub.empty:
            print(f"  {p:<18} {'-- no combination excludes it --':>40}"); continue
        b = sub.iloc[0]
        star = '  <-- in rank 1' if p in lead else ''
        dp = (b.pred_mm - r1.pred_mm) if (b.pred_mm is not None and r1.pred_mm is not None) else float('nan')
        print(f"  {p:<18} {b.adj_r2:>11.4f} {b.adj_r2-r1.adj_r2:>+9.4f} "
              f"{b.pred_mm if b.pred_mm is None else round(b.pred_mm,1):>9} {dp:>+8.1f}  {b.pillows}{star}")

    print(f"\n  wrote {fp}  ({len(df):,} rows)")


if __name__ == "__main__":
    main()
