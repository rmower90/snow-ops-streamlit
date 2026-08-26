"""
phase0_cv_scoring_diagnostic.py

PHASE 0 ONLY -- diagnostic. Changes no production behaviour and writes nothing into any
production path.

Question: identify_best_stations() ranks candidate pillow combinations using cross_val_loo,
which holds out one ASO FLIGHT at a time. ASO flies ~4 times per season and snowpack is
strongly autocorrelated within a season, so a held-out flight leaves ~3 near-duplicate
siblings from the same year in the training set. That should make the ranking statistic
optimistic. How optimistic, and does it change any conclusion we have drawn?

Method: reproduce the exact candidate set the production selector enumerated for a given
(date, elevation band, imputation mode) -- CV#1 -> station_lst -> combinations(1..5) -- then
score every one of those combinations BOTH ways:

    flight-wise  cross_val_loo   (production)
    year-wise    cross_val_loyo  (all flights from a year held out together, fixed
                                  combination, no feature selection inside the fold)

Today's inferred SWE is computed once per combination. It depends on today's pillow values
and the training frame, neither of which involves the scoring method, so it must be
identical under both -- reported as a control rather than assumed.

Usage (conda env: bor):
    python phase0_cv_scoring_diagnostic.py FRIANT 2025 2025-04-01
    python phase0_cv_scoring_diagnostic.py FRIANT 1983 1983-04-01
"""

import argparse
import sys
from datetime import datetime
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn import metrics

sys.path.insert(1, str(Path(__file__).resolve().parent))
sys.path.insert(1, '/home/rossamower/bin/scripts/')

from mlr_ab_harness import replay_preprocessing
import lm_model as lm_model
import preprocessing as preprocessing

OUT = Path("/home/rossamower/work/aso/data_test/phase0_cv_scoring/")


def score(aso, index, fn):
    """RMSE / R2 / adj-R2 for one fixed combination under one CV scheme."""
    pred = np.array(fn(aso, index.copy()))       # copy: cross_val_* may zero non-finites in place
    pred[pred < 0] = 0
    obs = aso.values
    rmse = float(metrics.root_mean_squared_error(obs, pred))
    r2 = float(np.corrcoef(pred, obs)[0, 1] ** 2)
    n, k = len(pred), index.shape[0]
    adj = 1 - ((1 - r2) * (n - 1) / (n - k - 1)) if n - k - 1 > 0 else np.nan
    return rmse, r2, adj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("aso_site_name")
    ap.add_argument("water_year", type=int)
    ap.add_argument("date")
    ap.add_argument("--elev-band", default="total")
    ap.add_argument("--model", default="predict NaNs", choices=["drop NaNs", "predict NaNs"])
    ap.add_argument("--qa-file", default="pillow_wy_1980_2025_qa6.nc")
    ap.add_argument("--impute-cache-suffix", default="_qa6")
    ap.add_argument("--fit-intercept", type=int, default=0)
    args = ap.parse_args()

    d0 = pd.Timestamp(args.date)
    want_impute = (args.model == "predict NaNs")
    OUT.mkdir(parents=True, exist_ok=True)

    P = replay_preprocessing(args.aso_site_name, args.water_year, args.qa_file, "wy", True)
    obs = P["obs_test_lst"]
    times = pd.DatetimeIndex(pd.to_datetime(obs[0].time.values)).normalize()
    t_idx = int(np.where(times == d0)[0][0])
    cur_vals, all_pils_QA, baseline_pils_, _ = preprocessing.process_daily_qa(
        t_idx, obs, obs, P["baseline_pils"], P["exclude_pillows"], printOutput=False)

    # --- reproduce run_cross_val_selection2 up to summary_data_total, for this mode ---
    elev_idx = -1  # 'total' is the last elev entry, matching production's elev_band index
    aso_ts = P["aso_tseries_train"].aso_swe
    if want_impute:
        obs_use, universe, df_use = lm_model.impute_pillow_prediction(
            P["df_sum_total"], all_pils_QA, P["obs_data_qa"], args.aso_site_name,
            datetime(d0.year, d0.month, d0.day), obs_threshold=0.50,
            cache_suffix=args.impute_cache_suffix)
        universe = list(universe)
    else:
        obs_use, universe, df_use = P["obs_data_qa"], list(baseline_pils_), P["df_sum_total"]

    obs6 = [o for o in obs_use if o.name in universe]
    missing = df_use[df_use[universe].isnull().sum(axis=1) > 0].time.values
    aso2 = aso_ts[~aso_ts.date.isin(missing)]
    sdt = lm_model.summarize_data(obs6, aso2[:, elev_idx])
    aso_band = aso2[:, elev_idx]

    # --- CV#1 -> station_lst : the exact candidate pool production enumerated ---
    _, best_stations = lm_model.cross_val_loyo_pred_select(
        aso_band, sdt[:-1, :].copy(), P["start_wy"], P["end_wy"], showOutput=False)
    station_lst = list(set(sum(best_stations, [])))
    names = [da.name for da in obs6]
    yrs = sorted({int(aso_band[i].date.dt.year.values) for i in range(len(aso_band))})

    print(f"PHASE 0  {args.aso_site_name} WY{args.water_year} {d0.date()}  "
          f"band={args.elev_band}  mode={args.model}")
    print(f"  flights={len(aso_band)}  years={len(yrs)}  universe={len(universe)}  "
          f"|station_lst|={len(station_lst)}")

    # --- score every combination both ways ---
    rows = []
    for i in range(1, 6):
        for val in combinations(station_lst, i):
            idx = list(val)
            rf, r2f, af = score(aso_band, sdt[idx, :], lm_model.cross_val_loo)
            ry, r2y, ay = score(aso_band, sdt[idx, :], lm_model.cross_val_loyo)
            rows.append({'pillows': ','.join(names[j] for j in idx), 'n_pillows': len(idx),
                         'rmse_flight': rf, 'rmse_year': ry, 'r2_flight': r2f, 'r2_year': r2y,
                         'adjr2_flight': af, 'adjr2_year': ay, '_idx': tuple(idx)})
    df = pd.DataFrame(rows)

    # --- today's prediction per combination (control: independent of scoring method) ---
    def frame_for(idxs):
        fr = aso_band.to_dataframe()
        for j in idxs:
            di = obs6[j][obs6[j].time.isin(fr.index.values)].to_dataframe()
            di.index.names = ['date']
            fr = pd.merge(left=fr, right=di, left_index=True, right_index=True)
        return fr

    cv = cur_vals.reset_index(names='time')
    preds = []
    for _, r in df.iterrows():
        try:
            yh, _, _ = lm_model.run_daily_prediction(
                frame_for(list(r['_idx'])).copy(), cv, r['pillows'].split(','), 0,
                conversion=0, area_m2=1.0, outdir=None, add_zeroASO=True,
                fit_intercept=bool(args.fit_intercept))
            preds.append(float(yh))
        except Exception:
            preds.append(np.nan)
    df['pred_mm'] = preds
    df = df.drop(columns=['_idx'])

    df['rank_flight'] = df.adjr2_flight.rank(ascending=False, method='min').astype(int)
    df['rank_year'] = df.adjr2_year.rank(ascending=False, method='min').astype(int)
    df['rank_flight_rmse'] = df.rmse_flight.rank(ascending=True, method='min').astype(int)
    df['rank_year_rmse'] = df.rmse_year.rank(ascending=True, method='min').astype(int)

    fp = OUT / (f"{args.aso_site_name}_wy{args.water_year}_{d0.date()}_"
                f"{args.elev_band}_{args.model.replace(' ','')}_phase0.csv")
    df.to_csv(fp, index=False)

    # ---------------- compact summary ----------------
    f1 = df.loc[df.rank_flight.idxmin()]
    y1 = df.loc[df.rank_year.idxmin()]
    rho = spearmanr(df.adjr2_flight, df.adjr2_year).statistic
    rho_rmse = spearmanr(df.rmse_flight, df.rmse_year).statistic
    topN = min(10, len(df))
    ovl = len(set(df.nsmallest(topN, 'rank_flight').pillows) &
              set(df.nsmallest(topN, 'rank_year').pillows))

    print(f"\n  combinations evaluated: {len(df)}")
    print(f"  RANK 1 flight-wise : {f1.pillows:<28} adjR2={f1.adjr2_flight:.4f} "
          f"RMSE={f1.rmse_flight:7.2f}  pred={f1.pred_mm:7.1f}")
    print(f"  RANK 1 year-wise   : {y1.pillows:<28} adjR2={y1.adjr2_year:.4f} "
          f"RMSE={y1.rmse_year:7.2f}  pred={y1.pred_mm:7.1f}")
    print(f"  rank 1 changed     : {f1.pillows != y1.pillows}")
    print(f"  flight-wise rank1 re-scored year-wise: adjR2={f1.adjr2_year:.4f} "
          f"RMSE={f1.rmse_year:.2f}")
    print(f"\n  optimism (flight - year), across all combinations:")
    print(f"    adjR2 : mean {(df.adjr2_flight-df.adjr2_year).mean():+.4f}  "
          f"median {(df.adjr2_flight-df.adjr2_year).median():+.4f}  "
          f"max {(df.adjr2_flight-df.adjr2_year).max():+.4f}")
    print(f"    RMSE  : mean {(df.rmse_year-df.rmse_flight).mean():+.2f} mm  "
          f"median {(df.rmse_year-df.rmse_flight).median():+.2f}  "
          f"max {(df.rmse_year-df.rmse_flight).max():+.2f}   (year - flight; +ve = flight optimistic)")
    print(f"    RMSE ratio year/flight: median {(df.rmse_year/df.rmse_flight).median():.2f}x")
    print(f"\n  ranking stability: Spearman(adjR2) = {rho:+.3f}   "
          f"Spearman(RMSE) = {rho_rmse:+.3f}   top-{topN} overlap = {ovl}/{topN}")

    # separation between rank1 and rank10 under each scheme
    for lbl, rc, mc in (('flight', 'rank_flight', 'adjr2_flight'), ('year', 'rank_year', 'adjr2_year')):
        s = df.nsmallest(topN, rc)
        print(f"  separation top-{topN} {lbl:<6}: adjR2 {s[mc].max():.4f} -> {s[mc].min():.4f} "
              f"(spread {s[mc].max()-s[mc].min():.4f})")

    # prediction spread across the top-N under each ranking
    for lbl, rc in (('flight', 'rank_flight'), ('year', 'rank_year')):
        s = df.nsmallest(topN, rc)
        print(f"  pred spread top-{topN} {lbl:<6}: sd={s.pred_mm.std():6.2f}  "
              f"range {s.pred_mm.min():.1f}-{s.pred_mm.max():.1f} "
              f"({s.pred_mm.max()-s.pred_mm.min():.1f} mm)")

    # jackknife under each scheme
    print(f"\n  jackknife (best combination excluding each pillow):")
    for lbl, rc in (('flight', 'rank_flight'), ('year', 'rank_year')):
        base = df.loc[df[rc].idxmin()]
        lead = set(base.pillows.split(','))
        deltas = {}
        for p in sorted({q for s in df.pillows for q in s.split(',')}):
            sub = df[~df.pillows.str.split(',').apply(lambda x: p in x)]
            if sub.empty:
                continue
            b = sub.loc[sub[rc].idxmin()]
            deltas[p] = b.pred_mm - base.pred_mm
        nz = {k: v for k, v in deltas.items() if abs(v) > 1e-9}
        lo = base.pred_mm + min(list(nz.values()) + [0.0])
        hi = base.pred_mm + max(list(nz.values()) + [0.0])
        print(f"    {lbl:<6}: base={base.pred_mm:7.1f}  n_nonzero={len(nz)}/{len(deltas)}  "
              f"span {lo:.1f}-{hi:.1f} = {hi-lo:.1f} mm  "
              f"worst={max(nz, key=lambda k: abs(nz[k])) if nz else '-'} "
              f"({max(nz.values(), key=abs) if nz else 0:+.1f})")
    print(f"\n  wrote {fp}")


if __name__ == "__main__":
    main()
