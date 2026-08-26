"""
phase1_stopping_rule_diagnostic.py

PHASE 1 DIAGNOSTIC ONLY. Changes no production behaviour, writes nothing to any production
path, and does not implement a replacement stopping rule.

What is being measured
----------------------
fit_best_model() does two separable things per outer fold:

    add_parameter(parameters, x, y)                     -> WHICH pillow enters next
                                                           uses training observations only
    error = RMSE(lm.predict(x_hat[:, new_parameters]),  -> WHEN forward selection stops
                 y_hat)                                    uses the HELD-OUT year's ASO

The while-loop continues while `error <= best_error`, so it keeps adding pillows until the
held-out year's RMSE stops improving, then returns the subset from BEFORE the failing
addition. So the outer test year participates in choosing model SIZE, though not model
ORDER. `best_error` -- printed by cross_val_loyo_pred_select and used nowhere else -- is the
held-out RMSE of a model whose size that same held-out data selected.

This script replays the entry order exactly (same add_parameter calls, training data only),
then compares two stopping rules along that identical sequence:

    current   stop when the OUTER held-out year's RMSE stops improving   (leaky)
    inner     stop at minimum inner leave-one-YEAR-out CV RMSE, computed
              entirely within the outer training years                   (leakage-free)

The outer year's ASO is used only to score the final prediction, once, under each rule.

Usage (conda env: bor):
    python phase1_stopping_rule_diagnostic.py FRIANT 2025 2025-04-01
    python phase1_stopping_rule_diagnostic.py FRIANT 1983 1983-04-01
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn import linear_model, metrics

sys.path.insert(1, str(Path(__file__).resolve().parent))
sys.path.insert(1, '/home/rossamower/bin/scripts/')

from mlr_ab_harness import replay_preprocessing
import lm_model as lm_model
import preprocessing as preprocessing

OUT = Path("/home/rossamower/work/aso/data_test/phase1_stopping_rule/")
MAX_STEPS = 8          # cap on sequence length explored; production is effectively unbounded


def fit(x, y, cols):
    lm = linear_model.LinearRegression()
    lm.fit(x[:, cols], y)
    return lm


def entry_order(x, y, max_steps):
    """Training-derived pillow entry sequence, using production's add_parameter verbatim."""
    seq, cur = [], []
    for _ in range(min(max_steps, x.shape[1])):
        nxt = lm_model.add_parameter(cur, x, y)
        new = [c for c in nxt if c not in cur]
        if not new:                     # add_parameter can re-propose an existing column
            break
        seq.append(new[0]); cur = nxt
    return seq


def inner_loyo_rmse(x, y, tr_years, cols):
    """Leave-one-YEAR-out CV RMSE for a fixed column set, inside the training years only."""
    pred = np.full(len(y), np.nan)
    for yy in np.unique(tr_years):
        te = np.where(tr_years == yy)[0]
        tr = np.where(tr_years != yy)[0]
        if len(tr) == 0:
            continue
        pred[te] = fit(x[tr], y[tr], cols).predict(x[te][:, cols])
    ok = ~np.isnan(pred)
    if ok.sum() == 0:
        return np.nan
    return float(metrics.root_mean_squared_error(y[ok], pred[ok]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("aso_site_name")
    ap.add_argument("water_year", type=int)
    ap.add_argument("date")
    ap.add_argument("--model", default="predict NaNs", choices=["drop NaNs", "predict NaNs"])
    ap.add_argument("--qa-file", default="pillow_wy_1980_2025_qa6.nc")
    ap.add_argument("--impute-cache-suffix", default="_qa6")
    args = ap.parse_args()

    d0 = pd.Timestamp(args.date)
    want_impute = (args.model == "predict NaNs")
    OUT.mkdir(parents=True, exist_ok=True)

    # ---- same setup as Phase 0, so cases are comparable ----
    P = replay_preprocessing(args.aso_site_name, args.water_year, args.qa_file, "wy", True)
    obs = P["obs_test_lst"]
    times = pd.DatetimeIndex(pd.to_datetime(obs[0].time.values)).normalize()
    t_idx = int(np.where(times == d0)[0][0])
    _, all_pils_QA, baseline_pils_, _ = preprocessing.process_daily_qa(
        t_idx, obs, obs, P["baseline_pils"], P["exclude_pillows"], printOutput=False)

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
    sdt = lm_model.summarize_data(obs6, aso2[:, -1])
    aso_band = aso2[:, -1]
    names = [da.name for da in obs6]

    feats = sdt[:-1, :].copy()
    feats[~np.isfinite(feats)] = 0                      # production does this in-place; do it once here
    yv = aso_band.values
    yr = np.array([int(aso_band[i].date.dt.year.values) for i in range(len(aso_band))])
    years = sorted(np.unique(yr))

    print(f"PHASE 1  {args.aso_site_name} WY{args.water_year} {d0.date()}  mode={args.model}")
    print(f"  flights={len(yv)}  outer years={len(years)}  feature universe={feats.shape[0]}")
    print(f"  production max_params default = x.shape[0]-1 = n_TRAIN_OBS-1 "
          f"(~{len(yv)-4}), not n_features -- effectively unbounded here\n")

    rows = []
    for oy in years:
        te = np.where(yr == oy)[0]
        tr = np.where(yr != oy)[0]
        X, Y = feats[:, tr].T, yv[tr]
        Xh, Yh = feats[:, te].T, yv[te]
        tr_years = yr[tr]

        seq = entry_order(X, Y, MAX_STEPS)
        if not seq:
            continue

        # --- current (leaky) rule: replicate fit_best_model's loop exactly ---
        cur_sel, err, best_err, params, newp = None, 1e7, 1e7, [], []
        max_params = X.shape[0] - 1
        while (err <= best_err) and (len(params) < max_params):
            best_err = err
            params = newp
            newp = lm_model.add_parameter(params, X, Y)
            err = float(np.sqrt(np.mean((fit(X, Y, newp).predict(Xh[:, newp]) - Yh) ** 2)))
        cur_sel = list(params)

        # --- leakage-free: min inner-LOYO RMSE along the same sequence ---
        inner = [(k, inner_loyo_rmse(X, Y, tr_years, seq[:k])) for k in range(1, len(seq) + 1)]
        inner_ok = [(k, v) for k, v in inner if np.isfinite(v)]
        k_star = min(inner_ok, key=lambda t: t[1])[0] if inner_ok else 1
        inn_sel = seq[:k_star]

        # --- predict the outer year once under each rule ---
        def outer(cols):
            if not cols:
                return np.full(len(te), np.nan)
            return fit(X, Y, cols).predict(Xh[:, cols])

        rows.append({
            'outer_year': oy, 'n_test_flights': len(te), 'n_train_flights': len(tr),
            'n_train_years': len(np.unique(tr_years)),
            'entry_order': '>'.join(names[i] for i in seq),
            'k_current': len(cur_sel), 'k_inner': k_star,
            'set_current': ','.join(names[i] for i in cur_sel),
            'set_inner': ','.join(names[i] for i in inn_sel),
            'same_set': set(cur_sel) == set(inn_sel),
            'pred_current': outer(cur_sel), 'pred_inner': outer(inn_sel), 'obs': Yh,
            'inner_curve': ';'.join(f'{k}:{v:.1f}' for k, v in inner_ok),
        })

    d = pd.DataFrame(rows)

    def pooled(col):
        p = np.concatenate(d[col].values); o = np.concatenate(d['obs'].values)
        ok = np.isfinite(p) & np.isfinite(o)
        return (float(metrics.root_mean_squared_error(o[ok], p[ok])),
                float(np.corrcoef(p[ok], o[ok])[0, 1] ** 2))
    rmse_c, r2_c = pooled('pred_current')
    rmse_i, r2_i = pooled('pred_inner')

    uni_c = sorted({p for s in d.set_current for p in s.split(',') if p})
    uni_i = sorted({p for s in d.set_inner for p in s.split(',') if p})

    print(f"  {'year':<6}{'trF':>4}{'trY':>4}  {'k_cur':>5}{'k_inn':>6}  {'same':>5}  "
          f"{'set_current':<26}{'set_inner':<26}")
    for _, r in d.iterrows():
        print(f"  {r.outer_year:<6}{r.n_train_flights:>4}{r.n_train_years:>4}  "
              f"{r.k_current:>5}{r.k_inner:>6}  {str(r.same_set):>5}  "
              f"{r.set_current:<26}{r.set_inner:<26}")

    chg = int((~d.same_set).sum())
    print(f"\n  selection changed: {chg}/{len(d)} folds ({100*chg/len(d):.0f}%)")
    print(f"  model size  current mean {d.k_current.mean():.2f}  inner mean {d.k_inner.mean():.2f}"
          f"   -> inner favours {'SIMPLER' if d.k_inner.mean() < d.k_current.mean() else 'MORE COMPLEX' if d.k_inner.mean() > d.k_current.mean() else 'SAME'}")
    print(f"  station_lst  current n={len(uni_c)}: {','.join(uni_c)}")
    print(f"               inner   n={len(uni_i)}: {','.join(uni_i)}")
    print(f"               jaccard={len(set(uni_c)&set(uni_i))/max(len(set(uni_c)|set(uni_i)),1):.2f}"
          f"  only-current={sorted(set(uni_c)-set(uni_i))}  only-inner={sorted(set(uni_i)-set(uni_c))}")
    print(f"\n  pooled outer-year skill (honest for inner; optimistic for current):")
    print(f"    current : RMSE {rmse_c:7.2f}  R2 {r2_c:.4f}")
    print(f"    inner   : RMSE {rmse_i:7.2f}  R2 {r2_i:.4f}"
          f"   -> {rmse_i-rmse_c:+.2f} mm ({100*(rmse_i-rmse_c)/rmse_c:+.1f}%)")

    dd = d.copy()
    dd['d_pred_mean'] = [float(np.nanmean(np.abs(a - b))) for a, b in zip(d.pred_current, d.pred_inner)]
    print(f"\n  outer-year prediction change |current - inner| (mm): "
          f"mean {dd.d_pred_mean.mean():.1f}  max {dd.d_pred_mean.max():.1f}")
    print(f"  sensitivity vs training years: "
          + '  '.join(f'{int(r.n_train_years)}y:{r.d_pred_mean:.0f}mm' for _, r in dd.iterrows()))

    keep = ['outer_year', 'n_test_flights', 'n_train_flights', 'n_train_years', 'entry_order',
            'k_current', 'k_inner', 'set_current', 'set_inner', 'same_set', 'inner_curve']
    fp = OUT / f"{args.aso_site_name}_wy{args.water_year}_{d0.date()}_{args.model.replace(' ','')}_phase1.csv"
    d[keep].to_csv(fp, index=False)
    print(f"\n  wrote {fp}")


if __name__ == "__main__":
    main()
