"""
build_model_library.py

Step 2 of the pretrained MLR lookup work: fit every allowed pillow combination against the
canonical frame from step 1, store the coefficients and diagnostics, and demonstrate the
daily lookup operation.

    canonical frame (35 x 25)  ->  C(25, 1..5) = 68,405 fitted models  ->  parquet
    today's available pillows  ->  bitmask filter  ->  best eligible model  ->  prediction

Nothing here touches the daily or historic production runners.

Validation performed:
  - exact model count, no duplicate pillow_key
  - stored coefficients and every metric re-derived by an independent sklearn refit
  - rank 1 from the table vs an independent brute-force search over the same combinations,
    at full availability and under several restricted availability sets
  - tie-break determinism
  - parquet round-trip

Usage (conda env: bor):
    python build_model_library.py FRIANT
    python build_model_library.py FRIANT --no-brute-force      # skip the 68k-fit recheck
    python build_model_library.py FRIANT --dry-run
"""

import argparse
import math
import sys
import time
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn import linear_model

sys.path.insert(1, str(Path(__file__).resolve().parents[2] / "src"))

from snow_ops.mlr.lookup import frame as cf          # noqa: E402
from snow_ops.mlr.lookup import fit as lf            # noqa: E402
from snow_ops.mlr.lookup import store as ls          # noqa: E402
from snow_ops.mlr.lookup.query import ModelLibrary, model_pillows  # noqa: E402


# ---------------------------------------------------------------------------
# Independent reference implementations. Deliberately NOT reusing fit.py: a cross-check
# that shares code with the thing it checks only proves self-consistency.
# ---------------------------------------------------------------------------

def reference_fit(X, y, cols, *, fit_intercept, add_zero_aso):
    """Refit one combination from scratch and recompute every metric by hand."""
    n, k = X.shape[0], len(cols)
    Xs = X[:, cols]
    if add_zero_aso:
        Xf = np.vstack([Xs, np.zeros((1, k))])
        yf = np.append(y, 0.0)
    else:
        Xf, yf = Xs, y

    lm = linear_model.LinearRegression(fit_intercept=fit_intercept).fit(Xf, yf)
    resid = y - lm.predict(Xs)
    sse = float((resid ** 2).sum())
    tss_c = float(((y - y.mean()) ** 2).sum())
    tss_u = float((y ** 2).sum())
    dfr = n - k

    r2 = 1 - sse / tss_c
    out = {
        "intercept": float(lm.intercept_) if fit_intercept else 0.0,
        "coefs": [float(c) for c in lm.coef_],
        "r2": r2,
        "adj_r2": 1 - (sse / dfr) / (tss_c / (n - 1)),
        "r2_uncentered": 1 - sse / tss_u,
        "adj_r2_uncentered": 1 - (sse / dfr) / (tss_u / n),
        "adj_r2_legacy": 1 - (1 - r2) * (n - 1) / (n - k - 1),
        "rmse": math.sqrt(sse / n),
        "mae": float(np.abs(resid).mean()),
        "sse": sse,
    }
    sigma2 = sse / n
    ll = -0.5 * n * (math.log(2 * math.pi) + math.log(sigma2) + 1)
    K = k + 1
    out["log_lik"] = ll
    out["aic"] = -2 * ll + 2 * K
    out["aicc"] = out["aic"] + 2 * K * (K + 1) / (n - K - 1)
    out["bic"] = -2 * ll + K * math.log(n)
    return out


def brute_force_best(X, y, universe, allowed_idx, max_pillows, *,
                     fit_intercept, add_zero_aso):
    """Search every combination drawn from `allowed_idx` and return the best by adjusted
    R2 under the agreed tie-break. Naive on purpose: no table, no bitmask, no shared
    helpers."""
    n = X.shape[0]
    tss_c = float(((y - y.mean()) ** 2).sum())
    best = None
    n_scored = 0
    for k in range(1, max_pillows + 1):
        for cols in combinations(sorted(allowed_idx), k):
            Xs = X[:, list(cols)]
            if add_zero_aso:
                Xf = np.vstack([Xs, np.zeros((1, k))])
                yf = np.append(y, 0.0)
            else:
                Xf, yf = Xs, y
            lm = linear_model.LinearRegression(fit_intercept=fit_intercept).fit(Xf, yf)
            sse = float(((y - lm.predict(Xs)) ** 2).sum())
            adj = 1 - (sse / (n - k)) / (tss_c / (n - 1))
            key = ",".join(universe[i] for i in cols)
            n_scored += 1
            # tie-break: higher adj_r2, then fewer predictors, then lexicographic key
            cand = (-adj, k, key)
            if best is None or cand < best[0]:
                best = (cand, {"pillow_key": key, "adj_r2": adj,
                               "n_predictors": k, "rmse": math.sqrt(sse / n)})
    return best[1], n_scored


# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("basin")
    ap.add_argument("--band", default="total")
    ap.add_argument("--mode", default="season", choices=["season"])
    ap.add_argument("--library-id", default="all_years")
    ap.add_argument("--qa-file", default="pillow_wy_1980_2025_qa6.nc")
    ap.add_argument("--max-pillows", type=int, default=cf.MAX_PILLOWS)
    ap.add_argument("--config-dir", default="/home/rossamower/work/aso/configs/")
    ap.add_argument("--n-refit-checks", type=int, default=40)
    ap.add_argument("--no-brute-force", action="store_true",
                   help="skip the full-availability brute-force recheck (68k extra fits)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    t_all = time.perf_counter()

    # ---------------------------------------------------------------- frame
    t0 = time.perf_counter()
    frame = cf.build_canonical_frame(
        args.basin, elev_band=args.band, mode=args.mode, obs_qa_file=args.qa_file,
        library_id=args.library_id, config_dir=args.config_dir, verbose=False)
    t_frame = time.perf_counter() - t0
    fm = frame.manifest
    X, y, universe = frame.X, frame.y, list(frame.universe)
    fit_intercept = bool(fm["fit_intercept"])
    add_zero_aso = bool(fm["add_zero_aso"])
    print(f"CANONICAL FRAME  {fm['frame_id']}")
    print(f"  X={X.shape} y={y.shape}  loaded/built in {t_frame:.1f}s")

    # -------------------------------------------------------------- library
    t0 = time.perf_counter()
    table = lf.build_model_library(frame, max_pillows=args.max_pillows)
    t_fit = time.perf_counter() - t0
    manifest = ls.build_library_manifest(table, fm, max_pillows=args.max_pillows,
                                         build_seconds=t_fit)
    lib = ModelLibrary(table=table, manifest=manifest)

    print(f"\nMODEL LIBRARY")
    print(f"  models        : {len(table):,}")
    print(f"  per k         : {manifest['models_per_k']}")
    print(f"  fit time      : {t_fit:.1f}s  ({t_fit / len(table) * 1e6:.0f} us/model)")
    print(f"  cond number   : max={manifest['cond_number_max']:.3g} "
          f"p99={manifest['cond_number_p99']:.3g}")

    # ----------------------------------------------------------- validation
    print(f"\nVALIDATION")
    checks = []

    expected = lf.expected_n_models(len(universe), args.max_pillows)
    checks.append((f"exactly {expected:,} models produced",
                   len(table) == expected, f"{len(table):,}"))
    checks.append(("no duplicate pillow_key",
                   not table["pillow_key"].duplicated().any(),
                   f"{table['pillow_key'].nunique():,} unique"))
    checks.append(("pillow_key set == canonical enumeration",
                   set(table["pillow_key"]) == {
                       ",".join(universe[i] for i in c)
                       for k in range(1, args.max_pillows + 1)
                       for c in combinations(range(len(universe)), k)},
                   "exact set match"))
    checks.append(("every pillow_key sorted (canonical order)",
                   all(list(k.split(",")) == sorted(k.split(","))
                       for k in table["pillow_key"]),
                   "all canonical"))
    checks.append(("intercept == 0 everywhere (fit_intercept=False)",
                   bool((table["intercept"] == 0.0).all()),
                   f"max |intercept| = {table['intercept'].abs().max():g}"))

    # --- stored coefficients vs independent sklearn refits ------------------
    rng = np.random.default_rng(0)
    ranked = lf.rank_frame(table, "adj_r2").reset_index(drop=True)
    sample_idx = set()
    for k in range(1, args.max_pillows + 1):                 # stratify by k
        pool = ranked.index[ranked["n_predictors"] == k]
        sample_idx.update(rng.choice(pool, size=min(4, len(pool)), replace=False).tolist())
    for q in range(10):                                      # stratify by rank decile
        lo = int(q * len(ranked) / 10)
        hi = int((q + 1) * len(ranked) / 10) - 1
        sample_idx.update(rng.integers(lo, max(hi, lo + 1),
                                       size=min(2, hi - lo + 1)).tolist())
    sample_idx.update([0, 1, 2, len(ranked) - 1])             # rank 1..3 and last
    # No .head() here: sample_idx is sorted by RANK, so truncating it would silently drop
    # the worst-ranked models -- including the deliberately-included last one -- and leave
    # the check biased toward the top of the table.
    sample = ranked.loc[sorted(sample_idx)]

    metric_cols = ["r2", "adj_r2", "r2_uncentered", "adj_r2_uncentered",
                   "adj_r2_legacy", "rmse", "mae", "sse", "log_lik",
                   "aic", "aicc", "bic"]
    worst_coef, worst_metric, worst_metric_name = 0.0, 0.0, ""
    for _, row in sample.iterrows():
        pillows = model_pillows(row)
        cols = [universe.index(p) for p in pillows]
        ref = reference_fit(X, y, cols, fit_intercept=fit_intercept,
                            add_zero_aso=add_zero_aso)
        stored = [row[f"coef_{i + 1}"] for i in range(len(pillows))]
        worst_coef = max(worst_coef, max(abs(a - b) for a, b in
                                         zip(stored, ref["coefs"])))
        for mc in metric_cols:
            d = abs(float(row[mc]) - ref[mc])
            rel = d / max(abs(ref[mc]), 1e-12)
            if rel > worst_metric:
                worst_metric, worst_metric_name = rel, mc
    checks.append((f"stored coefficients == independent sklearn refit "
                   f"({len(sample)} models, k=1..{args.max_pillows}, "
                   f"incl. rank 1-3 and rank {len(ranked):,})",
                   worst_coef < 1e-9, f"max |diff| = {worst_coef:.3e}"))
    checks.append((f"all 12 stored metrics == independently recomputed",
                   worst_metric < 1e-10,
                   f"max relative diff = {worst_metric:.3e} (worst: {worst_metric_name})"))

    # --- metric formulas vs an INDEPENDENT implementation (statsmodels) ----
    # reference_fit above re-derives the metrics, but from the same formulas as fit.py, so
    # their agreement only proves self-consistency. statsmodels is a genuinely separate
    # implementation of no-intercept OLS and is what pins the definitions down. It is fit
    # on the 35 REAL rows, so agreement also independently confirms the synthetic zero row
    # is inert under fit_intercept=False.
    try:
        import statsmodels.api as sm
        sm_worst = {}
        for _, row in sample.head(12).iterrows():
            pillows = model_pillows(row)
            cols = [universe.index(p) for p in pillows]
            k = len(cols)
            res = sm.OLS(y, X[:, cols]).fit()          # no constant added
            pairs = {
                "coef": (np.abs(np.asarray(
                    [row[f"coef_{i+1}"] for i in range(k)]) - res.params).max(), 0.0),
                "r2_uncentered": (row["r2_uncentered"], res.rsquared),
                "adj_r2_uncentered": (row["adj_r2_uncentered"], res.rsquared_adj),
                "log_lik": (row["log_lik"], res.llf),
                "aic (ours - 2, statsmodels omits sigma^2)": (row["aic"] - 2.0, res.aic),
                "bic (ours - log n, same reason)":
                    (row["bic"] - math.log(len(y)), res.bic),
                "df_resid": (float(row["df_resid"]), float(res.df_resid)),
            }
            for nm, (a, b) in pairs.items():
                d = abs(a - b) / max(abs(b), 1.0)
                sm_worst[nm] = max(sm_worst.get(nm, 0.0), d)
        worst_sm = max(sm_worst.values())
        checks.append(("metric formulas verified against statsmodels OLS (no constant)",
                       worst_sm < 1e-9,
                       "max rel diff " + ", ".join(
                           f"{k2}={v:.1e}" for k2, v in sorted(sm_worst.items()))))
    except ImportError:
        print("  (statsmodels unavailable -- formula cross-check skipped)")

    # --- adj_r2 / adj_r2_uncentered rank equivalence -----------------------
    r_c = lf.rank_frame(table, "adj_r2")["pillow_key"].tolist()
    r_u = lf.rank_frame(table, "adj_r2_uncentered")["pillow_key"].tolist()
    checks.append(("adj_r2 and adj_r2_uncentered rank identically "
                   "(convention does not change selection)",
                   r_c == r_u, "orderings identical" if r_c == r_u else "DIFFER"))
    r_l = lf.rank_frame(table, "adj_r2_legacy")["pillow_key"].tolist()
    checks.append(("adj_r2_legacy rank-1 recorded (may differ; not the default)",
                   True, f"legacy rank 1 = {r_l[0]}  |  default rank 1 = {r_c[0]}"))

    # --- table rank 1 vs independent brute force --------------------------
    all_idx = list(range(len(universe)))
    if not args.no_brute_force:
        t0 = time.perf_counter()
        bf, n_scored = brute_force_best(X, y, universe, all_idx, args.max_pillows,
                                        fit_intercept=fit_intercept,
                                        add_zero_aso=add_zero_aso)
        t_bf = time.perf_counter() - t0
        tbl_best = lib.select_best_model(universe, score_rule="adj_r2")
        ok = (tbl_best["pillow_key"] == bf["pillow_key"]
              and abs(tbl_best["adj_r2"] - bf["adj_r2"]) < 1e-12)
        checks.append((f"table rank 1 == brute force over all {n_scored:,} combinations",
                       ok, f"{bf['pillow_key']}  adj_r2={bf['adj_r2']:.10f} "
                           f"({t_bf:.1f}s brute force)"))
    else:
        print("  (full-availability brute force skipped)")

    # --- restricted availability: the station-loss query shape ------------
    scenarios = []
    r1_pillows = model_pillows(lf.rank_frame(table, "adj_r2").iloc[0])
    for p in r1_pillows:                                     # drop each winner pillow
        scenarios.append((f"minus {p}", [q for q in universe if q != p]))
    rng2 = np.random.default_rng(7)
    for i in range(4):                                       # random availability sets
        m = rng2.choice(universe, size=rng2.integers(5, 20), replace=False).tolist()
        scenarios.append((f"random#{i + 1} (n={len(m)})", sorted(m)))
    scenarios.append(("single pillow LLE", ["LLE"]))
    scenarios.append(("none available", []))
    scenarios.append(("includes an unknown pillow", sorted(universe[:8]) + ["ZZZ"]))

    print("\n  restricted-availability: table vs brute force")
    all_restricted_ok = True
    for name, avail in scenarios:
        best = lib.select_best_model(avail, score_rule="adj_r2")
        allowed = [universe.index(p) for p in avail if p in universe]
        if not allowed:
            ok = best is None
            all_restricted_ok &= ok
            print(f"    [{'PASS' if ok else 'FAIL'}] {name:28s} -> "
                  f"{'None (no eligible model)' if best is None else best['pillow_key']}")
            continue
        bf, n_scored = brute_force_best(X, y, universe, allowed, args.max_pillows,
                                        fit_intercept=fit_intercept,
                                        add_zero_aso=add_zero_aso)
        ok = (best is not None and best["pillow_key"] == bf["pillow_key"]
              and abs(best["adj_r2"] - bf["adj_r2"]) < 1e-12)
        all_restricted_ok &= ok
        print(f"    [{'PASS' if ok else 'FAIL'}] {name:28s} -> {bf['pillow_key']:36s} "
              f"adj_r2={bf['adj_r2']:.6f}  ({n_scored:,} combos)")
    checks.append((f"table rank 1 == brute force under {len(scenarios)} "
                   f"restricted availability sets", all_restricted_ok,
                   "includes every single-pillow drop from the rank-1 model"))

    # --- eligibility is a pure filter -------------------------------------
    sub = sorted(universe[:6])
    elig = lib.eligible_models(sub)
    checks.append(("eligible_models returns exactly the sub-universe combinations",
                   len(elig) == lf.expected_n_models(len(sub), args.max_pillows),
                   f"{len(elig):,} == C({len(sub)},1..{args.max_pillows})"))
    checks.append(("no eligible model requires an unavailable pillow",
                   all(set(k.split(",")) <= set(sub) for k in elig["pillow_key"]),
                   f"all {len(elig):,} within {sub}"))

    # --- prediction from stored coefficients ------------------------------
    best = lib.select_best_model(universe, score_rule="adj_r2")
    vals = {p: float(v) for p, v in zip(universe, X[-1])}     # last real flight
    pred = lib.predict(best, vals)
    cols = [universe.index(p) for p in model_pillows(best)]
    direct = linear_model.LinearRegression(fit_intercept=fit_intercept).fit(
        np.vstack([X[:, cols], np.zeros((1, len(cols)))]), np.append(y, 0.0)
    ).predict(X[-1:, cols])[0]
    checks.append(("predict_from_lookup == direct sklearn predict",
                   abs(pred - max(direct, 0.0)) < 1e-9,
                   f"lookup={pred:.6f} direct={direct:.6f} "
                   f"(truth {y[-1]:.1f} mm on {frame.dates[-1].date()})"))

    # --- imputation-burden filter works without a rebuild -----------------
    strict = lib.select_best_model(universe, score_rule="adj_r2",
                                  max_pillow_frac_imputed=0.10)
    checks.append(("max_pillow_frac_imputed filter changes selection without refit",
                   strict is not None,
                   f"unfiltered={best['pillow_key']} "
                   f"(burden {best['max_pillow_frac_imputed']:.3f})  ->  "
                   f"<=0.10: {strict['pillow_key']} "
                   f"(burden {strict['max_pillow_frac_imputed']:.3f}, "
                   f"adj_r2 {strict['adj_r2']:.4f})"))

    # --- determinism of the ranking ---------------------------------------
    shuffled = table.sample(frac=1.0, random_state=42)
    checks.append(("ranking is order-independent (tie-break is total)",
                   lf.rank_frame(shuffled, "adj_r2")["pillow_key"].tolist() == r_c,
                   "shuffled input -> identical ordering"))

    # ------------------------------------------------------------ artifacts
    if not args.dry_run:
        written = ls.save_model_library(table, manifest)
        rt = ls.load_model_library(written["manifest"])
        cmp_cols = ["pillow_key", "pillow_mask", "n_predictors", "intercept",
                    "adj_r2", "rmse", "mae", "aicc", "bic"] + \
                   [f"coef_{i}" for i in range(1, 6)]
        checks.append(("parquet round-trip is exact",
                       rt.table[cmp_cols].equals(table[cmp_cols]),
                       f"{len(rt.table):,} rows, {len(table.columns)} columns"))

    ok = True
    for name, passed, detail in checks:
        ok &= bool(passed)
        print(f"  [{'PASS' if passed else 'FAIL'}] {name}")
        print(f"         {detail}")

    # --------------------------------------------------------------- top 10
    print("\nTOP 10 BY ADJUSTED R-SQUARED (in-sample; selection statistic, NOT skill)")
    top = lf.rank_frame(table, "adj_r2").head(10)
    hdr = (f"  {'#':>2} {'k':>2} {'pillows':34s} {'adj_r2':>8s} {'r2':>8s} "
           f"{'RMSE':>7s} {'MAE':>7s} {'AICc':>8s} {'imp_max':>7s} {'imp_mean':>8s}")
    print(hdr); print("  " + "-" * (len(hdr) - 2))
    for i, (_, r) in enumerate(top.iterrows(), start=1):
        print(f"  {i:>2} {r['n_predictors']:>2} {r['pillow_key']:34s} "
              f"{r['adj_r2']:8.5f} {r['r2']:8.5f} {r['rmse']:7.2f} {r['mae']:7.2f} "
              f"{r['aicc']:8.2f} {r['max_pillow_frac_imputed']:7.3f} "
              f"{r['mean_pillow_frac_imputed']:8.3f}")

    if not args.dry_run:
        print("\nARTIFACTS")
        for k, v in written.items():
            print(f"  {k:12s} {v}  ({Path(v).stat().st_size:,} bytes)")

    print(f"\nRUNTIME  frame {t_frame:.1f}s + fit {t_fit:.1f}s "
          f"+ validation = {time.perf_counter() - t_all:.1f}s total")
    print("  " + ("ALL CHECKS PASSED" if ok else "*** CHECKS FAILED ***"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
