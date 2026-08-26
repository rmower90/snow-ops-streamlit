"""
build_all_bands.py

Generalise the step-1 canonical frame and step-2 model library from the `total` band to
every ASO elevation band, for one basin / season / predict-NaNs / full history.

This is a loop over the existing code paths -- build_canonical_frame, build_model_library,
save_canonical_frame, save_model_library -- and deliberately contains no modelling logic of
its own. Every convention is inherited unchanged: fixed historical imputation universe,
frac_missing < 0.50, MAX_PILLOWS = 5, fit_intercept = False, add_zero_aso = True,
statistics on the real ASO flights only, default ranking by in-sample adjusted R2.

One artifact pair per band:
    canonical_frames/{basin}_season_{band}_all_years_{qa}_{hash}.parquet   + manifest
    libraries/{basin}_season_{band}_all_years_{qa}_{hash}_models.parquet   + manifest

A structural expectation worth stating up front, because it is also a cross-check: the
pillow block of df_sum_total is band-independent -- combine_aso_insitu merges the pillow
series onto flight dates and only the ASO column varies with band. So the retained
universe, the imputed-cell mask and the model count should be IDENTICAL across bands, and
only y (hence every coefficient and metric) should differ. The script asserts this rather
than assuming it.

Per-band imputation is therefore recomputed 8 times over identical inputs, which costs
about 25s each. That is accepted rather than optimised away: elev_band is part of frame_id,
so each band's artifact stays self-contained and independently reproducible, and the
identical results across bands become evidence instead of an assumption.

Usage (conda env: bor):
    python build_all_bands.py FRIANT
    python build_all_bands.py FRIANT --bands total,8000-9000
    python build_all_bands.py FRIANT --dry-run
"""

import argparse
import sys
import time
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

sys.path.insert(1, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(1, str(Path(__file__).resolve().parent))

from snow_ops.mlr.lookup import frame as cf          # noqa: E402
from snow_ops.mlr.lookup import fit as lf            # noqa: E402
from snow_ops.mlr.lookup import store as ls          # noqa: E402
from snow_ops.mlr.lookup.query import ModelLibrary, model_pillows  # noqa: E402
from build_model_library import reference_fit        # noqa: E402  (independent refit)

import metadata  # noqa: E402


def aso_band_labels(basin: str, config_dir: str) -> list[str]:
    cfg = metadata.load_yaml(Path(config_dir) / "regions" / f"{basin}.yaml")
    ds = xr.open_dataset(cfg["data_filepaths"]["aso_temporal"], engine="netcdf4")
    return [str(v) for v in ds["elev"].values]


def validate_band(frame, table, lib, *, n_refit: int, rng) -> list[tuple]:
    """The five checks asked for, per band."""
    universe = list(frame.universe)
    checks = []

    # 1. complete canonical support
    problems = cf.validate_canonical_frame(frame)
    checks.append(("canonical frame invariants", not problems,
                   "; ".join(problems) or "all pass"))
    checks.append(("complete support: X and y all finite",
                   bool(np.isfinite(frame.X).all() and np.isfinite(frame.y).all()),
                   f"{int((~np.isfinite(frame.X)).sum())} non-finite in X, "
                   f"{int((~np.isfinite(frame.y)).sum())} in y"))

    # 2. expected model count from THIS band's retained universe
    expected = lf.expected_n_models(len(universe), frame.manifest["max_pillows"])
    checks.append((f"model count == C({len(universe)}, 1..5) = {expected:,}",
                   len(table) == expected, f"{len(table):,}"))

    # 3. no duplicate combinations
    checks.append(("no duplicate pillow_key; key set == canonical enumeration",
                   (not table["pillow_key"].duplicated().any())
                   and set(table["pillow_key"]) == {
                       ",".join(universe[i] for i in c)
                       for k in range(1, frame.manifest["max_pillows"] + 1)
                       for c in combinations(range(len(universe)), k)},
                   f"{table['pillow_key'].nunique():,} unique, exact set match"))

    # 5. stored models vs direct sklearn (4. round-trip happens after saving)
    ranked = lf.rank_frame(table, "adj_r2").reset_index(drop=True)
    idx = set()
    for k in range(1, frame.manifest["max_pillows"] + 1):
        pool = ranked.index[ranked["n_predictors"] == k]
        idx.update(rng.choice(pool, size=min(2, len(pool)), replace=False).tolist())
    idx.update([0, 1, len(ranked) - 1])
    sample = ranked.loc[sorted(idx)[:n_refit]]

    metrics = ["r2", "adj_r2", "rmse", "mae", "aicc", "bic", "sse", "log_lik"]
    worst_c = worst_m = 0.0
    for _, row in sample.iterrows():
        cols = [universe.index(p) for p in model_pillows(row)]
        ref = reference_fit(frame.X, frame.y, cols,
                            fit_intercept=frame.manifest["fit_intercept"],
                            add_zero_aso=frame.manifest["add_zero_aso"])
        stored = [row[f"coef_{i + 1}"] for i in range(len(cols))]
        worst_c = max(worst_c, max(abs(a - b) for a, b in zip(stored, ref["coefs"])))
        for mc in metrics:
            worst_m = max(worst_m, abs(float(row[mc]) - ref[mc])
                          / max(abs(ref[mc]), 1e-12))
    checks.append((f"{len(sample)} stored models == direct sklearn refit",
                   worst_c < 1e-9 and worst_m < 1e-10,
                   f"max |coef diff| = {worst_c:.3e}, max rel metric diff = {worst_m:.3e}"))
    return checks


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("basin")
    ap.add_argument("--mode", default="season", choices=["season"])
    ap.add_argument("--library-id", default="all_years")
    ap.add_argument("--qa-file", default="pillow_wy_1980_2025_qa6.nc")
    ap.add_argument("--bands", default="", help="comma-separated subset; default = all")
    ap.add_argument("--config-dir", default="/home/rossamower/work/aso/configs/")
    ap.add_argument("--n-refit", type=int, default=12)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    bands = ([b.strip() for b in args.bands.split(",") if b.strip()]
             or aso_band_labels(args.basin, args.config_dir))
    rng = np.random.default_rng(0)
    t_all = time.perf_counter()

    print(f"ALL-BAND BUILD  basin={args.basin} mode={args.mode} "
          f"library={args.library_id} qa={Path(args.qa_file).stem.split('_')[-1]}")
    print(f"  bands ({len(bands)}): {', '.join(bands)}\n")

    summary, all_ok, reference = [], True, None
    for band in bands:
        t0 = time.perf_counter()
        print(f"--- {band} " + "-" * (68 - len(band)))
        frame = cf.build_canonical_frame(
            args.basin, elev_band=band, mode=args.mode, obs_qa_file=args.qa_file,
            library_id=args.library_id, config_dir=args.config_dir, verbose=False)
        t_frame = time.perf_counter() - t0
        fm = frame.manifest

        t0 = time.perf_counter()
        table = lf.build_model_library(frame, verbose=False)
        t_fit = time.perf_counter() - t0
        lib_manifest = ls.build_library_manifest(
            table, fm, max_pillows=fm["max_pillows"], build_seconds=t_fit)
        lib = ModelLibrary(table=table, manifest=lib_manifest)

        checks = validate_band(frame, table, lib, n_refit=args.n_refit, rng=rng)

        # cross-band structural check: the pillow block does not depend on the band
        if reference is None:
            reference = {"universe": frame.universe, "imputed": frame.imputed.copy(),
                         "dates": frame.dates, "band": band}
        else:
            same = (frame.universe == reference["universe"]
                    and np.array_equal(frame.imputed, reference["imputed"])
                    and frame.dates.equals(reference["dates"]))
            checks.append((f"pillow block identical to band '{reference['band']}' "
                           f"(only y should differ)", same,
                           "universe, imputed mask and flight dates all match"
                           if same else "DIFFER"))

        if not args.dry_run:
            w_frame = cf.save_canonical_frame(frame)
            w_lib = ls.save_model_library(table, lib_manifest)
            rt_f = cf.load_canonical_frame(w_frame["manifest"])
            rt_l = ls.load_model_library(w_lib["manifest"])
            cmp_cols = (["pillow_key", "pillow_mask", "n_predictors", "intercept",
                         "adj_r2", "rmse", "mae", "aicc", "bic"]
                        + [f"coef_{i}" for i in range(1, 6)])
            checks.append(("parquet/manifest round-trip exact (frame + library)",
                           np.array_equal(rt_f.X, frame.X)
                           and np.array_equal(rt_f.y, frame.y)
                           and rt_l.table[cmp_cols].equals(table[cmp_cols]),
                           f"frame {rt_f.X.shape}, library {len(rt_l.table):,} rows"))

        band_ok = all(p for _, p, _ in checks)
        all_ok &= band_ok
        for name, passed, detail in checks:
            print(f"  [{'PASS' if passed else 'FAIL'}] {name}")
            if not passed:
                print(f"         {detail}")

        best = lf.rank_frame(table, "adj_r2").iloc[0]
        summary.append({
            "band": band, "elev_idx": fm["elev_idx"],
            "n_flights": fm["stats_n_obs"], "n_retained": fm["n_retained"],
            "n_models": len(table),
            "rank1_pillows": best["pillow_key"],
            "adj_r2": best["adj_r2"], "rmse": best["rmse"],
            "imp_max": best["max_pillow_frac_imputed"],
            "imp_mean": best["mean_pillow_frac_imputed"],
            "frame_s": t_frame, "fit_s": t_fit,
        })
        print(f"  built in {t_frame:.0f}s frame + {t_fit:.0f}s fit   "
              f"rank 1: {best['pillow_key']}  adj_r2={best['adj_r2']:.5f}\n")

    # ------------------------------------------------------------- summary
    s = pd.DataFrame(summary)
    print("=" * 112)
    print(f"{args.basin} / {args.mode} / predict NaNs / full history "
          f"-- one canonical frame + one model library per band")
    print("=" * 112)
    hdr = (f"{'band':13s} {'flights':>7s} {'pillows':>7s} {'models':>7s} "
           f"{'rank-1 combination':34s} {'adj_r2':>8s} {'RMSE':>7s} "
           f"{'imp_max':>7s} {'imp_mean':>8s}")
    print(hdr); print("-" * len(hdr))
    for r in summary:
        print(f"{r['band']:13s} {r['n_flights']:>7d} {r['n_retained']:>7d} "
              f"{r['n_models']:>7,d} {r['rank1_pillows']:34s} {r['adj_r2']:8.5f} "
              f"{r['rmse']:7.2f} {r['imp_max']:7.3f} {r['imp_mean']:8.3f}")

    print(f"\nRUNTIME  {time.perf_counter() - t_all:.0f}s total "
          f"({s['frame_s'].sum():.0f}s frames + {s['fit_s'].sum():.0f}s fitting, "
          f"{len(bands)} bands, {s['n_models'].sum():,} models)")
    print("  " + ("ALL BANDS PASSED" if all_ok else "*** SOME BANDS FAILED ***"))
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
