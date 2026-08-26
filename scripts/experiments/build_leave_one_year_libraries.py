"""
build_leave_one_year_libraries.py

Historical-year sensitivity component: for FRIANT / season / predict-NaNs, build one
canonical frame + one pretrained model library per (elevation band, dropped water year),
excluding that water year's ASO flights entirely from the historical training record.

This is a loop over the SAME build_canonical_frame / build_model_library / save_* code
paths the full-history libraries already use -- no new modelling logic.
`excluded_years` was already plumbed through build_canonical_frame in step 1 for exactly
this purpose. Each dropped-year build independently:

  - excludes that water year's ASO flight rows BEFORE the frac_missing < 0.50 eligibility
    rule is evaluated, so the retained universe is recomputed on the REDUCED frame, not
    inherited from the full-history library
  - imputes missing pillow values using the SAME donor search over the full >=2013 daily
    pillow record -- imputation donors are not restricted by excluded_years, because that
    is a different, denser dataset (daily pillow observations) than the ASO flight rows
    being excluded
  - fits all C(n_retained, 1..5) combinations under the same conventions as the
    full-history libraries: fit_intercept=False, add_zero_aso=True, statistics on the real
    ASO flights only, default ranking by in-sample adjusted R2

Every build already self-validates for free: build_canonical_frame raises if any of its
invariants fail, and build_model_library raises if the model count or pillow_key set is
wrong. This script additionally re-checks a SMALL SAMPLE of the resulting libraries against
independent sklearn refits and an independent brute-force search (see --n-sample-checks).

Usage (conda env: bor):
    python build_leave_one_year_libraries.py FRIANT
    python build_leave_one_year_libraries.py FRIANT --bands total,8000-9000
    python build_leave_one_year_libraries.py FRIANT --dry-run
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

sys.path.insert(1, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(1, str(Path(__file__).resolve().parent))

from snow_ops.mlr.lookup import frame as cf                       # noqa: E402
from snow_ops.mlr.lookup import fit as lf                          # noqa: E402
from snow_ops.mlr.lookup import store as ls                        # noqa: E402
from snow_ops.mlr.lookup.build import build_and_save_one           # noqa: E402
from snow_ops.mlr.lookup.frame import aso_band_labels, full_history_water_years  # noqa: E402
from snow_ops.mlr.lookup.query import ModelLibrary, model_pillows  # noqa: E402
from build_model_library import reference_fit, brute_force_best    # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("basin")
    ap.add_argument("--mode", default="season", choices=["season"])
    ap.add_argument("--qa-file", default="pillow_wy_1980_2025_qa6.nc")
    ap.add_argument("--bands", default="", help="comma-separated subset; default = all")
    ap.add_argument("--config-dir", default="/home/rossamower/work/aso/configs/")
    ap.add_argument("--n-sample-checks", type=int, default=6,
                    help="how many (band, dropped WY) libraries to deep-check "
                         "(sklearn refit + brute-force query)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    bands = ([b.strip() for b in args.bands.split(",") if b.strip()]
             or aso_band_labels(args.basin, args.config_dir))
    water_years = full_history_water_years(args.basin, args.config_dir)
    rng = np.random.default_rng(11)

    print(f"LEAVE-ONE-WATER-YEAR LIBRARIES  basin={args.basin} mode={args.mode}")
    print(f"  bands ({len(bands)}): {', '.join(bands)}")
    print(f"  water years ({len(water_years)}): {water_years}")
    print(f"  total builds: {len(bands)} x {len(water_years)} = "
          f"{len(bands) * len(water_years)}\n", flush=True)

    t_all = time.perf_counter()
    records = []
    built = {}   # (band, wy) -> (frame, table, lib_manifest)

    for band in bands:
        for wy in water_years:
            library_id = f"drop_wy{wy}"
            result = build_and_save_one(
                args.basin, elev_band=band, mode=args.mode, obs_qa_file=args.qa_file,
                excluded_years=[wy], library_id=library_id, config_dir=args.config_dir,
                save=not args.dry_run, verbose=False)
            frame, table, lib_manifest = (
                result["frame"], result["table"], result["lib_manifest"])
            t_frame, t_fit = result["t_frame"], result["t_fit"]

            fm = frame.manifest
            assert fm["excluded_years"] == [wy], \
                f"manifest excluded_years {fm['excluded_years']} != [{wy}]"

            best = lf.rank_frame(table, "adj_r2").iloc[0]
            records.append(dict(
                band=band, dropped_wy=wy, n_flights=fm["stats_n_obs"],
                n_retained=fm["n_retained"], n_models=len(table),
                rank1_pillows=best["pillow_key"], adj_r2=best["adj_r2"],
                rmse=best["rmse"], t_frame=t_frame, t_fit=t_fit))
            built[(band, wy)] = (frame, table, lib_manifest)

            print(f"  {band:13s} drop WY{wy}  n_flights={fm['stats_n_obs']:2d} "
                  f"n_retained={fm['n_retained']:2d} n_models={len(table):6,d}  "
                  f"rank1={best['pillow_key']:30s} adj_r2={best['adj_r2']:.4f}  "
                  f"({t_frame:4.0f}s+{t_fit:4.0f}s)", flush=True)

    t_build = time.perf_counter() - t_all

    # ------------------------------------------------------ small sample validation
    print("\nSAMPLE VALIDATION  (independent sklearn refit + independent brute force)")
    keys = list(built.keys())
    n_check = min(args.n_sample_checks, len(keys))
    sample_keys = [keys[i] for i in
                   rng.choice(len(keys), size=n_check, replace=False)]

    checks = []
    for band, wy in sample_keys:
        frame, table, lib_manifest = built[(band, wy)]
        universe = list(frame.universe)
        fit_intercept = frame.manifest["fit_intercept"]
        add_zero_aso = frame.manifest["add_zero_aso"]

        checks.append((f"{band}/drop_wy{wy}: manifest self-identifies excluded_years",
                       frame.manifest["excluded_years"] == [wy],
                       str(frame.manifest["excluded_years"])))

        ranked = lf.rank_frame(table, "adj_r2").reset_index(drop=True)
        idxs = sorted(set([0, len(ranked) - 1]
                          + rng.integers(0, len(ranked), size=3).tolist()))
        worst = 0.0
        for i in idxs:
            row = ranked.iloc[i]
            cols = [universe.index(p) for p in model_pillows(row)]
            ref = reference_fit(frame.X, frame.y, cols,
                                fit_intercept=fit_intercept, add_zero_aso=add_zero_aso)
            stored = [row[f"coef_{j + 1}"] for j in range(len(cols))]
            worst = max(worst, max(abs(a - b) for a, b in zip(stored, ref["coefs"])))
        checks.append((f"{band}/drop_wy{wy}: {len(idxs)} stored models == "
                       f"direct sklearn refit", worst < 1e-9, f"max|diff|={worst:.3e}"))

        lib = ModelLibrary(table=table, manifest=lib_manifest)
        best_tbl = lib.select_best_model(universe, score_rule="adj_r2")
        bf, n_scored = brute_force_best(
            frame.X, frame.y, universe, list(range(len(universe))),
            frame.manifest["max_pillows"],
            fit_intercept=fit_intercept, add_zero_aso=add_zero_aso)
        ok = (best_tbl is not None and best_tbl["pillow_key"] == bf["pillow_key"]
              and abs(best_tbl["adj_r2"] - bf["adj_r2"]) < 1e-9)
        checks.append((f"{band}/drop_wy{wy}: select_best_model == brute force "
                       f"over this library's own {n_scored:,} combinations", ok,
                       f"table={best_tbl['pillow_key']}  brute_force={bf['pillow_key']}"))

    ok_all = all(p for _, p, _ in checks)
    for name, passed, detail in checks:
        print(f"  [{'PASS' if passed else 'FAIL'}] {name}")
        print(f"         {detail}")

    # --------------------------------------------------------------------- summary
    s = pd.DataFrame(records)
    print("\n" + "=" * 100)
    print(f"{args.basin} / {args.mode} / predict NaNs -- leave-one-water-year libraries")
    print("=" * 100)
    pivot_models = s.pivot(index="band", columns="dropped_wy", values="n_models")
    pivot_retained = s.pivot(index="band", columns="dropped_wy", values="n_retained")
    print("\nmodels per (band, dropped WY):")
    print(pivot_models.to_string())
    print("\nretained pillows per (band, dropped WY):")
    print(pivot_retained.to_string())

    print(f"\nRUNTIME  {t_build:.0f}s build "
          f"({s['t_frame'].sum():.0f}s frames + {s['t_fit'].sum():.0f}s fitting, "
          f"{len(records)} libraries, {s['n_models'].sum():,} models total) "
          f"+ validation, {time.perf_counter() - t_all:.0f}s grand total")
    print("  " + ("ALL SAMPLE CHECKS PASSED" if ok_all else "*** SAMPLE CHECKS FAILED ***"))

    if not args.dry_run:
        summary_csv = Path(f"/home/rossamower/work/aso/data/mlr_prediction/{args.basin}/"
                           "model_library/leave_one_year_summary.csv")
        s.to_csv(summary_csv, index=False)
        print(f"  summary written to {summary_csv}")

    return 0 if ok_all else 1


if __name__ == "__main__":
    raise SystemExit(main())
