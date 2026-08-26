"""
build_canonical_frame.py

Step 1 of the pretrained MLR lookup work: build and persist the canonical predict-NaNs
historical training matrix for one (basin, elevation band, mode, library).

This does NOT enumerate or fit any pillow combinations. It produces the single completed
predictor matrix that every combination will later be fit against, plus a manifest that
records everything needed to reproduce it.

What makes the matrix "canonical": today's QA-approved pillow set has no influence on it.
The existing production pathway passes `all_pils_QA` into
`lm_model.impute_pillow_prediction`, which makes the retained universe, the imputed
values, the surviving flight count and the on-disk cache all day-dependent. Here the same
function is handed the fixed historical candidate list instead, so the result is a pure
function of (ASO record, pillow QA file, missingness threshold).

Usage (conda env: bor):
    python build_canonical_frame.py FRIANT
    python build_canonical_frame.py FRIANT --band total --mode season
    python build_canonical_frame.py FRIANT --rebuild-imputation
    python build_canonical_frame.py FRIANT --check-determinism
    python build_canonical_frame.py FRIANT --dry-run          # report, write nothing
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(1, str(Path(__file__).resolve().parents[2] / "src"))

from snow_ops.mlr.lookup import frame as cf  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("basin")
    ap.add_argument("--band", default="total")
    ap.add_argument("--mode", default="season", choices=["season"])
    ap.add_argument("--library-id", default="all_years")
    ap.add_argument("--qa-file", default="pillow_wy_1980_2025_qa6.nc")
    ap.add_argument("--obs-threshold", type=float, default=cf.DEFAULT_OBS_THRESHOLD)
    ap.add_argument("--exclude-years", default="",
                    help="comma-separated water years to omit (leave-one-year libraries)")
    ap.add_argument("--config-dir", default="/home/rossamower/work/aso/configs/")
    ap.add_argument("--rebuild-imputation", action="store_true",
                    help="delete the frame's own imputation cache and recompute")
    ap.add_argument("--check-determinism", action="store_true",
                    help="build twice in-process and require identical output; pair with "
                         "--rebuild-imputation to exercise the imputation itself rather "
                         "than a cache read")
    ap.add_argument("--dry-run", action="store_true", help="do not write artifacts")
    args = ap.parse_args()

    excluded = tuple(int(y) for y in args.exclude_years.split(",") if y.strip())

    kw = dict(elev_band=args.band, mode=args.mode, obs_qa_file=args.qa_file,
              obs_threshold=args.obs_threshold, excluded_years=excluded,
              library_id=args.library_id, config_dir=args.config_dir)

    frame = cf.build_canonical_frame(args.basin, rebuild_imputation=args.rebuild_imputation,
                                     **kw)
    m = frame.manifest

    # ---------------------------------------------------------------- report
    print("\n" + "=" * 78)
    print(f"CANONICAL FRAME  {m['frame_id']}")
    print("=" * 78)
    print(f"  X shape                : {tuple(m['shape_X'])}")
    print(f"  y shape                : {tuple(m['shape_y'])}")
    print(f"  stats_n_obs / fit_n_obs: {m['stats_n_obs']} / {m['fit_n_obs']}"
          f"   (add_zero_aso={m['add_zero_aso']}, fit_intercept={m['fit_intercept']})")
    print(f"  residual NaN dropped   : {m['n_flights_dropped_residual_nan']}")
    print(f"  flights                : {m['n_flights_raw']} raw -> {m['stats_n_obs']} retained")
    print(f"  date range             : {m['train_date_min']} -> {m['train_date_max']}")
    print(f"  water years            : {m['water_years']}")
    print(f"  flights per WY         : {m['flights_per_water_year']}")

    print(f"\n  RETAINED {m['n_retained']} of {m['n_candidate_pillows']} candidates "
          f"(rule: {m['eligibility_rule']})")
    print(f"    {', '.join(m['retained_universe'])}")
    print(f"  DROPPED {len(m['dropped_pillows'])}:")
    for p, v in sorted(m["dropped_pillows"].items(), key=lambda kv: -kv[1]):
        print(f"    {p:5s} frac_missing={v:.3f}")

    print(f"\n  IMPUTATION  method={m['impute_method']}")
    print(f"    cells imputed        : {m['n_imputed_cells']} of {m['n_cells']} "
          f"({m['frac_imputed_overall']:.3%})")
    print(f"    max_pillow_frac      : {m['max_pillow_frac_imputed']:.3f}")
    print(f"    mean_pillow_frac     : {m['mean_pillow_frac_imputed']:.3f}")
    print(f"    donor window         : >= {m['impute_donor_window_start_year']}, "
          f"max {m['impute_donor_max']} donors")
    print(f"    cache hit            : {m['impute_cache_hit']}")
    print("    per-pillow frac imputed (descending):")
    for p, v in sorted(m["frac_imputed_per_pillow"].items(), key=lambda kv: -kv[1]):
        bar = "#" * int(round(v * 40))
        print(f"      {p:5s} {v:6.3f}  {bar}")

    # a few concrete rows, so the matrix is inspectable at a glance
    show = ["time", "water_year", cf._ASO_COL] + list(m["retained_universe"])[:6]
    with pd.option_context("display.width", 200, "display.max_columns", 50):
        print("\n  first 3 rows (first 6 pillows):")
        print(frame.table[show].head(3).to_string(index=False))
        print("  last 3 rows:")
        print(frame.table[show].tail(3).to_string(index=False))

    # ---------------------------------------------------- assertions / tests
    print("\n  ASSERTIONS")
    checks = []

    problems = cf.validate_canonical_frame(frame)
    checks.append(("validate_canonical_frame (all invariants)", not problems,
                   "; ".join(problems) or "all pass"))

    obs = ~frame.imputed
    checks.append(("imputation altered NO observed cell",
                   bool(np.array_equal(frame.X[obs], frame.X_raw[obs])),
                   f"{int(obs.sum())} observed cells exact; "
                   f"{m['observed_cells_restored_from_raw']} were restored from raw "
                   f"(max CSV delta {m['observed_cell_max_csv_delta_mm']:.3e} mm)"))

    checks.append(("imputed mask == pre-imputation NaN pattern",
                   bool(np.array_equal(np.isnan(frame.X_raw), frame.imputed)),
                   f"{int(frame.imputed.sum())} cells were NaN, all now finite"))

    checks.append(("y taken from raw ASO, not the imputation CSV",
                   m["target_max_csv_delta_mm"] < 1e-6,
                   f"max CSV delta {m['target_max_csv_delta_mm']:.3e} mm"))

    checks.append(("every retained pillow has a value on every retained flight",
                   bool(np.isfinite(frame.X).all()),
                   f"{int((~np.isfinite(frame.X)).sum())} non-finite cells"))

    checks.append(("y complete", bool(np.isfinite(frame.y).all()),
                   f"{int((~np.isfinite(frame.y)).sum())} non-finite"))

    # identical dates and y for every combination is trivially true for a single shared
    # matrix, but assert the consequence that matters: no column has its own NaN pattern
    checks.append(("all combinations share identical dates and y "
                   "(no per-column NaN pattern)",
                   int(frame.table[list(m["retained_universe"])].isnull().sum().sum()) == 0,
                   "0 NaN in any pillow column"))

    checks.append(("no daily-availability input in the manifest",
                   not any(k in m for k in ("all_pils_QA", "prediction_date", "t_idx")),
                   "no all_pils_QA / prediction_date / t_idx key"))

    checks.append(("synthetic zero row absent from the canonical frame",
                   pd.Timestamp("1900-01-01") not in frame.dates
                   and not bool((frame.X == 0).all(axis=1).any()),
                   "added at fit time only"))

    # imputation-burden accessor works and is bounded
    burden = frame.imputation_burden(list(m["retained_universe"])[:3])
    checks.append(("imputation_burden() returns bounded fractions",
                   all(0.0 <= burden[k] <= 1.0 for k in
                       ("frac_imputed", "max_pillow_frac_imputed",
                        "mean_pillow_frac_imputed")),
                   json.dumps(burden)))

    # the imputation mask must agree with the pre-imputation NaN pattern
    checks.append(("imputed mask cell count == manifest",
                   int(frame.imputed.sum()) == m["n_imputed_cells"],
                   f"{int(frame.imputed.sum())} cells"))

    # observed cells must be untouched by imputation: where not imputed, X must equal
    # the raw observation. Verified by re-deriving the raw frame is out of scope here;
    # instead assert the mask is consistent with the retained-column missingness.
    checks.append(("imputed mask column means == frac_imputed_per_pillow",
                   np.allclose(frame.imputed.mean(axis=0),
                               [m["frac_imputed_per_pillow"][p]
                                for p in m["retained_universe"]]),
                   "exact"))

    checks.append(("retained universe sorted (stable bitmask bit order)",
                   list(frame.universe) == sorted(frame.universe),
                   ",".join(frame.universe[:4]) + ",..."))

    checks.append(("git provenance recorded",
                   bool(m["git"]["commit"]) and m["git"]["branch"] is not None,
                   f"{m['git']['branch']}@{m['git']['commit_short']} "
                   f"dirty={m['git']['dirty']}"))

    checks.append(("QA provenance recorded (qa_tag + file digest)",
                   bool(m["qa_tag"]) and bool(m["obs_qa"]["sha256_16"]),
                   f"{m['qa_tag']} sha={m['obs_qa']['sha256_16']}"))

    checks.append(("stale cache from another QA rung cannot be reused",
                   m["qa_tag"] in m["frame_id"] and m["inputs_hash"] in m["frame_id"],
                   f"cache path carries qa_tag and inputs_hash={m['inputs_hash']}"))

    checks.append(("imputation cache is outside the production namespace",
                   "/imputation/" not in m["impute_cache_path"]
                   and "model_library" in m["impute_cache_path"],
                   m["impute_cache_path"]))

    if args.check_determinism:
        # With --rebuild-imputation this re-runs the donor search from scratch, which is
        # the only version of this test that says anything about the imputation. Without
        # it the second build is a cache read and only the assembly is exercised.
        f2 = cf.build_canonical_frame(args.basin,
                                      rebuild_imputation=args.rebuild_imputation,
                                      verbose=False, **kw)
        same = (f2.manifest["frame_id"] == m["frame_id"]
                and np.array_equal(f2.X, frame.X, equal_nan=True)
                and np.array_equal(f2.y, frame.y, equal_nan=True)
                and np.array_equal(f2.imputed, frame.imputed)
                and f2.universe == frame.universe)
        checks.append((f"determinism: rebuild is bit-identical "
                       f"({'imputation recomputed' if args.rebuild_imputation else 'cache read'})",
                       same, f"frame_id {f2.manifest['frame_id'][-8:]}"))

    ok = True
    for name, passed, detail in checks:
        ok &= bool(passed)
        print(f"    [{'PASS' if passed else 'FAIL'}] {name}")
        print(f"           {detail}")

    # ------------------------------------------------------------- artifacts
    if args.dry_run:
        print("\n  --dry-run: no artifacts written")
    else:
        written = cf.save_canonical_frame(frame)
        print("\n  ARTIFACTS")
        for k, v in written.items():
            sz = Path(v).stat().st_size if Path(v).exists() else 0
            print(f"    {k:14s} {v}  ({sz:,} bytes)")

        rt = cf.load_canonical_frame(written["manifest"])
        rt_ok = (np.array_equal(rt.X, frame.X) and np.array_equal(rt.y, frame.y)
                 and np.array_equal(rt.imputed, frame.imputed)
                 and rt.universe == frame.universe)
        print(f"    [{'PASS' if rt_ok else 'FAIL'}] parquet round-trip is exact")
        ok &= rt_ok

    print("\n" + ("  ALL CHECKS PASSED" if ok else "  *** CHECKS FAILED ***"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
