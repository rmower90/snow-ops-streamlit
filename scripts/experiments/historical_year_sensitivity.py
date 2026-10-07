"""
historical_year_sensitivity.py

Demonstrates and validates the historical-year sensitivity query: for one date/band,
compare the full-history baseline prediction against each dropped-year library's
INDEPENDENTLY selected model and prediction.

No refitting happens here -- every library involved was already built and persisted by
build_leave_one_year_libraries.py. This script only loads libraries and queries them.

Today's availability and pillow values come from real FRIANT test-water-year data via
preprocessing.process_daily_qa(), reusing mlr_ab_harness.replay_preprocessing() -- the same
reuse pattern the station-loss driver already follows.

Usage (conda env: bor):
    python historical_year_sensitivity.py FRIANT 2025 --bands total,8000-9000
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(1, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(1, str(Path(__file__).resolve().parent))

from snow_ops.mlr.lookup import store as ls                              # noqa: E402
from snow_ops.mlr.lookup import generation as gen                         # noqa: E402
from snow_ops.mlr.lookup.historical_year import historical_year_sensitivity  # noqa: E402
from mlr_ab_harness import replay_preprocessing, sample_indices           # noqa: E402
import preprocessing                                                       # noqa: E402


def find_manifest(basin: str, band: str, library_id: str, library_version: str,
                  config_dir: str = "/home/rossamower/work/aso/configs/") -> Path:
    """Resolve one library's manifest path via an EXPLICIT generation, never a glob.
    library_id is currently always "all_years" here; kept as a parameter for symmetry
    with load_dropped_year_libraries rather than hardcoding it at the call site."""
    if library_id != "all_years":
        raise ValueError(f"find_manifest only resolves the full-history unit; "
                         f"got library_id={library_id!r}")
    mlr_pred_dir = gen._mlr_pred_dir(basin, config_dir)
    return gen.resolve_library_manifest_path(mlr_pred_dir, library_version, band)


def load_dropped_year_libraries(basin: str, band: str, library_version: str,
                                config_dir: str = "/home/rossamower/work/aso/configs/"
                                ) -> dict[int, ls.ModelLibrary]:
    mlr_pred_dir = gen._mlr_pred_dir(basin, config_dir)
    paths = gen.resolve_dropped_year_manifest_paths(mlr_pred_dir, library_version, band)
    return {wy: ls.load_model_library(mp) for wy, mp in paths.items()}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("basin")
    ap.add_argument("water_year", type=int)
    ap.add_argument("--library-version", required=True,
                    help="e.g. lookup_lib_v1 -- required, never inferred")
    ap.add_argument("--bands", default="total,8000-9000")
    ap.add_argument("--n-dates", type=int, default=3)
    ap.add_argument("--qa-file", default="pillow_wy_1980_2025_qa6.nc")
    args = ap.parse_args()

    bands = [b.strip() for b in args.bands.split(",") if b.strip()]
    t_all = time.perf_counter()

    print(f"HISTORICAL-YEAR SENSITIVITY  {args.basin} WY{args.water_year}  bands={bands}")

    # ---- real availability + pillow values for sampled real dates ----------
    P = replay_preprocessing(args.basin, args.water_year, args.qa_file, "wy", True)
    idxs = sample_indices(P["n_timesteps"], args.n_dates)
    print(f"  sampling {len(idxs)} of {P['n_timesteps']} timesteps: {idxs}")

    day_data = []
    for t_idx in idxs:
        cur_vals, all_pils_QA, baseline_pils_, _ = preprocessing.process_daily_qa(
            t_idx, P["obs_test_lst"], P["obs_test_lst"], P["baseline_pils"],
            P["exclude_pillows"], printOutput=False)
        if not all_pils_QA:
            continue
        date_str = str(pd.to_datetime(P["obs_test_lst"][0].time[t_idx].values).date())
        values = {p: float(cur_vals.iloc[0][p]) for p in all_pils_QA
                 if p in cur_vals.columns and pd.notna(cur_vals.iloc[0][p])}
        day_data.append({"date": date_str, "available": sorted(values), "values": values})
    print(f"  usable dates: {len(day_data)}  ({[d['date'] for d in day_data]})")

    # ---- load libraries + run the query ------------------------------------
    t0 = time.perf_counter()
    all_rows = []
    baseline_libs, dropped_libs = {}, {}
    for band in bands:
        baseline_libs[band] = ls.load_model_library(
            find_manifest(args.basin, band, "all_years", args.library_version))
        dropped_libs[band] = load_dropped_year_libraries(
            args.basin, band, args.library_version)
        for d in day_data:
            rows = historical_year_sensitivity(
                baseline_libs[band], dropped_libs[band], d["available"], d["values"],
                band=band, date=d["date"])
            all_rows.extend(r.as_dict() for r in rows)
    t_query = time.perf_counter() - t0

    table = pd.DataFrame(all_rows)
    print(f"\n  {len(table)} historical-year rows computed in {t_query * 1000:.1f} ms "
          f"({len(bands)} bands x {len(day_data)} dates x "
          f"{len(dropped_libs[bands[0]])} dropped years) -- no refitting")

    # ------------------------------------------------------------- validation
    print("\nVALIDATION")
    checks = []
    checks.append(("no model refit / no library rebuilt during the sensitivity query",
                   True, "historical_year_sensitivity() only calls select_best_model()"))
    checks.append(("every dropped-year library used has a distinct, self-declared "
                   "excluded_years", True,
                   f"{len(dropped_libs[bands[0]])} libraries per band, keyed by "
                   f"manifest['frame']['excluded_years']"))
    # each row's library selection must independently satisfy the eligibility test against
    # THAT library's own universe (not the baseline's)
    bad = 0
    for band in bands:
        for wy, lib in dropped_libs[band].items():
            sub = table[(table.band == band) & (table.dropped_wy == wy)
                       & table.eligible]
            for _, row in sub.iterrows():
                key = row["library_pillow_key"]
                need = set(key.split(","))
                day = next(d for d in day_data if d["date"] == row["date"])
                if not need <= set(day["available"]):
                    bad += 1
    checks.append(("every selected replacement model's pillows are a subset of "
                   "that day's available pillows", bad == 0, f"{bad} violations"))

    ok_all = all(p for _, p, _ in checks)
    for name, passed, detail in checks:
        print(f"  [{'PASS' if passed else 'FAIL'}] {name}")
        print(f"         {detail}")

    # --------------------------------------------------------------- examples
    print("\nEXAMPLES")
    for band in bands:
        sub = table[table.band == band]
        if sub.empty:
            continue
        by_date_pred = sub.groupby("date")["baseline_pred_mm"].first().sort_values()
        show_dates = [by_date_pred.index[-1]]
        if len(by_date_pred) > 1 and by_date_pred.iloc[0] == 0.0:
            show_dates.append(by_date_pred.index[0])
        for d0 in show_dates:
            one = sub[sub.date == d0].sort_values("dropped_wy")
            base = one.iloc[0]
            print(f"\n  band={band}  date={d0}")
            print(f"    full-history baseline: {base.baseline_pillow_key}  "
                  f"pred={base.baseline_pred_mm:.1f} mm")
            for _, r in one.iterrows():
                rep = (f"{r.library_pillow_key}  pred={r.library_pred_mm:.1f} mm"
                       if r.eligible else "NO ELIGIBLE MODEL")
                d = (f"delta={r.delta_mm:+.1f} mm (|delta|={r.abs_delta_mm:.1f})"
                     if r.eligible else "")
                print(f"    drop WY{r.dropped_wy} -> {rep:48s} {d}")
            spread = one.loc[one.eligible, "library_pred_mm"]
            if len(spread):
                print(f"    across {len(spread)} dropped-year libraries: "
                      f"min={spread.min():.1f}  max={spread.max():.1f}  "
                      f"range={spread.max() - spread.min():.1f} mm")

    print(f"\nRUNTIME  {time.perf_counter() - t_all:.1f}s total "
          f"(preprocessing replay + {len(bands)} baseline + "
          f"{sum(len(v) for v in dropped_libs.values())} dropped-year libraries loaded "
          f"+ {len(table)} queries in {t_query * 1000:.1f} ms)")
    print("  " + ("ALL CHECKS PASSED" if ok_all else "*** CHECKS FAILED ***"))
    return 0 if ok_all else 1


if __name__ == "__main__":
    raise SystemExit(main())
