"""
prediction_result_demo.py

Demonstrates and validates build_prediction_result(): one function call that returns the
full-history baseline prediction plus its two independent sensitivity dimensions,
station-loss and historical-training, for one date/band.

No model fitting, no library rebuild, no new eligibility logic. This script:

  1. loads the already-built full-history and leave-one-year libraries (read-only)
  2. gets today's available pillows + pillow values from real FRIANT test-WY data via
     preprocessing.process_daily_qa(), reusing mlr_ab_harness.replay_preprocessing() --
     the same reuse pattern the station-loss and historical-year drivers already follow
  3. calls build_prediction_result() once per date/band
  4. independently re-derives the same three pieces (baseline, station-loss,
     historical-year) by calling the ORIGINAL functions directly, and asserts the combined
     result reproduces them exactly

Usage (conda env: bor):
    python prediction_result_demo.py FRIANT 2025 --bands total,8000-9000 --dates 2025-04-01
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(1, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(1, str(Path(__file__).resolve().parent))

from snow_ops.mlr.lookup import store as ls                                # noqa: E402
from snow_ops.mlr.lookup.sensitivity import station_loss_sensitivity       # noqa: E402
from snow_ops.mlr.lookup.historical_year import historical_year_sensitivity  # noqa: E402
from snow_ops.mlr.lookup.prediction_result import (                        # noqa: E402
    build_prediction_result, summary_table)
from historical_year_sensitivity import find_manifest, load_dropped_year_libraries  # noqa: E402
from mlr_ab_harness import replay_preprocessing, sample_indices            # noqa: E402
import preprocessing                                                        # noqa: E402


def real_day_data(basin: str, water_year: int, qa_file: str, dates: list[str] | None,
                  n_dates: int) -> list[dict]:
    """Real availability + pillow values for either specific calendar dates or an evenly
    sampled set of timesteps across the water year."""
    P = replay_preprocessing(basin, water_year, qa_file, "wy", True)
    times = pd.to_datetime(P["obs_test_lst"][0].time.values)

    if dates:
        idxs = [int(np.where(times.normalize() == pd.Timestamp(d))[0][0]) for d in dates]
    else:
        idxs = sample_indices(P["n_timesteps"], n_dates)

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
    return day_data


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("basin")
    ap.add_argument("water_year", type=int)
    ap.add_argument("--library-version", required=True,
                    help="e.g. lookup_lib_v1 -- required, never inferred")
    ap.add_argument("--bands", default="total,8000-9000")
    ap.add_argument("--dates", default="2025-04-01",
                    help="comma-separated YYYY-MM-DD, or empty to sample --n-dates")
    ap.add_argument("--n-dates", type=int, default=5)
    ap.add_argument("--qa-file", default="pillow_wy_1980_2025_qa6.nc")
    args = ap.parse_args()

    bands = [b.strip() for b in args.bands.split(",") if b.strip()]
    dates = [d.strip() for d in args.dates.split(",") if d.strip()] or None

    t_all = time.perf_counter()
    print(f"PREDICTION RESULT DEMO  {args.basin} WY{args.water_year}  bands={bands}")

    day_data = real_day_data(args.basin, args.water_year, args.qa_file, dates, args.n_dates)
    print(f"  dates: {[d['date'] for d in day_data]}")

    # ---- load libraries (read-only; nothing is fit or rebuilt here) --------
    t0 = time.perf_counter()
    baseline_libs, dropped_libs = {}, {}
    for band in bands:
        baseline_libs[band] = ls.load_model_library(
            find_manifest(args.basin, band, "all_years", args.library_version))
        dropped_libs[band] = load_dropped_year_libraries(
            args.basin, band, args.library_version)
    t_load = time.perf_counter() - t0
    print(f"  loaded {len(bands)} baseline + "
          f"{sum(len(v) for v in dropped_libs.values())} dropped-year libraries "
          f"in {t_load:.2f}s")

    # ---- build the combined result + independently re-derive every piece --
    t0 = time.perf_counter()
    results = []
    checks = []
    for band in bands:
        blib, dlibs = baseline_libs[band], dropped_libs[band]
        for d in day_data:
            r = build_prediction_result(
                blib, dlibs, d["available"], d["values"],
                basin=args.basin, band=band, date=d["date"])
            if r is None:
                continue
            results.append(r)

            # -- 1. baseline reproduces select_best_model() + predict() exactly --
            direct_baseline = blib.select_best_model(d["available"], score_rule="adj_r2")
            direct_pred = blib.predict(direct_baseline, d["values"])
            checks.append((
                f"{band}/{d['date']}: baseline == select_best_model()+predict()",
                r.baseline_pillow_key == direct_baseline["pillow_key"]
                and abs(r.baseline_prediction_mm - direct_pred) < 1e-12
                and r.baseline_n_predictors == int(direct_baseline["n_predictors"])
                and abs(r.baseline_adj_r2 - float(direct_baseline["adj_r2"])) < 1e-12
                and abs(r.baseline_rmse - float(direct_baseline["rmse"])) < 1e-12,
                f"combined={r.baseline_pillow_key} pred={r.baseline_prediction_mm:.6f}  "
                f"direct={direct_baseline['pillow_key']} pred={direct_pred:.6f}"))

            # -- 2. station-loss members exactly reproduce the standalone function --
            direct_sl = station_loss_sensitivity(
                blib, d["available"], d["values"], band=band, date=d["date"])
            combined_sl = r.station_loss_sensitivity.members
            sl_ok = (len(direct_sl) == len(combined_sl) and all(
                a.dropped_pillow == b.dropped_pillow
                and a.replacement_pillow_key == b.pillow_key
                and a.replacement_eligible == b.eligible
                and ((a.replacement_pred_mm is None and b.prediction_mm is None)
                     or abs(a.replacement_pred_mm - b.prediction_mm) < 1e-12)
                for a, b in zip(direct_sl, combined_sl)))
            checks.append((f"{band}/{d['date']}: station-loss members == "
                           f"station_loss_sensitivity() ({len(combined_sl)} members)",
                           sl_ok, "exact field-by-field match" if sl_ok else "MISMATCH"))

            # -- 3. historical-year members exactly reproduce the standalone function --
            direct_hy = historical_year_sensitivity(
                blib, dlibs, d["available"], d["values"], band=band, date=d["date"])
            combined_hy = r.historical_training_sensitivity.members
            hy_ok = (len(direct_hy) == len(combined_hy) and all(
                a.dropped_wy == b.excluded_wy
                and a.library_pillow_key == b.pillow_key
                and a.eligible == b.eligible
                and ((a.library_pred_mm is None and b.prediction_mm is None)
                     or abs(a.library_pred_mm - b.prediction_mm) < 1e-12)
                for a, b in zip(direct_hy, combined_hy)))
            checks.append((f"{band}/{d['date']}: historical-year members == "
                           f"historical_year_sensitivity() ({len(combined_hy)} members)",
                           hy_ok, "exact field-by-field match" if hy_ok else "MISMATCH"))

            # -- 4. every selected model's pillows are a subset of today's available --
            avail = set(d["available"])
            bad = 0
            if not (set(r.baseline_pillow_key.split(",")) <= avail):
                bad += 1
            for m in combined_sl:
                if m.eligible and not (set(m.pillow_key.split(",")) <= avail - {m.dropped_pillow}):
                    bad += 1
            for m in combined_hy:
                if m.eligible and not (set(m.pillow_key.split(",")) <= avail):
                    bad += 1
            checks.append((f"{band}/{d['date']}: no selected model requires an "
                           f"unavailable pillow", bad == 0, f"{bad} violations"))

    t_query = time.perf_counter() - t0

    print(f"\n  {len(results)} PredictionResults built "
          f"({len(bands)} bands x {len(day_data)} dates) in {t_query * 1000:.2f} ms "
          f"-- no fitting, no imputation, read-only queries only")

    print("\nVALIDATION")
    ok_all = all(p for _, p, _ in checks)
    for name, passed, detail in checks:
        print(f"  [{'PASS' if passed else 'FAIL'}] {name}")
        if not passed:
            print(f"         {detail}")
    if ok_all:
        print(f"  all {len(checks)} checks passed")

    # ------------------------------------------------------------- reporting
    print("\nCOMPACT SUMMARY")
    st = summary_table(results)
    with pd.option_context("display.width", 220, "display.max_columns", 30):
        print(st.to_string(index=False))

    print("\nINDIVIDUAL SENSITIVITY MEMBERS")
    for r in results:
        print(f"\n  band={r.elevation_band}  date={r.date}")
        print(f"    baseline: {r.baseline_pillow_key}  "
              f"pred={r.baseline_prediction_mm:.1f} mm  "
              f"(k={r.baseline_n_predictors}, adj_r2={r.baseline_adj_r2:.4f}, "
              f"rmse={r.baseline_rmse:.2f})")

        print(f"\n    historical-training sensitivity "
              f"({len(r.historical_training_sensitivity.members)} dropped-year members):")
        print(r.historical_training_sensitivity.table().to_string(index=False))
        h = r.historical_training_sensitivity
        print(f"    summary: min={h.min_prediction_mm:.1f} max={h.max_prediction_mm:.1f} "
              f"range={h.range_mm:.1f} std={h.std_prediction_mm:.1f} "
              f"max_abs_delta={h.max_abs_delta_mm:.1f} "
              f"(most influential: WY{h.most_influential_excluded_wy})")

        print(f"\n    station-loss sensitivity "
              f"({len(r.station_loss_sensitivity.members)} dropped-pillow members):")
        print(r.station_loss_sensitivity.table().to_string(index=False))
        s = r.station_loss_sensitivity
        print(f"    summary: min={s.min_prediction_mm:.1f} max={s.max_prediction_mm:.1f} "
              f"range={s.range_mm:.1f} std={s.std_prediction_mm:.1f} "
              f"max_abs_delta={s.max_abs_delta_mm:.1f} "
              f"(most influential: {s.most_influential_dropped_pillow})")

    print(f"\nRUNTIME  libraries loaded in {t_load:.2f}s; "
          f"{len(results)} combined predictions + validation in {t_query * 1000:.2f} ms "
          f"(after libraries are loaded)")
    print(f"  grand total {time.perf_counter() - t_all:.2f}s")
    print("  " + ("ALL CHECKS PASSED" if ok_all else "*** CHECKS FAILED ***"))
    return 0 if ok_all else 1


if __name__ == "__main__":
    raise SystemExit(main())
