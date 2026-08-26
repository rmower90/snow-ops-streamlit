"""
daily_lookup_prediction.py

Standalone daily prediction driver using ONLY the pretrained model libraries: one basin,
one date, all elevation bands. No regression fitting, no imputation, no cross-validation,
and no library rebuilding happen anywhere in this path.

NEW, PARALLEL path. Does not modify, call, or write into mlr_prediction.py's output tree --
every table here lands under model_library/daily_lookup/, never inside the production
imputation/ or vN/ directories. This is the single-day counterpart to
historic_lookup_prediction.py, sharing its library-loading and provenance helpers rather
than duplicating them.

WHICH "EXISTING DAILY QA PATHWAY" THIS REUSES

mlr_prediction.py (the live daily runner) calls preprocessing.process_daily_qa() with the
CURRENT water year's raw insitu file as `obs_data_raw` and the QA-processed insitu file as
`obs_data_qa` -- two genuinely different daily-cadence NetCDFs
({basin}_insitu_obs_daily_wy{YYYY}.nc under raw/ and processed/). FRIANT has no such live
WY2026 file in this environment (checked: only the historical pillow_wy_1980_2025*.nc
archives exist under insitu/FRIANT/{raw,processed}/) -- there is no live feed to test
against for this basin.

So this driver reuses process_daily_qa() the same way every earlier lookup driver in this
branch already does: via mlr_ab_harness.replay_preprocessing(), which supplies the SAME
test-water-year daily observations as both the raw and QA arguments (matching how
mlr_prediction_historic.py itself replays a past water year). It is the same QA/eligibility
function production uses, applied to one requested date instead of live raw+processed
files that do not exist here.

Per requested date:

    1. preprocessing.process_daily_qa() -- today's QA-approved pillow set and values.
       This is the ONLY per-date computation; it does not fit, impute, or touch a single
       pretrained coefficient.
    2. per elevation band: build_prediction_result(baseline_lib, dropped_year_libs,
       available_pillows, pillow_values) -- baseline + station-loss sensitivity +
       historical-training sensitivity, entirely from already-fitted tables.

Five output tables under model_library/daily_lookup/{basin}_{date}/:

    baseline_mm.csv                   one row (the date), one column per elevation band
                                       plus 'Basin' for the 'total' band
    sensitivity_summary.csv           one row per band: baseline + both summaries
    historical_training_members.csv   long form: band, excluded_wy, pillow_key, ...
    station_loss_members.csv          long form: band, dropped_pillow, pillow_key, ...
    selected_model.csv                band, pillow_key, n_predictors, adj_r2, rmse,
                                       imputation burden of the winning model, n available
    run_manifest.json                 provenance: libraries used, requested date, git state

Usage (conda env: bor):
    python daily_lookup_prediction.py FRIANT 2025 2025-04-01
"""

from __future__ import annotations

import argparse
import ast
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(1, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(1, str(Path(__file__).resolve().parent))

from snow_ops.mlr.lookup.prediction_result import build_prediction_result  # noqa: E402
from build_all_bands import aso_band_labels                                # noqa: E402
from historic_lookup_prediction import load_all_libraries, _git_state      # noqa: E402
from mlr_ab_harness import replay_preprocessing                           # noqa: E402
import preprocessing                                                       # noqa: E402


def run_daily_lookup(
    basin: str, water_year: int, date_str: str, bands: list[str],
    baseline_libs: dict, dropped_libs: dict,
    qa_file: str = "pillow_wy_1980_2025_qa6.nc", score_rule: str = "adj_r2",
) -> dict:
    """
    Run the lookup-only path for ONE date across all bands. Returns the five output tables
    plus timing, and the (available_pillows, pillow_values) actually used -- so the caller
    can independently re-verify without re-running the preprocessing replay.
    """
    t0 = time.perf_counter()
    P = replay_preprocessing(basin, water_year, qa_file, "wy", True)
    times = pd.to_datetime(P["obs_test_lst"][0].time.values)
    hits = np.where(times.normalize() == pd.Timestamp(date_str))[0]
    if not len(hits):
        raise SystemExit(f"{date_str} not in WY{water_year} test range "
                         f"({times.min().date()} .. {times.max().date()})")
    t_idx = int(hits[0])

    cur_vals, all_pils_QA, _, _ = preprocessing.process_daily_qa(
        t_idx, P["obs_test_lst"], P["obs_test_lst"], P["baseline_pils"],
        P["exclude_pillows"], printOutput=False)
    t_qa = time.perf_counter() - t0

    values = {p: float(cur_vals.iloc[0][p]) for p in all_pils_QA
             if p in cur_vals.columns and pd.notna(cur_vals.iloc[0][p])}
    available = sorted(values)

    t0 = time.perf_counter()
    baseline_row = {"date": date_str}
    summary_rows, hist_rows, station_rows, model_rows = [], [], [], []
    n_no_baseline = []
    for band in bands:
        r = build_prediction_result(
            baseline_libs[band], dropped_libs[band], available, values,
            score_rule=score_rule, basin=basin, band=band, date=date_str)
        if r is None:
            baseline_row[band] = np.nan
            n_no_baseline.append(band)
            continue
        baseline_row[band] = r.baseline_prediction_mm
        summary_rows.append(r.summary_dict())
        hist_rows.extend({"band": band, **m.__dict__}
                         for m in r.historical_training_sensitivity.members)
        station_rows.extend({"band": band, **m.__dict__}
                            for m in r.station_loss_sensitivity.members)
        wrow = baseline_libs[band].table.loc[
            baseline_libs[band].table["pillow_key"] == r.baseline_pillow_key].iloc[0]
        model_rows.append({
            "band": band, "pillow_key": r.baseline_pillow_key,
            "n_predictors": r.baseline_n_predictors, "adj_r2": r.baseline_adj_r2,
            "rmse": r.baseline_rmse,
            "max_pillow_frac_imputed": float(wrow["max_pillow_frac_imputed"]),
            "mean_pillow_frac_imputed": float(wrow["mean_pillow_frac_imputed"]),
            "n_available_pillows": len(available),
        })
    t_predict = time.perf_counter() - t0

    baseline_df = pd.DataFrame([baseline_row]).rename(columns={"total": "Basin"})
    return dict(
        baseline=baseline_df, summary=pd.DataFrame(summary_rows),
        historical_training=pd.DataFrame(hist_rows),
        station_loss=pd.DataFrame(station_rows),
        selected_model=pd.DataFrame(model_rows),
        available=available, values=values,
        stats=dict(t_qa=t_qa, t_predict=t_predict, n_no_baseline=n_no_baseline),
    )


def save_outputs(basin: str, date_str: str, results: dict, bands: list[str],
                 libraries_used: dict) -> Path:
    out_dir = Path(f"/home/rossamower/work/aso/data/mlr_prediction/{basin}/"
                   f"model_library/daily_lookup/{basin}_{date_str}")
    out_dir.mkdir(parents=True, exist_ok=True)

    results["baseline"].to_csv(out_dir / "baseline_mm.csv", index=False, float_format="%.6f")
    results["summary"].to_csv(out_dir / "sensitivity_summary.csv", index=False, float_format="%.6f")
    results["historical_training"].to_csv(out_dir / "historical_training_members.csv",
                                          index=False, float_format="%.6f")
    results["station_loss"].to_csv(out_dir / "station_loss_members.csv",
                                   index=False, float_format="%.6f")
    results["selected_model"].to_csv(out_dir / "selected_model.csv", index=False, float_format="%.6f")

    repo_root = Path(__file__).resolve().parents[2]
    manifest = {
        "basin": basin, "date": date_str, "bands": bands,
        "built_at_utc": datetime.now(timezone.utc).isoformat(),
        "git": _git_state(repo_root),
        "libraries_used": libraries_used,
        "n_available_pillows": len(results["available"]),
        "available_pillows": results["available"],
        "stats": results["stats"],
        "note": ("Lookup-only single-date prediction path. No regression fitting, "
                "imputation, or cross-validation occurred. Parallel to, and does not "
                "modify, mlr_prediction.py."),
    }
    (out_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2, default=str) + "\n")
    return out_dir


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("basin")
    ap.add_argument("water_year", type=int)
    ap.add_argument("date", help="YYYY-MM-DD")
    ap.add_argument("--bands", default="", help="comma-separated subset; default = all")
    ap.add_argument("--qa-file", default="pillow_wy_1980_2025_qa6.nc")
    ap.add_argument("--config-dir", default="/home/rossamower/work/aso/configs/")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    bands = ([b.strip() for b in args.bands.split(",") if b.strip()]
             or aso_band_labels(args.basin, args.config_dir))
    t_all = time.perf_counter()

    print(f"DAILY LOOKUP PREDICTION  {args.basin}  date={args.date}  bands={bands}")

    baseline_libs, dropped_libs, t_load = load_all_libraries(args.basin, bands)
    print(f"  library load: {t_load:.2f}s "
          f"({len(bands)} baseline + {sum(len(v) for v in dropped_libs.values())} "
          f"dropped-year libraries)")

    results = run_daily_lookup(args.basin, args.water_year, args.date, bands,
                               baseline_libs, dropped_libs, qa_file=args.qa_file)
    st = results["stats"]
    print(f"  daily QA/preprocessing: {st['t_qa']:.3f}s  "
          f"({len(results['available'])} pillows available)")
    print(f"  lookup/prediction: {st['t_predict']:.3f}s  "
          f"({len(bands)} bands, ~17 lookup queries/band, 0 fits)")
    if st["n_no_baseline"]:
        print(f"  bands with no eligible baseline model: {st['n_no_baseline']}")

    print("\nBASELINE PREDICTIONS (mm)")
    print(results["baseline"].to_string(index=False))

    # ------------------------------------------------------------- validation
    print("\nVALIDATION")
    checks = []

    # 1. every band's baseline reproduces a direct, independently-derived
    #    build_prediction_result() call
    t0 = time.perf_counter()
    P2 = replay_preprocessing(args.basin, args.water_year, args.qa_file, "wy", True)
    times2 = pd.to_datetime(P2["obs_test_lst"][0].time.values)
    t_idx2 = int(np.where(times2.normalize() == pd.Timestamp(args.date))[0][0])
    cur_vals2, all_pils_QA2, _, _ = preprocessing.process_daily_qa(
        t_idx2, P2["obs_test_lst"], P2["obs_test_lst"], P2["baseline_pils"],
        P2["exclude_pillows"], printOutput=False)
    values2 = {p: float(cur_vals2.iloc[0][p]) for p in all_pils_QA2
              if p in cur_vals2.columns and pd.notna(cur_vals2.iloc[0][p])}
    available2 = sorted(values2)
    same_availability = (available2 == results["available"])
    checks.append(("independent re-derivation of today's QA availability matches",
                   same_availability,
                   f"{len(available2)} pillows, identical: {same_availability}"))

    bdf = results["baseline"]
    for band in bands:
        direct = build_prediction_result(
            baseline_libs[band], dropped_libs[band], available2, values2,
            basin=args.basin, band=band, date=args.date)
        col = "Basin" if band == "total" else band
        table_val = bdf[col].iloc[0]
        ok = (pd.isna(table_val) if direct is None
             else abs(direct.baseline_prediction_mm - table_val) < 1e-9)
        checks.append((f"{band}: baseline == direct build_prediction_result() call",
                       bool(ok),
                       f"table={table_val}  direct="
                       f"{direct.baseline_prediction_mm if direct else None}"))
    t_validate_rederive = time.perf_counter() - t0

    # 2. no lm_model import anywhere in this module (AST, not substring -- a substring
    #    search would trip on this very comment)
    tree = ast.parse(Path(__file__).read_text())
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
    checks.append(("no lm_model import anywhere in this module (no fitting/CV possible)",
                   "lm_model" not in imported, f"imports: {sorted(imported)}"))

    # 3. every selected model (baseline, station-loss replacements, historical-year
    #    replacements) uses only today's actually-available pillows
    bad = []
    avail_set = set(results["available"])
    for _, row in results["selected_model"].iterrows():
        if not set(row["pillow_key"].split(",")) <= avail_set:
            bad.append(("baseline", row["band"], row["pillow_key"]))
    for _, row in results["station_loss"].iterrows():
        if pd.notna(row.get("pillow_key")) and row.get("eligible"):
            need = set(row["pillow_key"].split(","))
            if not need <= (avail_set - {row["dropped_pillow"]}):
                bad.append(("station-loss", row["band"], row["pillow_key"]))
    for _, row in results["historical_training"].iterrows():
        if pd.notna(row.get("pillow_key")) and row.get("eligible"):
            if not set(row["pillow_key"].split(",")) <= avail_set:
                bad.append(("historical-year", row["band"], row["pillow_key"]))
    n_selections = (len(results["selected_model"]) + len(results["station_loss"])
                    + len(results["historical_training"]))
    checks.append((f"every selected model across baseline/station-loss/historical-year "
                   f"uses only available pillows ({n_selections} selections checked)",
                   len(bad) == 0, f"{len(bad)} violations: {bad[:5]}"))

    ok_all = all(p for _, p, _ in checks)
    for name, passed, detail in checks:
        print(f"  [{'PASS' if passed else 'FAIL'}] {name}")
        if not passed:
            print(f"         {detail}")
    print(f"  {sum(p for _, p, _ in checks)}/{len(checks)} checks passed  "
          f"(independent re-derivation took {t_validate_rederive:.2f}s)")

    # ---------------------------------------------------------------- outputs
    if not args.dry_run:
        libraries_used = {
            band: {"baseline_frame_id": baseline_libs[band].manifest["frame"]["frame_id"],
                  "dropped_years": sorted(dropped_libs[band])}
            for band in bands}
        out_dir = save_outputs(args.basin, args.date, results, bands, libraries_used)
        print(f"\nOUTPUTS  {out_dir}")
        for f in sorted(out_dir.iterdir()):
            print(f"    {f.name:34s} {f.stat().st_size:>8,d} bytes")

    print(f"\nRUNTIME  library load {t_load:.2f}s + daily QA/preprocessing "
          f"{st['t_qa']:.3f}s + lookup/prediction {st['t_predict']:.3f}s  =  "
          f"grand total {time.perf_counter() - t_all:.2f}s")
    print("  " + ("ALL CHECKS PASSED" if ok_all else "*** CHECKS FAILED ***"))
    return 0 if ok_all else 1


if __name__ == "__main__":
    raise SystemExit(main())
