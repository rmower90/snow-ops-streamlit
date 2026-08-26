"""
historic_lookup_prediction.py

Historic prediction driver using ONLY the pretrained full-history model libraries built in
earlier milestones. No regression fitting, no imputation, no cross-validation, and no
library rebuilding happen anywhere in this path.

NEW, PARALLEL path. Does not modify, call, or write into mlr_prediction_historic.py's
output tree -- every table here lands under model_library/historic_lookup/, never inside
the production imputation/ or vN/ directories. This script exists to answer whether the
pretrained-lookup architecture, run once per day across a full historical water year, is
fast and produces sensible predictions, BEFORE any integration with the production runner.

Per requested day:

    1. preprocessing.process_daily_qa() -- exactly the call mlr_prediction_historic.py and
       every earlier lookup driver in this branch already use -- returns today's
       QA-approved pillow set and values. This is the ONLY per-day computation; it does
       not fit, impute, or touch a single pretrained coefficient.
    2. per elevation band: build_prediction_result(baseline_lib, dropped_year_libs,
       available_pillows, pillow_values) -- baseline + station-loss sensitivity +
       historical-training sensitivity, entirely from already-fitted tables.

Five output tables under model_library/historic_lookup/{basin}_wy{YYYY}/:

    baseline_mm.csv                   Date x elevation band (mm). Columns are the raw ASO
                                       elev labels this branch's own artifacts already use
                                       (e.g. '8000-9000'), NOT the YAML display labels
                                       ('8k-9k') the production table uses -- see
                                       --compare-existing for the mapping between them.
                                       The 'total' band is renamed 'Basin', matching
                                       postprocessing.format_prediction_tables' convention.
    sensitivity_summary.csv           one row per (date, band): baseline + both summaries
    historical_training_members.csv   long form: date, band, excluded_wy, pillow_key, ...
    station_loss_members.csv          long form: date, band, dropped_pillow, pillow_key, ...
    selected_model.csv                date, band, pillow_key, n_predictors, adj_r2, rmse,
                                       imputation burden of the winning model, n available
    run_manifest.json                 provenance: libraries used, date range, git state

Usage (conda env: bor):
    python historic_lookup_prediction.py FRIANT 2025
    python historic_lookup_prediction.py FRIANT 2025 --start-date 2025-01-01 --end-date 2025-05-01
    python historic_lookup_prediction.py FRIANT 2025 --compare-existing
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(1, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(1, str(Path(__file__).resolve().parent))

from snow_ops.mlr.lookup import generation as gen                           # noqa: E402
from snow_ops.mlr.lookup import store as ls                                # noqa: E402
from snow_ops.mlr.lookup.frame import aso_band_labels                      # noqa: E402
from snow_ops.mlr.lookup.prediction_result import build_prediction_result  # noqa: E402
from mlr_ab_harness import replay_preprocessing                            # noqa: E402
import preprocessing                                                        # noqa: E402


def load_all_libraries(basin: str, library_version: str, bands: list[str],
                      config_dir: str = "/home/rossamower/work/aso/configs/"
                      ) -> tuple[dict, dict, float]:
    """
    Read-only load of the already-built full-history + leave-one-year libraries for one
    EXPLICIT library generation. Nothing is fit or rebuilt here.

    library_version is resolved via gen.load_generation_manifest() -- a deterministic path
    built from the version string, never a directory glob -- so once multiple generations
    coexist on disk, the wrong one can never be picked by lexicographic accident.
    """
    t0 = time.perf_counter()
    mlr_pred_dir = gen._mlr_pred_dir(basin, config_dir)
    baseline_libs, dropped_libs = {}, {}
    for band in bands:
        baseline_libs[band] = ls.load_model_library(
            gen.resolve_library_manifest_path(mlr_pred_dir, library_version, band))
        dropped_libs[band] = {
            wy: ls.load_model_library(mp) for wy, mp in
            gen.resolve_dropped_year_manifest_paths(mlr_pred_dir, library_version, band).items()}
    return baseline_libs, dropped_libs, time.perf_counter() - t0


def run_historic_lookup(
    basin: str, water_year: int, bands: list[str],
    baseline_libs: dict, dropped_libs: dict,
    qa_file: str = "pillow_wy_1980_2025_qa6.nc",
    start_date: str | None = None, end_date: str | None = None,
    score_rule: str = "adj_r2", verbose: bool = True,
) -> dict:
    """
    Run the lookup-only daily loop over [start_date, end_date] (default: the whole WY,
    day index 0 skipped -- matches mlr_prediction_historic.py's own `range(1, n_timesteps)`
    convention) and return the five output tables plus timing.
    """
    log = print if verbose else (lambda *a, **k: None)

    t0 = time.perf_counter()
    P = replay_preprocessing(basin, water_year, qa_file, "wy", True)
    t_replay = time.perf_counter() - t0

    times = pd.to_datetime(P["obs_test_lst"][0].time.values)
    n_timesteps = P["n_timesteps"]
    if start_date or end_date:
        lo = pd.Timestamp(start_date) if start_date else times.min()
        hi = pd.Timestamp(end_date) if end_date else times.max()
        idxs = [i for i in range(1, n_timesteps) if lo <= times[i] <= hi]
    else:
        idxs = list(range(1, n_timesteps))
    log(f"  preprocessing replay: {t_replay:.2f}s; requesting {len(idxs)} of "
        f"{n_timesteps - 1} days ({times[idxs[0]].date()} .. {times[idxs[-1]].date()})")

    baseline_rows, summary_rows, hist_rows, station_rows, model_rows = [], [], [], [], []
    n_no_pillows = 0
    n_no_baseline = {b: 0 for b in bands}

    t0 = time.perf_counter()
    for t_idx in idxs:
        date_str = str(times[t_idx].date())
        cur_vals, all_pils_QA, _, _ = preprocessing.process_daily_qa(
            t_idx, P["obs_test_lst"], P["obs_test_lst"], P["baseline_pils"],
            P["exclude_pillows"], printOutput=False)
        if not all_pils_QA:
            n_no_pillows += 1
            baseline_rows.append({"date": date_str, **{b: np.nan for b in bands}})
            continue
        values = {p: float(cur_vals.iloc[0][p]) for p in all_pils_QA
                 if p in cur_vals.columns and pd.notna(cur_vals.iloc[0][p])}
        available = sorted(values)

        baseline_row = {"date": date_str}
        for band in bands:
            r = build_prediction_result(
                baseline_libs[band], dropped_libs[band], available, values,
                score_rule=score_rule, basin=basin, band=band, date=date_str)
            if r is None:
                baseline_row[band] = np.nan
                n_no_baseline[band] += 1
                continue

            baseline_row[band] = r.baseline_prediction_mm
            summary_rows.append(r.summary_dict())
            hist_rows.extend({"date": date_str, "band": band, **m.__dict__}
                             for m in r.historical_training_sensitivity.members)
            station_rows.extend({"date": date_str, "band": band, **m.__dict__}
                                for m in r.station_loss_sensitivity.members)

            wrow = baseline_libs[band].table.loc[
                baseline_libs[band].table["pillow_key"] == r.baseline_pillow_key].iloc[0]
            model_rows.append({
                "date": date_str, "band": band, "pillow_key": r.baseline_pillow_key,
                "n_predictors": r.baseline_n_predictors, "adj_r2": r.baseline_adj_r2,
                "rmse": r.baseline_rmse,
                "max_pillow_frac_imputed": float(wrow["max_pillow_frac_imputed"]),
                "mean_pillow_frac_imputed": float(wrow["mean_pillow_frac_imputed"]),
                "n_available_pillows": len(available),
            })
        baseline_rows.append(baseline_row)
    t_predict = time.perf_counter() - t0

    baseline_df = pd.DataFrame(baseline_rows).rename(columns={"total": "Basin"})
    out = dict(
        baseline=baseline_df,
        summary=pd.DataFrame(summary_rows),
        historical_training=pd.DataFrame(hist_rows),
        station_loss=pd.DataFrame(station_rows),
        selected_model=pd.DataFrame(model_rows),
        stats=dict(t_replay=t_replay, t_predict=t_predict, n_days_requested=len(idxs),
                  n_days_no_pillows=n_no_pillows, n_days_no_baseline=n_no_baseline,
                  date_min=str(times[idxs[0]].date()), date_max=str(times[idxs[-1]].date())),
    )
    return out


def _git_state(repo_root: Path) -> dict:
    def run(*a):
        try:
            return subprocess.run(a, cwd=repo_root, capture_output=True, text=True,
                                  timeout=15, check=True).stdout.strip()
        except Exception:
            return None
    status = run("git", "status", "--porcelain")
    return {"branch": run("git", "rev-parse", "--abbrev-ref", "HEAD"),
           "commit": run("git", "rev-parse", "HEAD"),
           "dirty": bool(status)}


def save_outputs(basin: str, water_year: int, results: dict, bands: list[str],
                 libraries_used: dict, *, run_version: str, library_version: str,
                 mode: str, score_rule: str, qa_file: str,
                 config_dir: str = "/home/rossamower/work/aso/configs/") -> Path:
    out_dir = Path(f"/home/rossamower/work/aso/data/mlr_prediction/{basin}/"
                   f"model_library/historic_lookup/{basin}_wy{water_year}")
    out_dir.mkdir(parents=True, exist_ok=True)

    results["baseline"].to_csv(out_dir / "baseline_mm.csv", index=False, float_format="%.6f")
    results["summary"].to_csv(out_dir / "sensitivity_summary.csv", index=False, float_format="%.6f")
    results["historical_training"].to_csv(out_dir / "historical_training_members.csv",
                                          index=False, float_format="%.6f")
    results["station_loss"].to_csv(out_dir / "station_loss_members.csv",
                                   index=False, float_format="%.6f")
    results["selected_model"].to_csv(out_dir / "selected_model.csv", index=False, float_format="%.6f")

    repo_root = Path(__file__).resolve().parents[2]
    git = _git_state(repo_root)
    manifest = {
        # prediction-run identity -- run_version/library_version are the two axes this
        # milestone exists to decouple: which run event this is, and which pretrained
        # library it consumed. Neither is inferred; both were required CLI arguments.
        "run_id": run_version, "run_type": "historic",
        "library_version": library_version,
        "basin": basin, "water_year": water_year, "mode": mode, "score_rule": score_rule,
        "qa_file": qa_file, "bands": bands,
        "built_at_utc": datetime.now(timezone.utc).isoformat(),
        "git": git,
        "libraries_used": libraries_used,
        "stats": results["stats"],
        "row_counts": {k: len(v) for k, v in results.items() if k != "stats"},
        "note": ("Lookup-only historic prediction path. No regression fitting, "
                "imputation, or cross-validation occurred. Parallel to, and does not "
                "modify, mlr_prediction_historic.py."),
    }
    manifest_path = out_dir / "run_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, default=str) + "\n")

    mlr_pred_dir = gen._mlr_pred_dir(basin, config_dir)
    gen.append_prediction_run_registry_row(
        mlr_pred_dir, run_version=run_version, run_type="historic", basin=basin,
        library_version=library_version, wy_or_date=str(water_year),
        status="completed", git_commit=git["commit"], git_dirty=git["dirty"],
        manifest_path=str(manifest_path))
    return out_dir


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("basin")
    ap.add_argument("water_year", type=int)
    ap.add_argument("--library-version", required=True,
                    help="e.g. lookup_lib_v1 -- required, never inferred")
    ap.add_argument("--run-version", required=True,
                    help="e.g. lookup_hist_v1 -- required, never auto-incremented")
    ap.add_argument("--mode", default="season", choices=["season"])
    ap.add_argument("--score-rule", default="adj_r2")
    ap.add_argument("--bands", default="", help="comma-separated subset; default = all")
    ap.add_argument("--qa-file", default="pillow_wy_1980_2025_qa6.nc")
    ap.add_argument("--start-date", default=None)
    ap.add_argument("--end-date", default=None)
    ap.add_argument("--config-dir", default="/home/rossamower/work/aso/configs/")
    ap.add_argument("--n-validate", type=int, default=8,
                    help="sampled (date, band) pairs re-checked against a direct "
                         "build_prediction_result() call")
    ap.add_argument("--compare-existing", action="store_true",
                    help="diagnostic-only comparison against the v7_ens predict-NaNs table")
    ap.add_argument("--prod-root", default="/home/rossamower/work/aso/data/mlr_prediction/"
                    "FRIANT/v7_ens/season/mm")
    ap.add_argument("--dry-run", action="store_true", help="do not write output tables")
    args = ap.parse_args()

    bands = ([b.strip() for b in args.bands.split(",") if b.strip()]
             or aso_band_labels(args.basin, args.config_dir))
    t_all = time.perf_counter()

    print(f"HISTORIC LOOKUP PREDICTION  {args.basin} WY{args.water_year}  "
          f"run={args.run_version} library={args.library_version}  "
          f"bands={bands}  qa={args.qa_file}")

    baseline_libs, dropped_libs, t_load = load_all_libraries(
        args.basin, args.library_version, bands, config_dir=args.config_dir)
    print(f"  loaded {len(bands)} baseline + "
          f"{sum(len(v) for v in dropped_libs.values())} dropped-year libraries "
          f"in {t_load:.2f}s  (library_version={args.library_version})")
    libraries_used = {
        band: {"baseline_frame_id": baseline_libs[band].manifest["frame"]["frame_id"],
              "dropped_years": sorted(dropped_libs[band])}
        for band in bands}

    results = run_historic_lookup(
        args.basin, args.water_year, bands, baseline_libs, dropped_libs,
        qa_file=args.qa_file, start_date=args.start_date, end_date=args.end_date,
        score_rule=args.score_rule)

    st = results["stats"]
    print(f"\n  {st['n_days_requested']} days requested "
          f"({st['date_min']} .. {st['date_max']})")
    print(f"  {st['n_days_no_pillows']} days skipped: 0 usable pillows at all")
    print(f"  days with no eligible baseline model, by band: {st['n_days_no_baseline']}")
    print(f"  prediction loop: {st['t_predict']:.2f}s "
          f"({1000 * st['t_predict'] / max(st['n_days_requested'], 1):.2f} ms/day, "
          f"{len(bands)} bands/day)")

    # ------------------------------------------------------------- validation
    print("\nVALIDATION")
    checks = []
    rng = np.random.default_rng(5)
    bdf = results["baseline"]
    band_cols = [c for c in bands if c in bdf.columns] + (["Basin"] if "total" in bands else [])
    valid_rows = bdf.dropna(subset=[c for c in band_cols if c != "Basin"], how="all")
    n_check = min(args.n_validate, len(valid_rows))
    sample_dates = rng.choice(valid_rows["date"].values, size=n_check, replace=False)

    t0 = time.perf_counter()
    P = replay_preprocessing(args.basin, args.water_year, args.qa_file, "wy", True)
    times = pd.to_datetime(P["obs_test_lst"][0].time.values)
    for date_str in sample_dates:
        t_idx = int(np.where(times.normalize() == pd.Timestamp(date_str))[0][0])
        cur_vals, all_pils_QA, _, _ = preprocessing.process_daily_qa(
            t_idx, P["obs_test_lst"], P["obs_test_lst"], P["baseline_pils"],
            P["exclude_pillows"], printOutput=False)
        values = {p: float(cur_vals.iloc[0][p]) for p in all_pils_QA
                 if p in cur_vals.columns and pd.notna(cur_vals.iloc[0][p])}
        available = sorted(values)
        for band in bands:
            direct = build_prediction_result(
                baseline_libs[band], dropped_libs[band], available, values,
                basin=args.basin, band=band, date=str(date_str))
            col = "Basin" if band == "total" else band
            table_val = bdf.loc[bdf["date"] == date_str, col].iloc[0]
            if direct is None:
                ok = pd.isna(table_val)
            else:
                ok = abs(direct.baseline_prediction_mm - table_val) < 1e-9
            checks.append((f"{band}/{date_str}: table baseline == direct "
                           f"build_prediction_result()", bool(ok),
                           f"table={table_val}  direct="
                           f"{direct.baseline_prediction_mm if direct else None}"))
    print(f"  re-derivation of {n_check} sampled dates x {len(bands)} bands "
          f"took {time.perf_counter() - t0:.2f}s")

    # no fitting/imputation in the prediction path -- structural, not runtime-observed:
    # run_historic_lookup only calls preprocessing.process_daily_qa() (QA/availability,
    # no regression) and build_prediction_result() (pure lookup). Confirmed via the AST of
    # this module's own imports, not a substring search -- a substring search would trip on
    # this very comment, which mentions the module by name.
    import ast
    tree = ast.parse(Path(__file__).read_text())
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
    checks.append(("no lm_model import anywhere in this module (no fitting/CV possible)",
                   "lm_model" not in imported, f"imports: {sorted(imported)}"))

    # every selected model uses only QA-available pillows, checked against the ACTUAL
    # daily availability recorded for each row, not just the sample above
    bad = 0
    for _, row in results["selected_model"].iterrows():
        date_str, band, key = row["date"], row["band"], row["pillow_key"]
        t_idx = int(np.where(times.normalize() == pd.Timestamp(date_str))[0][0])
        cur_vals, all_pils_QA, _, _ = preprocessing.process_daily_qa(
            t_idx, P["obs_test_lst"], P["obs_test_lst"], P["baseline_pils"],
            P["exclude_pillows"], printOutput=False)
        if not set(key.split(",")) <= set(all_pils_QA):
            bad += 1
    checks.append((f"every selected model uses only that day's QA-available pillows "
                   f"({len(results['selected_model'])} selections checked)",
                   bad == 0, f"{bad} violations"))

    # off-season sanity: at least one zero/near-zero baseline prediction should exist
    # (Oct/Sep) and no NEGATIVE predictions anywhere (predict_from_lookup clips at 0)
    numeric_bands = [c for c in band_cols]
    all_vals = bdf[numeric_bands].to_numpy(dtype=float)
    checks.append(("no negative baseline predictions anywhere",
                   bool(np.nanmin(all_vals) >= 0.0),
                   f"min = {np.nanmin(all_vals):.6f} mm"))
    off_season = bdf[pd.to_datetime(bdf["date"]).dt.month.isin([9, 10])]
    off_vals = off_season[numeric_bands].to_numpy(dtype=float) if len(off_season) else np.array([])
    checks.append(("off-season (Sep/Oct) predictions are near-zero, not spuriously large",
                   len(off_vals) == 0 or bool(np.nanmax(off_vals) < 50.0),
                   f"max off-season value = "
                   f"{np.nanmax(off_vals) if len(off_vals) else 'n/a'} mm"))

    # output coverage matches the requested date range
    req_dates = pd.date_range(st["date_min"], st["date_max"])
    got_dates = pd.to_datetime(bdf["date"]).sort_values().reset_index(drop=True)
    checks.append(("output date coverage matches the requested range exactly",
                   len(got_dates) == len(req_dates)
                   and (got_dates.values == req_dates.values).all(),
                   f"requested {len(req_dates)} days, produced {len(got_dates)} rows"))

    ok_all = all(p for _, p, _ in checks)
    for name, passed, detail in checks:
        print(f"  [{'PASS' if passed else 'FAIL'}] {name}")
        if not passed or "violations" in str(detail):
            print(f"         {detail}")
    print(f"  {sum(p for _, p, _ in checks)}/{len(checks)} checks passed")

    # ---------------------------------------------------------------- outputs
    out_dir = None
    if not args.dry_run:
        out_dir = save_outputs(
            args.basin, args.water_year, results, bands, libraries_used,
            run_version=args.run_version, library_version=args.library_version,
            mode=args.mode, score_rule=args.score_rule, qa_file=args.qa_file,
            config_dir=args.config_dir)
        print(f"\nOUTPUTS  {out_dir}")
        for f in sorted(out_dir.iterdir()):
            print(f"    {f.name:34s} {f.stat().st_size:>10,d} bytes")

    # -------------------------------------------------- diagnostic comparison
    if args.compare_existing:
        print("\nDIAGNOSTIC COMPARISON vs existing v7_ens predict-NaNs table "
              "(architecture is intentionally different -- NOT a pass/fail check)")
        prod_path = Path(args.prod_root) / f"prediction_mm_wy{args.water_year}_combination.csv"
        if not prod_path.exists():
            print(f"  {prod_path} not found; skipping")
        else:
            prod = pd.read_csv(prod_path)
            prod = prod[prod["Training Infer NaNs"] == "predict NaNs"].copy()
            prod["Date"] = pd.to_datetime(prod["Date"]).dt.date.astype(str)

            label_map = {b: l for b, l in zip(
                [b for b in bands if b != "total"],
                ["<7k", "7k-8k", "8k-9k", "9k-10k", "10k-11k", "11k-12k", ">12k"])}
            label_map_rev = {v: k for k, v in label_map.items()}

            # bdf and prod both literally have a column named 'Basin'. merge()'s default
            # suffixes only apply to overlapping NON-join columns, so both survive as
            # 'Basin_x'/'Basin_y' -- a naive `if "Basin" in merged.columns` check after the
            # merge would silently find nothing and skip the single most important band.
            # Prefixing every lookup-side band column before merging removes the ambiguity
            # outright rather than relying on pandas' suffixing to be checked correctly.
            bdf_pref = bdf.rename(columns={c: f"lookup_{c}" for c in band_cols})
            merged = bdf_pref.merge(prod, left_on="date", right_on="Date", how="inner")
            print(f"  {len(merged)} overlapping dates "
                  f"({merged['date'].min()} .. {merged['date'].max()})")

            rows = []
            for lookup_col, prod_col in [("Basin", "Basin")] + list(label_map.items()):
                lc = f"lookup_{lookup_col}"
                if lc not in merged.columns or prod_col not in merged.columns:
                    continue
                a = merged[lc].to_numpy(dtype=float)
                b = merged[prod_col].to_numpy(dtype=float)
                d = a - b
                finite = np.isfinite(d)
                if finite.sum() == 0:
                    continue
                rows.append({
                    "band": lookup_col, "n": int(finite.sum()),
                    "mean_diff_mm": float(np.nanmean(d[finite])),
                    "mean_abs_diff_mm": float(np.nanmean(np.abs(d[finite]))),
                    "max_abs_diff_mm": float(np.nanmax(np.abs(d[finite]))),
                    "corr": float(np.corrcoef(a[finite], b[finite])[0, 1])
                            if finite.sum() > 1 else float("nan"),
                })
            diff_df = pd.DataFrame(rows)
            print(diff_df.to_string(index=False))
            if out_dir is not None:
                diff_df.to_csv(out_dir / "diagnostic_vs_v7ens.csv", index=False)
                print(f"  wrote {out_dir / 'diagnostic_vs_v7ens.csv'}")

    print(f"\nRUNTIME  libraries {t_load:.2f}s + preprocessing replay "
          f"{results['stats']['t_replay']:.2f}s + prediction loop "
          f"{results['stats']['t_predict']:.2f}s  =  grand total "
          f"{time.perf_counter() - t_all:.2f}s")
    print("  " + ("ALL CHECKS PASSED" if ok_all else "*** CHECKS FAILED ***"))
    return 0 if ok_all else 1


if __name__ == "__main__":
    raise SystemExit(main())
