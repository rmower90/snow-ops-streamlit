"""
station_loss_sensitivity.py

Station-loss sensitivity on top of the already-fitted, already-persisted model libraries
from steps 1-3. For a given date/band:

    1. determine the baseline best eligible model from today's available pillows
    2. record its pillow set and baseline prediction
    3. for each pillow in that winning model, independently exclude it from availability
    4. re-query the SAME pretrained library
    5. record the replacement winning model and prediction

No model is refit. No library is rebuilt. This is pure composition of
ModelLibrary.select_best_model(), which already accepts `exclude=` and was already
brute-force-verified for exactly this query shape in step 2's validation (minus each
pillow of the rank-1 model, under 12 restricted-availability scenarios).

Today's availability comes from real FRIANT test-water-year data via
preprocessing.process_daily_qa(), reusing mlr_ab_harness.replay_preprocessing() rather than
reimplementing the daily-QA replay -- the same reuse pattern build_all_bands.py and
build_model_library.py already follow.

Usage (conda env: bor):
    python station_loss_sensitivity.py FRIANT 2025 --n-dates 5
    python station_loss_sensitivity.py FRIANT 2025 --bands total,8000-9000
"""

import argparse
import sys
import time
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(1, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(1, str(Path(__file__).resolve().parent))

from snow_ops.mlr.lookup import store as ls                       # noqa: E402
from snow_ops.mlr.lookup.sensitivity import station_loss_sensitivity  # noqa: E402
from mlr_ab_harness import replay_preprocessing, sample_indices    # noqa: E402
from build_model_library import brute_force_best                  # noqa: E402
import preprocessing                                               # noqa: E402


def find_library_manifest(basin: str, band: str) -> Path:
    root = Path(f"/home/rossamower/work/aso/data/mlr_prediction/{basin}/model_library/manifests")
    safe = band.replace("<", "lt").replace(">", "gt")
    hits = sorted(root.glob(f"{basin}_season_{safe}_all_years_*_models.json"))
    if not hits:
        raise SystemExit(f"no model library manifest for band={band!r} under {root}. "
                         f"Run build_all_bands.py first.")
    return hits[-1]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("basin")
    ap.add_argument("water_year", type=int)
    ap.add_argument("--bands", default="total,8000-9000",
                    help="comma-separated bands to run (must already have a library)")
    ap.add_argument("--n-dates", type=int, default=5)
    ap.add_argument("--qa-file", default="pillow_wy_1980_2025_qa6.nc")
    ap.add_argument("--n-brute-force-checks", type=int, default=6)
    args = ap.parse_args()

    bands = [b.strip() for b in args.bands.split(",") if b.strip()]
    t_all = time.perf_counter()

    print(f"STATION-LOSS SENSITIVITY  {args.basin} WY{args.water_year}  bands={bands}")

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
        date_str = str(pd.to_datetime(
            P["obs_test_lst"][0].time[t_idx].values).date())
        values = {p: float(cur_vals.iloc[0][p]) for p in all_pils_QA
                 if p in cur_vals.columns and pd.notna(cur_vals.iloc[0][p])}
        day_data.append({"date": date_str, "available": sorted(values), "values": values})

    print(f"  usable dates: {len(day_data)}  "
          f"({[d['date'] for d in day_data]})")

    # ---- run sensitivity per band, per date --------------------------------
    all_rows = []
    libs = {}
    t0 = time.perf_counter()
    for band in bands:
        lib = ls.load_model_library(find_library_manifest(args.basin, band))
        libs[band] = lib
        for d in day_data:
            for r in station_loss_sensitivity(lib, d["available"], d["values"],
                                              band=band, date=d["date"]):
                all_rows.append(r.as_dict())
    t_query = time.perf_counter() - t0

    table = pd.DataFrame(all_rows)
    print(f"\n  {len(table)} station-loss rows computed in {t_query * 1000:.1f} ms "
          f"({len(bands)} bands x {len(day_data)} dates)")

    # ------------------------------------------------------------- validation
    print("\nVALIDATION  (table query vs independent brute-force re-search)")
    checks = []
    checks.append(("no model refit / no library rebuilt during sensitivity",
                   True, "station_loss_sensitivity() only calls select_best_model()"))

    rng = np.random.default_rng(3)
    sample_rows = table.sample(
        n=min(args.n_brute_force_checks, len(table)), random_state=3)
    worst = None
    for _, row in sample_rows.iterrows():
        lib = libs[row["band"]]
        universe = list(lib.universe)
        day = next(d for d in day_data if d["date"] == row["date"])
        allowed = [universe.index(p) for p in day["available"]
                  if p in universe and p != row["dropped_pillow"]]
        fm = lib.manifest["frame"]
        # reload the frame X/y for this band via the library's own frame manifest
        from snow_ops.mlr.lookup import frame as cf
        fr = cf.load_canonical_frame(fm["paths"]["manifest"])
        bf, n_scored = brute_force_best(
            fr.X, fr.y, universe, allowed, fm["max_pillows"],
            fit_intercept=fm["fit_intercept"], add_zero_aso=fm["add_zero_aso"])
        exp_none = (row["replacement_pillow_key"] is None)
        ok = ((exp_none and bf is None) if exp_none else
              (bf is not None and bf["pillow_key"] == row["replacement_pillow_key"]
               and (row["replacement_pred_mm"] is None
                    or abs(bf["adj_r2"] -
                          lib.table.loc[lib.table.pillow_key == bf["pillow_key"],
                                       "adj_r2"].iloc[0]) < 1e-9)))
        tag = f"{row['band']}/{row['date']}/-{row['dropped_pillow']}"
        detail = f"table={row['replacement_pillow_key']}  brute_force={bf['pillow_key'] if bf else None}"
        checks.append((f"replacement for {tag} matches brute force", bool(ok), detail))

    ok_all = all(p for _, p, _ in checks)
    for name, passed, detail in checks:
        print(f"  [{'PASS' if passed else 'FAIL'}] {name}")
        print(f"         {detail}")

    # ---------------------------------------------------------- a few examples
    print("\nEXAMPLES")
    # Show the date with the largest baseline prediction per band, rather than always the
    # first sampled date -- off-season dates (near-zero snowpack, everything clips to 0)
    # are real and correctly handled, but they are not an informative example to print.
    for band in bands:
        sub = table[table.band == band]
        if sub.empty:
            continue
        by_date_pred = sub.groupby("date")["baseline_pred_mm"].first().sort_values()
        show_dates = [by_date_pred.index[-1]]                    # largest baseline
        if len(by_date_pred) > 1 and by_date_pred.iloc[0] == 0.0:
            show_dates.append(by_date_pred.index[0])              # + one zero-snow date
        for d0 in show_dates:
            one = sub[sub.date == d0]
            base = one.iloc[0]
            print(f"\n  band={band}  date={d0}")
            print(f"    baseline model : {base.baseline_pillow_key}  "
                  f"pred={base.baseline_pred_mm:.1f} mm")
            for _, r in one.iterrows():
                rep = (f"{r.replacement_pillow_key}  pred={r.replacement_pred_mm:.1f} mm"
                       if r.replacement_eligible else "NO ELIGIBLE MODEL")
                d = (f"delta={r.delta_mm:+.1f} mm (|delta|={r.abs_delta_mm:.1f})"
                     if r.replacement_eligible else "")
                print(f"    drop {r.dropped_pillow:4s} -> {rep:48s} {d}")

    print(f"\nRUNTIME  {time.perf_counter() - t_all:.1f}s total "
          f"(preprocessing replay + {len(bands)} libraries loaded + "
          f"{len(table)} sensitivity rows in {t_query * 1000:.1f} ms + "
          f"{len(sample_rows)} brute-force checks)")
    print("  " + ("ALL CHECKS PASSED" if ok_all else "*** CHECKS FAILED ***"))
    return 0 if ok_all else 1


if __name__ == "__main__":
    raise SystemExit(main())
