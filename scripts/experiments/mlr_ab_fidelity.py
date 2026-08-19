"""
mlr_ab_fidelity.py

Fidelity check for mlr_ab_harness.py: does the harness reproduce production output?

Without this, the harness could be self-consistently wrong -- deterministic, diffable, and
measuring something that is not what the pipeline actually produces. It already caught one
such gap (the harness was skipping postprocessing.arrange_prediction_tables, so it emitted
`Total` where production emits `Band Sum` / `Basin`).

Compares the harness's prediction_mm.csv against the corresponding production
prediction_mm_wy{YYYY}_combination.csv, joining on (Date, Training Infer NaNs) and
requiring EXACT equality on every elevation column.

mm is the comparison target, not acreFt: format_prediction_tables divides acreFt by 1000
and casts to int (postprocessing.py:85-86), so acreFt is quantized to thousands and would
hide differences up to 999 acre-feet.

Usage:
    python mlr_ab_fidelity.py FRIANT 1997 --label A
    python mlr_ab_fidelity.py FRIANT 2017 --label A --prod-root .../FRIANT/v6_pre
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

HARNESS_ROOT = "/home/rossamower/work/aso/data_test/mlr_ab/"
PROD_ROOT = "/home/rossamower/work/aso/data/mlr_prediction/FRIANT/v6_pre"
JOIN_COLS = ["Date", "Training Infer NaNs"]
META_COLS = ["Date", "Model Type", "Training Infer NaNs", "Prediction QA"]


def norm_date(s):
    return pd.to_datetime(s).dt.date.astype(str)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("aso_site_name")
    ap.add_argument("water_year", type=int)
    ap.add_argument("--label", default="A")
    ap.add_argument("--unit", default="mm", choices=["mm", "acreFt", "pillows"])
    ap.add_argument("--harness-root", default=HARNESS_ROOT)
    ap.add_argument("--prod-root", default=PROD_ROOT)
    ap.add_argument("--seasonal-dir", default="season")
    args = ap.parse_args()

    hp = (Path(args.harness_root) / f"{args.aso_site_name}_wy{args.water_year}"
          / args.label / f"prediction_{args.unit}.csv")
    pp = (Path(args.prod_root) / args.seasonal_dir / args.unit
          / f"prediction_{args.unit}_wy{args.water_year}_combination.csv")

    for p in (hp, pp):
        if not p.exists():
            raise SystemExit(f"missing: {p}")

    h, p = pd.read_csv(hp), pd.read_csv(pp)
    h["Date"], p["Date"] = norm_date(h["Date"]), norm_date(p["Date"])

    print(f"=== FIDELITY  {args.aso_site_name} wy{args.water_year} [{args.unit}] ===")
    print(f"  harness    : {hp}   ({len(h)} rows)")
    print(f"  production : {pp}   ({len(p)} rows)")

    hc, pc = set(h.columns), set(p.columns)
    if hc != pc:
        print(f"\n  COLUMN MISMATCH")
        print(f"    harness only    : {sorted(hc - pc)}")
        print(f"    production only : {sorted(pc - hc)}")
        raise SystemExit("FIDELITY FAIL (schema)")

    value_cols = [c for c in p.columns if c not in META_COLS]
    print(f"  comparing {len(value_cols)} value columns: {value_cols}\n")

    ok = bad = missing = 0
    for _, hr in h.iterrows():
        m = p
        for c in JOIN_COLS:
            m = m[m[c] == hr[c]]
        if m.empty:
            missing += 1
            print(f"  NO PROD ROW  {hr['Date']} [{hr['Training Infer NaNs']}]")
            continue
        pr = m.iloc[0]
        deltas = {c: float(hr[c]) - float(pr[c]) for c in value_cols}
        worst = max(deltas.values(), key=abs)
        if worst == 0:
            ok += 1
        else:
            bad += 1
            off = {c: round(d, 6) for c, d in deltas.items() if d != 0}
            print(f"  MISMATCH  {hr['Date']} [{hr['Training Infer NaNs']}]  "
                  f"max|delta|={abs(worst):g}")
            print(f"            {off}")

    print(f"\n  exact match : {ok}")
    print(f"  mismatch    : {bad}")
    print(f"  no prod row : {missing}")
    verdict = "FIDELITY PASS" if (bad == 0 and missing == 0 and ok > 0) else "FIDELITY FAIL"
    print(f"\n  {verdict}")
    sys.exit(0 if verdict == "FIDELITY PASS" else 1)


if __name__ == "__main__":
    main()
