"""
insitu_qa_2026.py

Side-by-side QA review of the WY2026 pillow record under two CDEC sensor policies.

Production (insitu_qa.py) fetches with sensor_priority=(82, 3) -- SWE_ADJ preferred, raw
SWE filling its gaps ELEMENT-WISE, so one station's series can silently mix sensors across
time. The decision (email, 2026-10) is to ingest sensor 82 only. This script redraws the
qa_viz_2026 figures with the 82-only series overlaid on the legacy blended series so each
pillow can be eyeballed and `exclude_pillows_ranges` updated accordingly.

READ-ONLY with respect to production. It writes PNGs to a new directory and nothing else:
no QA recompute, no NetCDF, no CSV, no checklist. The automated QA flags are REUSED from
the existing insitu_qa_simple_wy_2026.csv -- that is the point, since the question is how
the 82-only series sits against the verdicts production already reached.

The two series end on different dates (legacy ~2026-08-18, 82-only 2026-10-01). That is
expected and left visible: by mid-August most pillows read zero anyway.

Usage (conda env: bor -- see env note at the bottom):
    python insitu_qa_2026.py                # USCASJ
    python insitu_qa_2026.py USCASJ --pillows KUP STL BCB
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
import xarray as xr

sys.path.insert(1, str(Path(__file__).resolve().parent.parent))

import metadata as metadata
import plotting as plotting


# the runtime config tree, NOT the repo's config/regions/. insitu_qa.py reads this same
# path; the repo copy has drifted and carries no exclude_pillows_ranges.
CONFIG_DIR = "/home/rossamower/work/aso/configs/"
OUT_SUBDIR = "qa_viz_2026_daily_v_82"
QA_LABEL = "Raw SWE (82)"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("aso_site_name", nargs="?", default="USCASJ")
    ap.add_argument("--config-dir", default=CONFIG_DIR)
    ap.add_argument("--pillows", nargs="+", default=None,
                    help="subset to redraw (default: all)")
    ap.add_argument("--ymax-quantile", type=float, default=0.999,
                    help="outlier-robust upper y-bound for the grid figure; "
                         "pass 1.0 for the legacy true-max behaviour")
    args = ap.parse_args()

    cfg = metadata.load_yaml(Path(f"{args.config_dir}regions/{args.aso_site_name}.yaml"))
    insitu_dir = cfg["data_filepaths"]["insitu_dir"]

    legacy_f = f"{insitu_dir}raw/{args.aso_site_name}_insitu_obs_daily_wy_2026.nc"
    sensor82_f = f"{insitu_dir}sensor82/pillow_wy_2026_sensor82.nc"
    flags_f = f"{insitu_dir}qa/insitu_qa_simple_wy_2026.csv"
    out_dir = f"{insitu_dir}qa/{OUT_SUBDIR}/"

    # unmasked on purpose: the raw line must be drawn THROUGH manual exclusion windows so
    # the window can be judged against what it is hiding.
    ds_legacy = xr.load_dataset(legacy_f)
    ds_82 = xr.load_dataset(sensor82_f)
    df_simple = pd.read_csv(flags_f)

    # manual windows come from the YAML, exactly as insitu_qa.py builds them.
    manual_windows_per_pillow = {}
    for r in cfg["pillow_api"].get("exclude_pillows_ranges") or []:
        manual_windows_per_pillow.setdefault(r["pillow"], []).append((r["start"], r["end"]))

    pillows = args.pillows or list(ds_legacy.data_vars)
    missing = [p for p in pillows if p not in ds_82.data_vars]
    if missing:
        print(f"WARNING: absent from the 82-only file, legacy line only: {', '.join(missing)}")

    print(f"INSITU QA 2026 -- blended vs sensor-82: {args.aso_site_name}")
    print(f"  legacy   : {legacy_f}")
    print(f"             {str(ds_legacy.time.values.min())[:10]} .. {str(ds_legacy.time.values.max())[:10]}")
    print(f"  82-only  : {sensor82_f}")
    print(f"             {str(ds_82.time.values.min())[:10]} .. {str(ds_82.time.values.max())[:10]}")
    print(f"  flags    : {flags_f} ({len(df_simple)} rows, REUSED -- no QA recompute)")
    print(f"  out      : {out_dir}")
    print(f"  pillows  : {len(pillows)}; manually excluded: {len(manual_windows_per_pillow)}\n")

    plotting.plot_all_pillows_qa_timeline(
        ds_raw=ds_legacy,
        df_simple=df_simple,
        ds_qa=ds_82,
        qa_label=QA_LABEL,
        ymax_quantile=args.ymax_quantile,
        start="2025-10-01",
        end=None,
        manual_windows_per_pillow=manual_windows_per_pillow,
        pillows=pillows,
        saveFIG=True,
        figDIR=out_dir,
        figFNAME="all_pils.png",
        dpi=300,
    )
    print("wrote all_pils.png")

    for pil in pillows:
        plotting.plot_pillow_qa_timeline(
            ds_raw=ds_legacy,
            pillow=pil,
            df_simple=df_simple,
            ds_qa=ds_82 if pil in ds_82.data_vars else None,
            qa_label=QA_LABEL,
            start="2025-10-01",
            end=None,
            manual_windows=manual_windows_per_pillow.get(pil, []),
            saveFIG=True,
            figDIR=out_dir,
        )
        print(f"  wrote {pil}.png")

    print(f"\nDONE. {len(pillows) + 1} figure(s) in {out_dir}")


if __name__ == "__main__":
    main()

# env note: `conda activate bor` alone is not enough -- `python` still resolves to base,
# whose /opt/miniforge3 proj.db is at DATABASE.LAYOUT.VERSION.MINOR=3 where pyproj wants
# >=4. Activate AND call the explicit interpreter:
#   source /opt/miniforge3/etc/profile.d/conda.sh && conda activate bor
#   /home/rossamower/work/aso/conda/envs/bor/bin/python insitu_qa_2026.py
