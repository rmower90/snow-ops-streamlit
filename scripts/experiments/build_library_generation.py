"""
build_library_generation.py

Cut (or verify) one named library generation -- lookup_lib_v1, lookup_lib_v2, ... -- as an
immutable, human-addressable index over a fixed set of already-content-addressed canonical
frames + fitted model libraries.

Default behaviour (no --rebuild) NEVER fits a model. It recomputes each unit's deterministic
frame_id (cheap: build_canonical_frame() hits the imputation cache) and requires a matching,
already-fitted library to exist on disk at the exact path store.library_paths() predicts.
Missing units are reported as problems; nothing is silently built to paper over them. This
is how lookup_lib_v1 is meant to be cut over the artifacts already validated in earlier
milestones -- see --dry-run below.

--rebuild actually fits any missing unit via build_and_save_one() (the same unit
build_all_bands.py / build_leave_one_year_libraries.py already use) -- the path a future
lookup_lib_v2 with a genuinely different configuration would take.

Usage (conda env: bor):
    # verify + cut lookup_lib_v1 over already-validated FRIANT artifacts, no rebuild
    python build_library_generation.py FRIANT lookup_lib_v1 --dry-run
    python build_library_generation.py FRIANT lookup_lib_v1

    # a future generation with a changed configuration would look like:
    python build_library_generation.py FRIANT lookup_lib_v2 --qa-file pillow_wy_1980_2025_qa7.nc --rebuild
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(1, str(Path(__file__).resolve().parents[2] / "src"))

from snow_ops.mlr.lookup import frame as cf                # noqa: E402
from snow_ops.mlr.lookup import generation as gen           # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("basin")
    ap.add_argument("library_version", help="e.g. lookup_lib_v1 -- required, never inferred")
    ap.add_argument("--mode", default="season", choices=["season"])
    ap.add_argument("--qa-file", default="pillow_wy_1980_2025_qa6.nc")
    ap.add_argument("--bands", default="", help="comma-separated subset; default = all")
    ap.add_argument("--config-dir", default="/home/rossamower/work/aso/configs/")
    ap.add_argument("--obs-threshold", type=float, default=cf.DEFAULT_OBS_THRESHOLD)
    ap.add_argument("--max-pillows", type=int, default=cf.MAX_PILLOWS)
    ap.add_argument("--rebuild", action="store_true",
                    help="fit any missing unit rather than requiring it already exist")
    ap.add_argument("--rebuild-imputation", action="store_true")
    ap.add_argument("--dry-run", action="store_true",
                    help="assemble and print the manifest; do not write it or the registry")
    args = ap.parse_args()

    bands = [b.strip() for b in args.bands.split(",") if b.strip()] or None

    print(f"LIBRARY GENERATION  {args.library_version}  basin={args.basin} "
          f"mode={args.mode} qa={Path(args.qa_file).stem.split('_')[-1]}  "
          f"rebuild={args.rebuild}")

    manifest, problems = gen.build_generation_manifest(
        args.basin, args.library_version, mode=args.mode, obs_qa_file=args.qa_file,
        config_dir=args.config_dir, bands=bands, obs_threshold=args.obs_threshold,
        max_pillows=args.max_pillows, rebuild=args.rebuild,
        rebuild_imputation=args.rebuild_imputation, verbose=True)

    print(f"\n  units: {manifest['n_units_total']} / {manifest['n_units_expected']} "
          f"({manifest['n_bands']} bands x {manifest['n_units_per_band']} units/band)")
    print(f"  convention: {manifest['convention']}")

    if problems:
        print(f"\n  {len(problems)} PROBLEM(S) -- generation is INCOMPLETE:")
        for p in problems:
            print(f"    - {p}")
    else:
        print("\n  complete: every expected unit found, one consistent convention")

    if args.dry_run:
        print("\n  --dry-run: manifest not written, registry not updated")
        return 0 if not problems else 1

    if problems:
        print("\n  refusing to save an incomplete generation manifest. "
             "Re-run with --rebuild, or build the missing units first.")
        return 1

    repo_root = Path(__file__).resolve().parents[2]
    manifest = gen.finalize_generation_manifest(manifest, repo_root)
    mlr_pred_dir = gen._mlr_pred_dir(args.basin, args.config_dir)
    written = gen.save_generation_manifest(manifest, mlr_pred_dir)
    reg = gen.append_library_registry_row(mlr_pred_dir, manifest)

    print(f"\n  wrote {written}")
    print(f"  appended {reg}")
    print(f"  git: {manifest['git']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
