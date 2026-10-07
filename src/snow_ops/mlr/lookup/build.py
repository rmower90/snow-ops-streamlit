"""
The one reusable unit both build_all_bands.py and build_leave_one_year_libraries.py already
perform per (band[, excluded_year]): build a canonical frame, fit its model library, and
optionally persist both.

This module changes NOTHING about modelling behaviour -- it is a pure extraction of the
four calls (build_canonical_frame, build_model_library, build_library_manifest, save_*)
those two scripts already made inline, so that a third caller (the library-generation
orchestrator) can invoke the identical unit without re-deriving it, and without those two
scripts having to import each other.
"""

from __future__ import annotations

import time
from typing import Iterable

from snow_ops.mlr.lookup import fit as lf
from snow_ops.mlr.lookup import frame as cf
from snow_ops.mlr.lookup import store as ls
from snow_ops.mlr.lookup.query import ModelLibrary


def build_and_save_one(
    basin: str,
    *,
    elev_band: str,
    mode: str,
    obs_qa_file: str,
    library_id: str,
    excluded_years: Iterable[int] = (),
    config_dir: str,
    max_pillows: int = cf.MAX_PILLOWS,
    obs_threshold: float = cf.DEFAULT_OBS_THRESHOLD,
    rebuild_imputation: bool = False,
    save: bool = True,
    verbose: bool = False,
) -> dict:
    """
    Build (and, unless save=False, persist) the canonical frame + fitted model library for
    one (basin, elev_band, mode, library_id). Identical to what build_all_bands.py and
    build_leave_one_year_libraries.py already do inline; extracted here so a third caller
    can reuse the exact same unit.

    Returns a dict with the frame, table, lib_manifest, ModelLibrary, written paths (if
    saved) and timings -- everything the two existing scripts' validation/reporting code
    already consumes.
    """
    t0 = time.perf_counter()
    frame = cf.build_canonical_frame(
        basin, elev_band=elev_band, mode=mode, obs_qa_file=obs_qa_file,
        obs_threshold=obs_threshold, excluded_years=excluded_years,
        library_id=library_id, config_dir=config_dir,
        rebuild_imputation=rebuild_imputation, verbose=verbose)
    t_frame = time.perf_counter() - t0

    t0 = time.perf_counter()
    table = lf.build_model_library(frame, max_pillows=max_pillows, verbose=verbose)
    t_fit = time.perf_counter() - t0

    lib_manifest = ls.build_library_manifest(
        table, frame.manifest, max_pillows=frame.manifest["max_pillows"],
        build_seconds=t_fit)
    lib = ModelLibrary(table=table, manifest=lib_manifest)

    written = ls.save_model_library(table, lib_manifest) if save else None
    if save:
        cf.save_canonical_frame(frame)

    return dict(frame=frame, table=table, lib_manifest=lib_manifest, lib=lib,
               written=written, t_frame=t_frame, t_fit=t_fit)
