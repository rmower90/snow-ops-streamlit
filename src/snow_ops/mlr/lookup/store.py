"""
Persistence for the pretrained model library.

Parquet is the source of truth: it keeps `pillow_mask` an integer, unused `coef_i` NULL
rather than the empty string, and float64 exact. The CSV export is a debug convenience and
is never read back -- a float round-trip through CSV is exactly the precision problem the
A/B tooling had to fight, and the same round-trip inside lm_model's imputation cache was
already caught perturbing observed values by ~1 ULP.

The library manifest embeds the canonical frame's manifest whole, under `frame`, so a
library is never separated from the provenance of the matrix it was fitted on.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from snow_ops.mlr.lookup.fit import METRIC_CONVENTIONS, SCORE_RULES
from snow_ops.mlr.lookup.query import ModelLibrary

LIBRARY_SCHEMA_VERSION = "model_library_v1"


def library_paths(frame_manifest: dict) -> dict[str, Path]:
    root = Path(frame_manifest["paths"]["root"])
    lib_id = frame_manifest["frame_id"]
    return {
        "root": root,
        "library": root / "libraries" / f"{lib_id}_models.parquet",
        "library_csv": root / "libraries" / f"{lib_id}_models_top.csv",
        "manifest": root / "manifests" / f"{lib_id}_models.json",
    }


def build_library_manifest(table: pd.DataFrame, frame_manifest: dict,
                           *, max_pillows: int, build_seconds: float | None = None) -> dict:
    return {
        "library_schema_version": LIBRARY_SCHEMA_VERSION,
        "library_id": frame_manifest["frame_id"],
        "built_at_utc": datetime.now(timezone.utc).isoformat(),
        "build_seconds": build_seconds,
        "n_models": int(len(table)),
        "max_pillows": max_pillows,
        "models_per_k": {str(k): int(v) for k, v in
                         table["n_predictors"].value_counts().sort_index().items()},
        # how the models were fitted -- read together with `frame`
        "estimator": "sklearn.linear_model.LinearRegression",
        "fit_intercept": frame_manifest["fit_intercept"],
        "add_zero_aso": frame_manifest["add_zero_aso"],
        "stats_exclude_synthetic_row": True,
        "fit_n_obs": frame_manifest["fit_n_obs"],
        "stats_n_obs": frame_manifest["stats_n_obs"],
        # selection policy
        "default_score_rule": frame_manifest["default_score_rule"],
        "score_is_in_sample": True,
        "score_intended_use": "model_selection",
        "score_rebuild_required_to_change": False,
        "available_score_rules": {k: ("higher_is_better" if v else "lower_is_better")
                                  for k, v in sorted(SCORE_RULES.items())},
        "tie_break": frame_manifest["tie_break"],
        "metric_conventions": METRIC_CONVENTIONS,
        # eligibility
        "bitmask_bit_order": frame_manifest["bitmask_bit_order"],
        "pillow_mask_dtype": "int64",
        "eligibility_test": "(pillow_mask & ~available_mask) == 0",
        # conditioning summary, so a pathological table is visible without a query
        "cond_number_max": float(table["cond_number"].max()),
        "cond_number_p99": float(table["cond_number"].quantile(0.99)),
        # the frame this was fitted on, verbatim
        "frame": frame_manifest,
        "paths": {k: str(v) for k, v in library_paths(frame_manifest).items()},
    }


def save_model_library(table: pd.DataFrame, manifest: dict,
                       *, csv_top_n: int = 200) -> dict[str, Path]:
    paths = {k: Path(v) for k, v in manifest["paths"].items()}
    paths["library"].parent.mkdir(parents=True, exist_ok=True)
    paths["manifest"].parent.mkdir(parents=True, exist_ok=True)

    table.to_parquet(paths["library"], index=False)
    written = {"library": paths["library"]}

    if csv_top_n:
        from snow_ops.mlr.lookup.fit import rank_frame
        top = rank_frame(table, manifest["default_score_rule"]).head(csv_top_n)
        top.to_csv(paths["library_csv"], index=False, float_format="%.17g")
        written["library_csv"] = paths["library_csv"]

    paths["manifest"].write_text(
        json.dumps(manifest, indent=2, sort_keys=True, default=str) + "\n")
    written["manifest"] = paths["manifest"]
    return written


def load_model_library(manifest_path: str | Path) -> ModelLibrary:
    manifest = json.loads(Path(manifest_path).read_text())
    table = pd.read_parquet(manifest["paths"]["library"])
    if len(table) != manifest["n_models"]:
        raise RuntimeError(f"library has {len(table)} rows, manifest says "
                           f"{manifest['n_models']}")
    return ModelLibrary(table=table, manifest=manifest)
