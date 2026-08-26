"""
Library GENERATIONS: an immutable, human-addressable name (`lookup_lib_v1`, `lookup_lib_v2`,
...) for one exact, content-addressed bundle of per-band canonical frames + fitted model
libraries.

WHY THIS EXISTS

Every artifact this branch has built so far is already content-addressed: `frame_id`
hashes every input that defines a canonical frame (ASO file, pillow QA file, threshold,
excluded years, schema version), so the SAME configuration always reproduces the SAME
frame_id (verified deterministic in the step-1/step-2 milestones). That makes frame_id
trustworthy but not memorable -- nobody submits a historic sweep against
"FRIANT_season_total_all_years_qa6_9f7668a0". A generation is nothing more than a small,
human-named INDEX over a fixed set of those hashes: it duplicates no model data, and
building it never re-fits a single model.

A generation manifest is written ONCE per version and is then immutable. If the underlying
frame_ids it points to still exist unchanged, the generation is still valid; there is no
"latest" resolution step here on purpose -- see resolve_library_manifest_path().

NO IMPLICIT "LATEST"

Every resolver in this module takes an explicit `library_version` string and does exactly
one thing: build the deterministic path `generations/{library_version}/generation_manifest.
json` and read it. No globbing, no directory listing, no lexicographic sort over content
hashes. This replaces the old find_manifest()/load_dropped_year_libraries() behaviour
(sorted glob, take the last match), which had no way to distinguish "newest" from
"alphabetically last" once more than one variant existed for a given (band, library_id).

Consistency, not re-fitting, is the safety check when CUTTING a generation from artifacts
that already exist on disk (the normal case -- see build_generation_manifest(rebuild=
False)): build_canonical_frame() is cheap when its imputation cache already hits (~1-2s),
so it is used to recompute each unit's frame_id and verify a matching, already-fitted
library is on disk at the exact deterministic path store.library_paths() predicts.
build_model_library() -- the ~30-40s-per-unit fitting step -- is never called in that path.
"""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from snow_ops.mlr.lookup import frame as cf
from snow_ops.mlr.lookup import store as ls
from snow_ops.mlr.lookup.build import build_and_save_one

GENERATION_SCHEMA_VERSION = "generation_manifest_v1"

# Fields that define a generation's regression/selection conventions. These are NOT part
# of frame_id's hash (see the module docstring in frame.py), so two units with the same
# data-relevant hash could in principle disagree on these if built with different
# arguments -- checked explicitly here rather than assumed from the hash matching.
_CONVENTION_FIELDS = ("fit_intercept", "add_zero_aso", "max_pillows",
                     "default_score_rule", "obs_threshold", "qa_tag")


def _mlr_pred_dir(basin: str, config_dir: str) -> Path:
    import metadata
    cfg = metadata.load_yaml(Path(config_dir) / "regions" / f"{basin}.yaml")
    return Path(cfg["data_filepaths"]["mlrPred_dir"])


def generation_root(mlr_pred_dir: Path) -> Path:
    return Path(mlr_pred_dir) / "model_library" / "generations"


def generation_dir(mlr_pred_dir: Path, library_version: str) -> Path:
    return generation_root(mlr_pred_dir) / library_version


def generation_manifest_path(mlr_pred_dir: Path, library_version: str) -> Path:
    return generation_dir(mlr_pred_dir, library_version) / "generation_manifest.json"


def expected_units(water_years: Iterable[int]) -> list[tuple[str, tuple[int, ...]]]:
    """[("all_years", ()), ("drop_wy2017", (2017,)), ...] -- the library_id/excluded_years
    pairs one generation is expected to bundle per band."""
    units = [("all_years", ())]
    for wy in sorted(water_years):
        units.append((f"drop_wy{wy}", (wy,)))
    return units


# ===========================================================================
# locating a single already-built unit, without fitting
# ===========================================================================

def locate_existing_unit(
    basin: str, *, elev_band: str, mode: str, obs_qa_file: str,
    library_id: str, excluded_years: Iterable[int], config_dir: str,
    obs_threshold: float = cf.DEFAULT_OBS_THRESHOLD, max_pillows: int = cf.MAX_PILLOWS,
) -> dict:
    """
    Recompute this unit's deterministic frame_id (cheap: build_canonical_frame() hits the
    imputation cache if it already exists) and check whether a fitted library already
    exists at the exact path store.library_paths() predicts for that frame_id.

    Returns {"status": "found", "frame_manifest": ..., "library_manifest_path": ...,
    "library_manifest": ...} or {"status": "missing", "frame_id": ..., "expected_path": ...}
    -- never raises for a missing artifact, since "not built yet" is an expected outcome
    when deciding whether --rebuild is required.
    """
    frame = cf.build_canonical_frame(
        basin, elev_band=elev_band, mode=mode, obs_qa_file=obs_qa_file,
        obs_threshold=obs_threshold, excluded_years=excluded_years,
        library_id=library_id, config_dir=config_dir, verbose=False)
    fm = frame.manifest
    paths = ls.library_paths(fm)

    if not (paths["library"].exists() and paths["manifest"].exists()):
        return {"status": "missing", "frame_id": fm["frame_id"],
               "expected_library_path": paths["library"],
               "expected_manifest_path": paths["manifest"]}

    lib_manifest = json.loads(paths["manifest"].read_text())
    if lib_manifest.get("n_models") is None:
        return {"status": "missing", "frame_id": fm["frame_id"],
               "expected_library_path": paths["library"],
               "expected_manifest_path": paths["manifest"]}

    return {"status": "found", "frame_manifest": fm,
           "library_manifest_path": paths["manifest"],
           "library_manifest": lib_manifest}


# ===========================================================================
# building a generation manifest
# ===========================================================================

def build_generation_manifest(
    basin: str, library_version: str, *,
    mode: str = "season", obs_qa_file: str = "pillow_wy_1980_2025_qa6.nc",
    config_dir: str = "/home/rossamower/work/aso/configs/",
    bands: list[str] | None = None,
    obs_threshold: float = cf.DEFAULT_OBS_THRESHOLD, max_pillows: int = cf.MAX_PILLOWS,
    rebuild: bool = False, rebuild_imputation: bool = False,
    verbose: bool = True,
) -> tuple[dict, list[str]]:
    """
    Assemble (but do not yet save) the generation manifest for `library_version`.

    rebuild=False (the default, and what cutting lookup_lib_v1 over already-validated
    artifacts uses): every unit must already exist on disk; build_model_library() is never
    called. Missing units are reported as problems, not built.

    rebuild=True: missing (or, if rebuild_imputation, all) units are actually built via
    build_and_save_one() -- this is the path a future lookup_lib_v2 with a changed
    configuration would use.

    Returns (manifest, problems). problems is empty iff the generation is complete and
    internally consistent; the caller decides whether to save on that basis.
    """
    log = print if verbose else (lambda *a, **k: None)

    all_bands = cf.aso_band_labels(basin, config_dir)
    bands = bands or all_bands
    water_years = cf.full_history_water_years(basin, config_dir)
    units = expected_units(water_years)

    members: dict[str, dict] = {}
    problems: list[str] = []
    convention: dict | None = None
    qa_tag = None

    for band in bands:
        members[band] = {}
        for library_id, excl in units:
            found = locate_existing_unit(
                basin, elev_band=band, mode=mode, obs_qa_file=obs_qa_file,
                library_id=library_id, excluded_years=excl, config_dir=config_dir,
                obs_threshold=obs_threshold, max_pillows=max_pillows)

            if found["status"] == "missing":
                if not rebuild:
                    problems.append(
                        f"{band}/{library_id}: not built (expected frame_id="
                        f"{found['frame_id']}). Re-run with --rebuild to fit it, or "
                        f"build it via build_all_bands.py / build_leave_one_year_"
                        f"libraries.py first.")
                    log(f"  [MISSING] {band}/{library_id}  frame_id={found['frame_id']}")
                    continue
                log(f"  [BUILD] {band}/{library_id}  (rebuild=True)")
                result = build_and_save_one(
                    basin, elev_band=band, mode=mode, obs_qa_file=obs_qa_file,
                    library_id=library_id, excluded_years=excl, config_dir=config_dir,
                    obs_threshold=obs_threshold, max_pillows=max_pillows,
                    rebuild_imputation=rebuild_imputation, save=True, verbose=False)
                fm, lm = result["frame"].manifest, result["lib_manifest"]
            else:
                fm, lm = found["frame_manifest"], found["library_manifest"]
                log(f"  [OK]      {band}/{library_id}  frame_id={fm['frame_id']}")

            # fit_intercept/add_zero_aso/max_pillows/default_score_rule/obs_threshold/
            # qa_tag all live on the frame manifest.
            this_convention = {k: fm.get(k, lm.get(k)) for k in _CONVENTION_FIELDS}
            if convention is None:
                convention = this_convention
                qa_tag = fm["qa_tag"]
            elif this_convention != convention:
                problems.append(
                    f"{band}/{library_id}: convention mismatch -- expected "
                    f"{convention}, found {this_convention}. Generations cannot mix "
                    f"regression conventions across bands/years.")

            members[band][library_id] = {
                "frame_id": fm["frame_id"],
                "excluded_years": list(excl),
                "frame_manifest_path": fm["paths"]["manifest"],
                "library_manifest_path": str(
                    ls.library_paths(fm)["manifest"]),
            }

    manifest = {
        "generation_schema_version": GENERATION_SCHEMA_VERSION,
        "library_version": library_version,
        "basin": basin, "mode": mode, "obs_qa_file": obs_qa_file, "qa_tag": qa_tag,
        "bands": bands, "water_years": water_years,
        "obs_threshold": obs_threshold, "max_pillows": max_pillows,
        "convention": convention,
        "members": members,
        "n_bands": len(bands), "n_units_per_band": len(units),
        "n_units_total": sum(len(v) for v in members.values()),
        "n_units_expected": len(bands) * len(units),
        "complete": not problems,
        "problems": problems,
    }
    return manifest, problems


def finalize_generation_manifest(manifest: dict, repo_root: Path) -> dict:
    """Stamp build time + git state onto an already-assembled manifest, just before
    saving. Kept separate from build_generation_manifest so a dry-run can inspect the
    manifest without it claiming a build time that never happened."""
    import subprocess

    def run(*a):
        try:
            return subprocess.run(a, cwd=repo_root, capture_output=True, text=True,
                                  timeout=15, check=True).stdout.strip()
        except Exception:
            return None
    status = run("git", "status", "--porcelain")
    manifest = dict(manifest)
    manifest["built_at_utc"] = datetime.now(timezone.utc).isoformat()
    manifest["git"] = {
        "branch": run("git", "rev-parse", "--abbrev-ref", "HEAD"),
        "commit": run("git", "rev-parse", "HEAD"),
        "dirty": bool(status),
        "dirty_files": sorted(l[2:].strip() for l in (status or "").splitlines() if l.strip()),
    }
    return manifest


def save_generation_manifest(manifest: dict, mlr_pred_dir: Path) -> Path:
    path = generation_manifest_path(mlr_pred_dir, manifest["library_version"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True, default=str) + "\n")
    return path


# ===========================================================================
# resolution -- explicit version in, exact manifest path out, no globbing
# ===========================================================================

def load_generation_manifest(mlr_pred_dir: Path, library_version: str) -> dict:
    path = generation_manifest_path(mlr_pred_dir, library_version)
    if not path.exists():
        raise FileNotFoundError(
            f"no generation manifest for library_version={library_version!r} at {path}. "
            f"library versions are never inferred -- build it with "
            f"build_library_generation.py first.")
    return json.loads(path.read_text())


def resolve_library_manifest_path(mlr_pred_dir: Path, library_version: str,
                                  band: str) -> Path:
    """The full-history ('all_years') library manifest path for one band, from an
    explicit generation. Raises rather than falling back to any other selection rule."""
    gm = load_generation_manifest(mlr_pred_dir, library_version)
    try:
        entry = gm["members"][band]["all_years"]
    except KeyError:
        raise KeyError(f"{library_version} has no full-history unit for band={band!r}; "
                       f"available bands: {sorted(gm['members'])}")
    return Path(entry["library_manifest_path"])


def resolve_dropped_year_manifest_paths(mlr_pred_dir: Path, library_version: str,
                                        band: str) -> dict[int, Path]:
    """{water_year: library_manifest_path} for every drop_wy* unit in one band, from an
    explicit generation."""
    gm = load_generation_manifest(mlr_pred_dir, library_version)
    if band not in gm["members"]:
        raise KeyError(f"{library_version} has no units for band={band!r}; "
                       f"available bands: {sorted(gm['members'])}")
    out = {}
    for library_id, entry in gm["members"][band].items():
        if library_id == "all_years":
            continue
        wy = entry["excluded_years"]
        if len(wy) != 1:
            raise RuntimeError(f"{library_id} has excluded_years={wy}, expected exactly one")
        out[int(wy[0])] = Path(entry["library_manifest_path"])
    return out


# ===========================================================================
# registries -- convenience/provenance indexes ONLY, never consulted to pick
# a version implicitly. Every function above requires an explicit
# library_version and would raise if one were guessed from these files.
# ===========================================================================

def _append_csv_row(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    is_new = not path.exists()
    with path.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if is_new:
            w.writeheader()
        w.writerow(row)


def library_registry_path(mlr_pred_dir: Path) -> Path:
    return Path(mlr_pred_dir) / "model_library" / "library_registry.csv"


def append_library_registry_row(mlr_pred_dir: Path, manifest: dict) -> Path:
    path = library_registry_path(mlr_pred_dir)
    _append_csv_row(path, {
        "version": manifest["library_version"], "basin": manifest["basin"],
        "created_at_utc": manifest["built_at_utc"],
        "git_commit": manifest["git"]["commit"], "git_dirty": manifest["git"]["dirty"],
        "qa_tag": manifest["qa_tag"], "n_bands": manifest["n_bands"],
        "n_units_total": manifest["n_units_total"], "complete": manifest["complete"],
        "generation_manifest_path": str(generation_manifest_path(
            mlr_pred_dir, manifest["library_version"])),
    })
    return path


def prediction_run_registry_path(mlr_pred_dir: Path) -> Path:
    return Path(mlr_pred_dir) / "model_library" / "prediction_run_registry.csv"


def append_prediction_run_registry_row(mlr_pred_dir: Path, *, run_version: str,
                                       run_type: str, basin: str, library_version: str,
                                       wy_or_date: str, status: str, git_commit: str,
                                       git_dirty: bool, manifest_path: str) -> Path:
    path = prediction_run_registry_path(mlr_pred_dir)
    _append_csv_row(path, {
        "run_version": run_version, "run_type": run_type, "basin": basin,
        "library_version": library_version, "wy_or_date": wy_or_date,
        "submitted_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": status, "git_commit": git_commit, "git_dirty": git_dirty,
        "manifest_path": manifest_path,
    })
    return path
