"""
Station-loss sensitivity: for today's baseline prediction, ask "what happens if this one
pillow the winning model relies on turns out to be missing or wrong?" for each pillow the
winner actually uses.

This adds NO machinery beyond what query.py already has. `select_best_model` already
accepts `exclude=`, and that argument was already exercised in the step-2 validation
(brute-force-verified for exactly this shape: minus each pillow of the rank-1 model). This
module is the composition:

    baseline = select_best_model(available)
    for pillow in baseline's combination:
        replacement = select_best_model(available, exclude=[pillow])

No model is refit and no library is rebuilt. Every candidate model was already fitted in
step 2; this only changes which stored rows are eligible.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence

import pandas as pd

from snow_ops.mlr.lookup.query import ModelLibrary, model_pillows


@dataclass
class StationLossResult:
    band: str
    date: str
    baseline_pillow_key: str
    baseline_pred_mm: float
    dropped_pillow: str
    replacement_pillow_key: str | None
    replacement_pred_mm: float | None
    delta_mm: float | None
    abs_delta_mm: float | None
    replacement_eligible: bool

    def as_dict(self) -> dict:
        return {
            "band": self.band, "date": self.date,
            "baseline_pillow_key": self.baseline_pillow_key,
            "baseline_pred_mm": self.baseline_pred_mm,
            "dropped_pillow": self.dropped_pillow,
            "replacement_pillow_key": self.replacement_pillow_key,
            "replacement_pred_mm": self.replacement_pred_mm,
            "delta_mm": self.delta_mm,
            "abs_delta_mm": self.abs_delta_mm,
            "replacement_eligible": self.replacement_eligible,
        }


def station_loss_sensitivity(
    lib: ModelLibrary,
    available_pillows: Sequence[str],
    pillow_values: Mapping[str, float],
    *,
    score_rule: str = "adj_r2",
    band: str = "",
    date: str = "",
) -> list[StationLossResult]:
    """
    Baseline best eligible model from `available_pillows`, then one independent
    station-loss query per pillow the baseline model actually uses.

    Returns [] if no baseline model is eligible today (a normal data gap, not an error --
    matches select_best_model's own None-not-raise convention).
    """
    baseline = lib.select_best_model(available_pillows, score_rule=score_rule)
    if baseline is None:
        return []

    baseline_pillows = model_pillows(baseline)
    baseline_pred = lib.predict(baseline, pillow_values)

    results = []
    for dropped in baseline_pillows:
        replacement = lib.select_best_model(
            available_pillows, score_rule=score_rule, exclude=[dropped])

        if replacement is None:
            results.append(StationLossResult(
                band=band, date=date,
                baseline_pillow_key=baseline["pillow_key"],
                baseline_pred_mm=baseline_pred, dropped_pillow=dropped,
                replacement_pillow_key=None, replacement_pred_mm=None,
                delta_mm=None, abs_delta_mm=None, replacement_eligible=False))
            continue

        replacement_pred = lib.predict(replacement, pillow_values)
        delta = replacement_pred - baseline_pred
        results.append(StationLossResult(
            band=band, date=date,
            baseline_pillow_key=baseline["pillow_key"],
            baseline_pred_mm=baseline_pred, dropped_pillow=dropped,
            replacement_pillow_key=replacement["pillow_key"],
            replacement_pred_mm=replacement_pred,
            delta_mm=delta, abs_delta_mm=abs(delta), replacement_eligible=True))
    return results


def station_loss_sensitivity_table(*args, **kwargs) -> pd.DataFrame:
    """Convenience wrapper returning the compact result structure as a DataFrame."""
    rows = [r.as_dict() for r in station_loss_sensitivity(*args, **kwargs)]
    return pd.DataFrame(rows)
