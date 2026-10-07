"""
Historical-year sensitivity: for a baseline (full-history) prediction, ask "how much would
today's prediction change if a whole ASO calendar year had never been in the training
record?" -- independently for each dropped-year library.

Composition only, and a pure query. Each dropped-year library from
build_leave_one_year_libraries.py was already fitted offline, with its own eligibility
universe recomputed on its own reduced frame. This module queries each one exactly the way
the full-history library is queried -- a bitmask filter plus select_best_model() -- and
does no refitting.

Deliberately NOT combined with station-loss sensitivity (sensitivity.py) here. The two
components stay independent for this milestone; a joint query (e.g. "drop this station
within the WY2019-excluded library") is future work, not implemented.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import pandas as pd

from snow_ops.mlr.lookup.query import ModelLibrary


@dataclass
class HistoricalYearResult:
    band: str
    date: str
    baseline_pillow_key: str
    baseline_pred_mm: float
    dropped_wy: int
    library_pillow_key: str | None
    library_pred_mm: float | None
    delta_mm: float | None
    abs_delta_mm: float | None
    eligible: bool

    def as_dict(self) -> dict:
        return {
            "band": self.band, "date": self.date,
            "baseline_pillow_key": self.baseline_pillow_key,
            "baseline_pred_mm": self.baseline_pred_mm,
            "dropped_wy": self.dropped_wy,
            "library_pillow_key": self.library_pillow_key,
            "library_pred_mm": self.library_pred_mm,
            "delta_mm": self.delta_mm,
            "abs_delta_mm": self.abs_delta_mm,
            "eligible": self.eligible,
        }


def historical_year_sensitivity(
    baseline_lib: ModelLibrary,
    dropped_year_libs: Mapping[int, ModelLibrary],
    available_pillows: Sequence[str],
    pillow_values: Mapping[str, float],
    *,
    score_rule: str = "adj_r2",
    band: str = "",
    date: str = "",
) -> list[HistoricalYearResult]:
    """
    Full-history baseline model + prediction, then one independently selected model +
    prediction from EACH dropped-year library, using the SAME available_pillows and
    pillow_values -- so the only thing that changes across rows is which historical years
    were in the training record when that library was fitted.

    Each dropped-year library has its own universe (possibly different from the baseline's
    -- see build_leave_one_year_libraries.py), so eligibility is evaluated independently per
    library, not against the baseline's universe.

    Returns [] if no baseline model is eligible today (a normal data gap, matching
    select_best_model's own None-not-raise convention).
    """
    baseline = baseline_lib.select_best_model(available_pillows, score_rule=score_rule)
    if baseline is None:
        return []
    baseline_pred = baseline_lib.predict(baseline, pillow_values)

    results = []
    for wy in sorted(dropped_year_libs):
        lib = dropped_year_libs[wy]
        m = lib.select_best_model(available_pillows, score_rule=score_rule)

        if m is None:
            results.append(HistoricalYearResult(
                band=band, date=date, baseline_pillow_key=baseline["pillow_key"],
                baseline_pred_mm=baseline_pred, dropped_wy=wy,
                library_pillow_key=None, library_pred_mm=None,
                delta_mm=None, abs_delta_mm=None, eligible=False))
            continue

        pred = lib.predict(m, pillow_values)
        delta = pred - baseline_pred
        results.append(HistoricalYearResult(
            band=band, date=date, baseline_pillow_key=baseline["pillow_key"],
            baseline_pred_mm=baseline_pred, dropped_wy=wy,
            library_pillow_key=m["pillow_key"], library_pred_mm=pred,
            delta_mm=delta, abs_delta_mm=abs(delta), eligible=True))
    return results


def historical_year_sensitivity_table(*args, **kwargs) -> pd.DataFrame:
    """Convenience wrapper returning the compact result structure as a DataFrame."""
    rows = [r.as_dict() for r in historical_year_sensitivity(*args, **kwargs)]
    return pd.DataFrame(rows)
