"""
A single reusable result for one (date, elevation band): the full-history baseline
prediction plus its two INDEPENDENT sensitivity dimensions, station-loss and
historical-training.

This module fits nothing and queries nothing new. It is a thin composition of three
functions already built and validated in earlier milestones:

    ModelLibrary.select_best_model() / .predict()   -- the baseline
    sensitivity.station_loss_sensitivity()          -- station-loss members
    historical_year.historical_year_sensitivity()   -- historical-training members

build_prediction_result() calls each exactly as its own driver already does, reshapes
their output into the requested field names, and adds the requested summary statistics.
No new eligibility logic, no new prediction arithmetic.

TWO DISTINCT DIMENSIONS -- DO NOT COMBINE THEM

Station-loss sensitivity asks: "what if one pillow the winning model relies on today goes
missing?" Historical-training sensitivity asks: "what if a whole ASO water year had never
been in the training record?" These measure different things -- one is about TODAY's
network robustness, the other is about the STABILITY of the fitted coefficients across the
historical record -- and are reported as two separate member lists with two separate
summaries. There is no combined interval, no pooled standard deviation across the two, and
no Cartesian product of "drop this station AND this water year" here. Building a single
statistical confidence interval out of either summary would overstate what has been
measured: these are in-sample selection/sensitivity diagnostics (see fit.py's
`score_is_in_sample` / `score_intended_use` manifest fields), not calibrated uncertainty.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from snow_ops.mlr.lookup.query import ModelLibrary
from snow_ops.mlr.lookup.sensitivity import station_loss_sensitivity
from snow_ops.mlr.lookup.historical_year import historical_year_sensitivity


# ===========================================================================
# historical-training sensitivity
# ===========================================================================

@dataclass
class HistoricalYearMember:
    excluded_wy: int
    pillow_key: str | None
    prediction_mm: float | None
    delta_from_baseline_mm: float | None
    abs_delta_mm: float | None
    eligible: bool


@dataclass
class HistoricalYearSensitivity:
    members: list[HistoricalYearMember] = field(default_factory=list)
    min_prediction_mm: float | None = None
    max_prediction_mm: float | None = None
    range_mm: float | None = None
    std_prediction_mm: float | None = None
    max_abs_delta_mm: float | None = None
    most_influential_excluded_wy: int | None = None

    def table(self) -> pd.DataFrame:
        """Long-form member table -- one row per dropped water year."""
        return pd.DataFrame([m.__dict__ for m in self.members])


# ===========================================================================
# station-loss sensitivity
# ===========================================================================

@dataclass
class StationLossMember:
    dropped_pillow: str
    pillow_key: str | None
    prediction_mm: float | None
    delta_from_baseline_mm: float | None
    abs_delta_mm: float | None
    eligible: bool


@dataclass
class StationLossSensitivity:
    members: list[StationLossMember] = field(default_factory=list)
    min_prediction_mm: float | None = None
    max_prediction_mm: float | None = None
    range_mm: float | None = None
    std_prediction_mm: float | None = None
    max_abs_delta_mm: float | None = None
    most_influential_dropped_pillow: str | None = None

    def table(self) -> pd.DataFrame:
        """Long-form member table -- one row per dropped pillow in the baseline model."""
        return pd.DataFrame([m.__dict__ for m in self.members])


# ===========================================================================
# combined result
# ===========================================================================

@dataclass
class PredictionResult:
    date: str
    basin: str
    elevation_band: str

    baseline_prediction_mm: float
    baseline_pillow_key: str
    baseline_n_predictors: int
    baseline_adj_r2: float
    baseline_rmse: float

    historical_training_sensitivity: HistoricalYearSensitivity
    station_loss_sensitivity: StationLossSensitivity

    def summary_dict(self) -> dict:
        """One flat row: baseline plus both sensitivity summaries. The individual
        members are NOT included here -- use historical_training_sensitivity.table() /
        station_loss_sensitivity.table() for the long-form data."""
        h, s = self.historical_training_sensitivity, self.station_loss_sensitivity
        return {
            "date": self.date, "basin": self.basin, "elevation_band": self.elevation_band,
            "baseline_prediction_mm": self.baseline_prediction_mm,
            "baseline_pillow_key": self.baseline_pillow_key,
            "baseline_n_predictors": self.baseline_n_predictors,
            "baseline_adj_r2": self.baseline_adj_r2,
            "baseline_rmse": self.baseline_rmse,
            "hist_min_prediction_mm": h.min_prediction_mm,
            "hist_max_prediction_mm": h.max_prediction_mm,
            "hist_range_mm": h.range_mm,
            "hist_std_prediction_mm": h.std_prediction_mm,
            "hist_max_abs_delta_mm": h.max_abs_delta_mm,
            "hist_most_influential_excluded_wy": h.most_influential_excluded_wy,
            "station_min_prediction_mm": s.min_prediction_mm,
            "station_max_prediction_mm": s.max_prediction_mm,
            "station_range_mm": s.range_mm,
            "station_std_prediction_mm": s.std_prediction_mm,
            "station_max_abs_delta_mm": s.max_abs_delta_mm,
            "station_most_influential_dropped_pillow": s.most_influential_dropped_pillow,
        }


def _summary_stats(preds: list[float], deltas: list[tuple], ) -> dict:
    """Shared aggregation for either sensitivity dimension. `deltas` is a list of
    (member_id, abs_delta_mm) pairs, already filtered to eligible members."""
    if not preds:
        return dict(min_prediction_mm=None, max_prediction_mm=None, range_mm=None,
                    std_prediction_mm=None, max_abs_delta_mm=None, most_influential=None)
    # Sample std needs >=2 points; a single eligible member has zero spread by
    # construction, not an undefined one, so it is reported as 0.0 rather than NaN.
    std = float(np.std(preds, ddof=1)) if len(preds) >= 2 else 0.0
    most_influential_id, max_abs = max(deltas, key=lambda t: t[1])
    return dict(min_prediction_mm=min(preds), max_prediction_mm=max(preds),
               range_mm=max(preds) - min(preds), std_prediction_mm=std,
               max_abs_delta_mm=max_abs, most_influential=most_influential_id)


def build_prediction_result(
    baseline_lib: ModelLibrary,
    dropped_year_libs: Mapping[int, ModelLibrary],
    available_pillows: Sequence[str],
    pillow_values: Mapping[str, float],
    *,
    score_rule: str = "adj_r2",
    basin: str = "",
    band: str = "",
    date: str = "",
) -> PredictionResult | None:
    """
    One date/band, one call: full-history baseline + station-loss sensitivity +
    historical-training sensitivity.

    Returns None if no full-history model is eligible today -- a normal data gap, matching
    select_best_model()'s own None-not-raise convention. When that happens neither
    sensitivity function is even called, since both are baseline-relative.

    No model is fit and no library is rebuilt: this function only calls
    select_best_model(), predict(), station_loss_sensitivity() and
    historical_year_sensitivity() -- all of which were validated against independent
    sklearn refits and brute-force search in earlier milestones.
    """
    baseline = baseline_lib.select_best_model(available_pillows, score_rule=score_rule)
    if baseline is None:
        return None
    baseline_pred = baseline_lib.predict(baseline, pillow_values)

    sl_raw = station_loss_sensitivity(
        baseline_lib, available_pillows, pillow_values,
        score_rule=score_rule, band=band, date=date)
    hy_raw = historical_year_sensitivity(
        baseline_lib, dropped_year_libs, available_pillows, pillow_values,
        score_rule=score_rule, band=band, date=date)

    sl_members = [StationLossMember(
        dropped_pillow=r.dropped_pillow, pillow_key=r.replacement_pillow_key,
        prediction_mm=r.replacement_pred_mm, delta_from_baseline_mm=r.delta_mm,
        abs_delta_mm=r.abs_delta_mm, eligible=r.replacement_eligible) for r in sl_raw]
    hy_members = [HistoricalYearMember(
        excluded_wy=r.dropped_wy, pillow_key=r.library_pillow_key,
        prediction_mm=r.library_pred_mm, delta_from_baseline_mm=r.delta_mm,
        abs_delta_mm=r.abs_delta_mm, eligible=r.eligible) for r in hy_raw]

    sl_stats = _summary_stats(
        [m.prediction_mm for m in sl_members if m.eligible],
        [(m.dropped_pillow, m.abs_delta_mm) for m in sl_members if m.eligible])
    hy_stats = _summary_stats(
        [m.prediction_mm for m in hy_members if m.eligible],
        [(m.excluded_wy, m.abs_delta_mm) for m in hy_members if m.eligible])

    station_loss = StationLossSensitivity(
        members=sl_members,
        min_prediction_mm=sl_stats["min_prediction_mm"],
        max_prediction_mm=sl_stats["max_prediction_mm"],
        range_mm=sl_stats["range_mm"], std_prediction_mm=sl_stats["std_prediction_mm"],
        max_abs_delta_mm=sl_stats["max_abs_delta_mm"],
        most_influential_dropped_pillow=sl_stats["most_influential"])
    historical_training = HistoricalYearSensitivity(
        members=hy_members,
        min_prediction_mm=hy_stats["min_prediction_mm"],
        max_prediction_mm=hy_stats["max_prediction_mm"],
        range_mm=hy_stats["range_mm"], std_prediction_mm=hy_stats["std_prediction_mm"],
        max_abs_delta_mm=hy_stats["max_abs_delta_mm"],
        most_influential_excluded_wy=hy_stats["most_influential"])

    return PredictionResult(
        date=date, basin=basin, elevation_band=band,
        baseline_prediction_mm=baseline_pred,
        baseline_pillow_key=baseline["pillow_key"],
        baseline_n_predictors=int(baseline["n_predictors"]),
        baseline_adj_r2=float(baseline["adj_r2"]),
        baseline_rmse=float(baseline["rmse"]),
        historical_training_sensitivity=historical_training,
        station_loss_sensitivity=station_loss)


def summary_table(results: Sequence[PredictionResult]) -> pd.DataFrame:
    """Compact summary across several PredictionResults (e.g. multiple bands/dates).
    Individual sensitivity members are NOT in this table -- see
    PredictionResult.historical_training_sensitivity.table() /
    .station_loss_sensitivity.table() for those."""
    return pd.DataFrame([r.summary_dict() for r in results])
