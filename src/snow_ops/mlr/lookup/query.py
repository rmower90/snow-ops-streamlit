"""
The daily half of the pretrained lookup: given the pillows that reported and passed QA
today, find the best eligible pretrained model and evaluate it.

No fitting. No imputation. No cross-validation. Today's availability is a bitmask, and
selection is a filter plus a sort over columns that are already on disk.

A model is eligible when every pillow it needs is available today:

    (pillow_mask & ~available_mask) == 0

Everything else in this module is built on that one line -- which is also why the two
planned sensitivity components need no new machinery. Station-loss sensitivity clears bits
from `available_mask` against one library (`exclude=`); historical-year sensitivity varies
the library against one mask.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from snow_ops.mlr.lookup.fit import SCORE_RULES, rank_frame


@dataclass
class ModelLibrary:
    """A pretrained model table plus the manifest that makes it interpretable.

    The manifest is not optional decoration: `pillow_mask` is meaningless without
    `bitmask_bit_order`, and the fitting conventions are needed to evaluate a stored model
    correctly.
    """

    table: pd.DataFrame
    manifest: dict

    @property
    def universe(self) -> tuple[str, ...]:
        return tuple(self.manifest["bitmask_bit_order"])

    @property
    def default_score_rule(self) -> str:
        return self.manifest.get("default_score_rule", "adj_r2")

    # ---------------------------------------------------------------- masks

    def availability_mask(self, available_pillows: Iterable[str]) -> dict:
        """Turn today's pillow names into a bitmask.

        Pillows available today but absent from the library's universe are reported rather
        than silently ignored: they are unusable (no model was ever fitted on them) but
        their presence is a signal that the library may be due a rebuild.
        """
        universe = self.universe
        bit = {p: i for i, p in enumerate(universe)}
        available = sorted(set(map(str, available_pillows)))

        usable = [p for p in available if p in bit]
        outside = [p for p in available if p not in bit]
        unavailable = [p for p in universe if p not in set(available)]

        mask = 0
        for p in usable:
            mask |= 1 << bit[p]
        return {
            "mask": mask,
            "usable": usable,
            "outside_universe": outside,
            "unavailable": unavailable,
            "n_usable": len(usable),
        }

    # ----------------------------------------------------------- filtering

    def eligible_models(
        self,
        available_pillows: Iterable[str],
        *,
        exclude: Iterable[str] = (),
        max_pillow_frac_imputed: float | None = None,
    ) -> pd.DataFrame:
        """Models whose required pillows are all available today.

        `exclude` removes specific pillows from the available set. It is the station-loss
        sensitivity hook and does nothing unless used.

        `max_pillow_frac_imputed` filters on imputation burden. It reads a stored column,
        so tightening it never requires a rebuild.
        """
        exclude = set(map(str, exclude))
        available = [p for p in map(str, available_pillows) if p not in exclude]
        info = self.availability_mask(available)
        mask = info["mask"]

        # required-and-not-available must be empty
        needed_but_missing = self.table["pillow_mask"].to_numpy() & ~np.int64(mask)
        out = self.table[needed_but_missing == 0]

        if max_pillow_frac_imputed is not None:
            out = out[out["max_pillow_frac_imputed"] <= max_pillow_frac_imputed]
        return out

    # ----------------------------------------------------------- selection

    def select_best_model(
        self,
        available_pillows: Iterable[str],
        score_rule: str = "adj_r2",
        *,
        exclude: Iterable[str] = (),
        max_pillow_frac_imputed: float | None = None,
    ) -> pd.Series | None:
        """Best eligible pretrained model, or None if today admits no model at all.

        Tie-break: score, then fewer predictors, then lexicographic pillow_key.

        Returning None rather than raising is deliberate -- a day with no usable pillows is
        a normal data gap, and the caller writes a NaN row for it exactly as the historic
        runner's existing except-branch already does.
        """
        if score_rule not in SCORE_RULES:
            raise ValueError(f"unknown score_rule {score_rule!r}; "
                             f"known: {sorted(SCORE_RULES)}")
        elig = self.eligible_models(
            available_pillows, exclude=exclude,
            max_pillow_frac_imputed=max_pillow_frac_imputed)
        if elig.empty:
            return None
        return rank_frame(elig, score_rule).iloc[0]

    # ---------------------------------------------------------- prediction

    def predict(self, model: pd.Series, pillow_values: Mapping[str, float],
                *, clip_negative: bool = True) -> float:
        """Evaluate one stored model. Just intercept + sum(coef * value)."""
        return predict_from_lookup(model, pillow_values, clip_negative=clip_negative)


def model_pillows(model: pd.Series) -> list[str]:
    """The pillows a stored model needs, read back out of the fixed-width layout."""
    return [model[f"pillow_{i}"] for i in range(1, 6)
            if model.get(f"pillow_{i}") is not None
            and not pd.isna(model.get(f"pillow_{i}"))]


def predict_from_lookup(model: pd.Series, pillow_values: Mapping[str, float],
                        *, clip_negative: bool = True) -> float:
    """
    Basin-mean SWE in mm from stored coefficients.

    `clip_negative` reproduces production: `run_daily_prediction` clamps a negative
    prediction to zero before it is published.
    """
    total = float(model["intercept"])
    for i in range(1, 6):
        pil = model.get(f"pillow_{i}")
        if pil is None or pd.isna(pil):
            break
        if pil not in pillow_values:
            raise KeyError(f"model needs pillow {pil!r}, which is not in pillow_values")
        total += float(model[f"coef_{i}"]) * float(pillow_values[pil])
    if clip_negative and total < 0.0:
        total = 0.0
    return total


def predict_acre_feet(pred_mm: float, area_m2: float) -> float:
    """mm -> acre-feet.

    No separate coefficients are needed for volume: run_daily_prediction fits on X*c and
    y*c with the same scalar, so slopes are scale-invariant and the acre-foot prediction is
    the millimetre prediction times the conversion factor.
    """
    return pred_mm * area_m2 * 0.001 * 0.000810714
