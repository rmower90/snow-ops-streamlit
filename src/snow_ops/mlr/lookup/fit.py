"""
Enumerate every allowed pillow combination, fit each one, and record its coefficients and
diagnostics. This is the offline half of the pretrained lookup: once it has run, the daily
path never fits anything.

FITTING CONVENTION (V4-V6, confirmed by A/B reconstruction)

    sklearn LinearRegression(fit_intercept=False), fit on the 35 real ASO flights PLUS the
    synthetic all-zero row -- 36 rows. All statistics are computed on the 35 real flights
    only.

Splitting fit rows from stats rows is sound here rather than a fudge. Under
fit_intercept=False a row of zeros adds nothing to X'X or X'y, so the coefficients are
*exactly* those of the 35-row fit (measured max|dcoef| = 0.0e+00) and SSE is unchanged.
What the extra row does change is n and ybar, which would inflate R2 by ~0.06 and shift
RMSE. So the effective model is the 35-row no-intercept OLS, and the honest degrees of
freedom are n = 35, df_resid = n - k. Under fit_intercept=True none of this would hold --
the zero row is a real anchoring observation there -- so `add_zero_aso` and `fit_intercept`
are recorded together in the manifest and must be read together.

METRIC DEFINITIONS

Every formula below was checked against statsmodels 0.14.5 OLS with no constant, which is
the same model we fit. n = 35 (real flights), k = n_predictors, SSE = sum of squared
residuals over the real flights.

    TSS_centered   = sum((y - ybar)^2)          df_total = n - 1
    TSS_uncentered = sum(y^2)                   df_total = n

    r2                  = 1 - SSE / TSS_centered
        The sklearn convention. `LinearRegression.score()` and `metrics.r2_score()` use
        CENTERED TSS even when fit_intercept=False -- verified. Can go negative, which is
        informative for an origin-forced fit, so it is kept as the primary reported R2.

    r2_uncentered       = 1 - SSE / TSS_uncentered
        The R / statsmodels convention for a no-intercept model: statsmodels reports this
        as `.rsquared` when no constant is present -- verified equal.

    adj_r2              = 1 - (SSE / (n - k)) / (TSS_centered / (n - 1))     <- DEFAULT RANK
    adj_r2_uncentered   = 1 - (SSE / (n - k)) / (TSS_uncentered / n)
        Note the residual df is n - k, NOT n - k - 1: no intercept is estimated, so no
        degree of freedom is spent on one. adj_r2_uncentered was verified equal to
        statsmodels' `.rsquared_adj` for a no-constant fit.

        Useful property: both adjusted variants are of the form 1 - C * SSE / (n - k) with
        C > 0 fixed by y alone, so the constant cancels in any pairwise comparison and the
        two RANK IDENTICALLY. The centered/uncentered choice changes the reported number
        but never the selected model.

    adj_r2_legacy       = 1 - (1 - r2) * (n - 1) / (n - k - 1)
        The formula already used across lm_model.py / crossval.py, stored purely so numbers
        remain comparable with existing outputs. It spends a df on an intercept that was
        not estimated, so it ranks by SSE / (n - k - 1) and CAN order models differently
        from adj_r2. Not the default.

    rmse = sqrt(SSE / n)        population form, matching sklearn root_mean_squared_error
    mae  = mean(|y - yhat|)

    Gaussian log-likelihood at the MLE, sigma2 = SSE / n:
        logLik = -0.5 * n * (log(2*pi) + log(sigma2) + 1)      <- verified == statsmodels llf

    K = k + 1               k slopes plus sigma^2. No intercept is estimated.
        aic  = -2*logLik + 2*K
        aicc = aic + 2*K*(K+1) / (n - K - 1)
        bic  = -2*logLik + K*log(n)

        statsmodels uses K = k (it does not count sigma^2), so its `.aic` / `.bic` are
        lower than ours by exactly 2 and log(n) -- a constant at fixed n, so AIC and BIC
        rankings are unaffected by the choice. AICc is different: its correction term
        depends on K, so counting sigma^2 genuinely matters and the Burnham & Anderson
        convention (K = k + 1) is used.

        AICc needs n - K - 1 > 0, i.e. k < n - 2 = 33. Comfortable at k <= 5.
"""

from __future__ import annotations

import math
from itertools import combinations
from typing import Iterator, Sequence

import numpy as np
import pandas as pd
from sklearn import linear_model

from snow_ops.mlr.lookup.frame import CanonicalFrame, MAX_PILLOWS

METRIC_CONVENTIONS = {
    "n_used_for_stats": "real ASO flights only (synthetic zero row excluded)",
    "df_resid": "n - k  (no intercept estimated)",
    "r2": "1 - SSE/TSS_centered  (sklearn r2_score convention)",
    "r2_uncentered": "1 - SSE/sum(y^2)  (statsmodels no-constant convention)",
    "adj_r2": "1 - (SSE/(n-k)) / (TSS_centered/(n-1))",
    "adj_r2_uncentered": "1 - (SSE/(n-k)) / (sum(y^2)/n)  (== statsmodels rsquared_adj)",
    "adj_r2_legacy": "1 - (1-r2)*(n-1)/(n-k-1)  (lm_model.py formula, for comparability)",
    "rmse": "sqrt(SSE/n)",
    "mae": "mean(|y-yhat|)",
    "logLik": "-0.5*n*(log(2pi)+log(SSE/n)+1)  (== statsmodels llf)",
    "K": "k + 1  (slopes + sigma^2; no intercept)",
    "aic": "-2*logLik + 2*K",
    "aicc": "aic + 2*K*(K+1)/(n-K-1)",
    "bic": "-2*logLik + K*log(n)",
    "note_adj_rank_equivalence": ("adj_r2 and adj_r2_uncentered rank identically; "
                                 "adj_r2_legacy may not"),
    "verified_against": "statsmodels 0.14.5 OLS (no constant)",
}

# score column -> True if a LARGER value is better
SCORE_RULES = {
    "adj_r2": True,
    "adj_r2_uncentered": True,
    "adj_r2_legacy": True,
    "r2": True,
    "r2_uncentered": True,
    "rmse": False,
    "mae": False,
    "aic": False,
    "aicc": False,
    "bic": False,
}

# int64 rather than uint64: 25 bits is nowhere near overflow, and int64 avoids numpy's
# uint64-with-Python-int casting surprises in the eligibility mask arithmetic. Good for
# up to 62 pillows.
_MASK_DTYPE = "int64"


def enumerate_combinations(n_pillows: int,
                           max_pillows: int = MAX_PILLOWS) -> Iterator[tuple[int, ...]]:
    """Column-index combinations, k = 1..max_pillows, in canonical sorted order.

    `combinations` over `range(n)` is already ascending, so every combination appears
    exactly once with its indices sorted. That is what makes the enumeration
    deterministic and dedup-free -- the legacy path derived its candidate list from
    `list(set(...))`, whose iteration order decided which of two tied combinations won.
    """
    for k in range(1, max_pillows + 1):
        yield from combinations(range(n_pillows), k)


def expected_n_models(n_pillows: int, max_pillows: int = MAX_PILLOWS) -> int:
    return sum(math.comb(n_pillows, k) for k in range(1, max_pillows + 1))


def pillow_mask(indices: Sequence[int]) -> int:
    """Bit i set means universe[i] is required by this model."""
    m = 0
    for i in indices:
        m |= 1 << int(i)
    return m


def build_model_library(
    frame: CanonicalFrame,
    *,
    max_pillows: int = MAX_PILLOWS,
    fit_intercept: bool | None = None,
    add_zero_aso: bool | None = None,
    progress_every: int = 10000,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Fit every pillow combination of size 1..max_pillows against the canonical frame.

    One row per combination. Ordinary sklearn fitting, one model at a time -- readable and
    fast enough (~68k fits in well under two minutes), so no custom solver.
    """
    if fit_intercept is None:
        fit_intercept = bool(frame.manifest["fit_intercept"])
    if add_zero_aso is None:
        add_zero_aso = bool(frame.manifest["add_zero_aso"])

    log = (lambda *a: print(*a, flush=True)) if verbose else (lambda *a: None)

    universe = list(frame.universe)
    X, y = frame.X, frame.y
    n, p = X.shape

    # Fit design: real flights plus the synthetic zero row. Built once and column-sliced
    # per combination rather than re-stacked 68,405 times.
    if add_zero_aso:
        X_fit_all = np.vstack([X, np.zeros((1, p))])
        y_fit = np.append(y, 0.0)
    else:
        X_fit_all, y_fit = X, y

    # Statistics are always over the real flights, so these are constants.
    tss_centered = float(((y - y.mean()) ** 2).sum())
    tss_uncentered = float((y ** 2).sum())
    frac_imputed_col = frame.imputed.mean(axis=0)

    total = expected_n_models(p, max_pillows)
    log(f"  fitting {total:,} combinations over {p} pillows "
        f"(fit_intercept={fit_intercept}, add_zero_aso={add_zero_aso}, "
        f"fit_n={X_fit_all.shape[0]}, stats_n={n})")

    rows = []
    for count, idx in enumerate(enumerate_combinations(p, max_pillows), start=1):
        cols = list(idx)
        k = len(cols)

        lm = linear_model.LinearRegression(fit_intercept=fit_intercept)
        lm.fit(X_fit_all[:, cols], y_fit)

        # residuals over the REAL flights only
        resid = y - lm.predict(X[:, cols])
        sse = float((resid ** 2).sum())

        r2 = 1.0 - sse / tss_centered
        r2_unc = 1.0 - sse / tss_uncentered
        df_resid = n - k                        # no intercept estimated
        adj_r2 = 1.0 - (sse / df_resid) / (tss_centered / (n - 1))
        adj_r2_unc = 1.0 - (sse / df_resid) / (tss_uncentered / n)
        adj_r2_legacy = (1.0 - (1.0 - r2) * (n - 1) / (n - k - 1)
                         if n - k - 1 > 0 else float("nan"))

        # Gaussian logLik at sigma2 = SSE/n. A perfect fit would send this to -inf; guard
        # so one degenerate combination cannot poison the whole table.
        if sse > 0.0:
            sigma2 = sse / n
            log_lik = -0.5 * n * (math.log(2.0 * math.pi) + math.log(sigma2) + 1.0)
            K = k + 1                            # slopes + sigma^2
            aic = -2.0 * log_lik + 2.0 * K
            aicc = (aic + 2.0 * K * (K + 1) / (n - K - 1)
                    if n - K - 1 > 0 else float("nan"))
            bic = -2.0 * log_lik + K * math.log(n)
        else:
            log_lik = aic = aicc = bic = float("nan")

        pillows = [universe[i] for i in cols]
        per_pillow_frac = frac_imputed_col[cols]

        row = {
            "pillow_key": ",".join(pillows),
            "pillow_mask": pillow_mask(cols),
            "n_predictors": k,
            "intercept": float(lm.intercept_) if fit_intercept else 0.0,
            # metrics
            "r2": r2,
            "adj_r2": adj_r2,
            "r2_uncentered": r2_unc,
            "adj_r2_uncentered": adj_r2_unc,
            "adj_r2_legacy": adj_r2_legacy,
            "rmse": math.sqrt(sse / n),
            "mae": float(np.abs(resid).mean()),
            "sse": sse,
            "log_lik": log_lik,
            "aic": aic,
            "aicc": aicc,
            "bic": bic,
            "n_obs": n,                          # == stats_n_obs
            "fit_n_obs": int(X_fit_all.shape[0]),
            "df_resid": df_resid,
            # imputation burden (B1) -- a property of the frame, so it is available here
            # without refitting and lets models be filtered on burden later
            "n_imputed_cells": int(frame.imputed[:, cols].sum()),
            "frac_imputed": float(frame.imputed[:, cols].mean()),
            "max_pillow_frac_imputed": float(per_pillow_frac.max()),
            "mean_pillow_frac_imputed": float(per_pillow_frac.mean()),
            # conditioning, to flag combinations whose coefficients are not trustworthy
            "cond_number": float(np.linalg.cond(X_fit_all[:, cols])),
        }
        # fixed-width coefficient layout: coef_i pairs with pillow_i
        for j in range(MAX_PILLOWS):
            row[f"pillow_{j + 1}"] = pillows[j] if j < k else None
            row[f"coef_{j + 1}"] = float(lm.coef_[j]) if j < k else float("nan")
        rows.append(row)

        if verbose and progress_every and count % progress_every == 0:
            log(f"    {count:,} / {total:,}")

    df = pd.DataFrame(rows)
    df["pillow_mask"] = df["pillow_mask"].astype(_MASK_DTYPE)

    if len(df) != total:
        raise RuntimeError(f"produced {len(df)} models, expected {total}")
    if df["pillow_key"].duplicated().any():
        dupes = df.loc[df["pillow_key"].duplicated(), "pillow_key"].head().tolist()
        raise RuntimeError(f"duplicate pillow_key rows: {dupes}")

    df = add_default_rank(df)
    log(f"  built {len(df):,} models")
    return df


def rank_frame(df: pd.DataFrame, score_rule: str = "adj_r2") -> pd.DataFrame:
    """Sort best-first under the agreed tie-break: score, then fewer predictors, then
    lexicographic pillow_key. The last two make the winner reproducible when scores tie
    exactly -- the legacy selector left that to set-iteration order."""
    if score_rule not in SCORE_RULES:
        raise ValueError(f"unknown score_rule {score_rule!r}; "
                         f"known: {sorted(SCORE_RULES)}")
    higher_is_better = SCORE_RULES[score_rule]
    return df.sort_values(
        by=[score_rule, "n_predictors", "pillow_key"],
        ascending=[not higher_is_better, True, True],
        kind="mergesort",       # stable, so the ordering is reproducible
    )


def add_default_rank(df: pd.DataFrame, score_rule: str = "adj_r2") -> pd.DataFrame:
    """Materialise a rank column for the DEFAULT rule only, as a convenience for
    inspection. It is derived, never the sole access path: `select_best_model` re-ranks
    from the metric columns, so switching score_rule never needs a rebuild."""
    ordered = rank_frame(df, score_rule)
    df = df.copy()
    df[f"rank_{score_rule}"] = pd.Series(
        np.arange(1, len(ordered) + 1), index=ordered.index)
    return df
