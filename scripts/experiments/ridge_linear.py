"""
ridge_linear.py -- ridge regression for basin SWE prediction.

VENDORED, then extended. The five functions below the divider are copied verbatim from
RIDGE_linear.py, developed on a separate HPC and staged at
/home/rossamower/work/aso/script_test/RIDGE_linear.py. They were extracted by AST rather
than retyped, so they are byte-identical to the source.

WHY A STANDALONE MODULE
-----------------------
summarize_data() also exists in scripts/lm_model.py. The version here is a strict superset:
with melt_dates=None it produces byte-identical output (verified against the operational
copy on a 30-pillow x 35-flight matrix, max|diff| = 0), and with melt_dates supplied it adds
one phase-indicator row. Carrying our own copy keeps this module importable without touching
lm_model.py, which the operational daily and historic drivers both depend on.

RELATIONSHIP TO THE OPERATIONAL MODEL
-------------------------------------
The baseline fits three SEPARATE models -- season, accum, melt -- rather than one model with
a phase term. The counterpart to the baseline `season` model is therefore ridge with
melt_dates=None (no phase column), which is also what removes any need to assign
accumulation/melt for a water year still in progress. The phase dummy remains available as a
different design, not a required one.

WHAT WAS ADDED
--------------
fit_ridge_band() / predict_ridge_band(), below. The vendored run_ridge_* functions perform
leave-one-water-year-out cross-validation and return CV predictions -- a skill assessment
over historic years. Predicting a target water year forward needs fit-on-everything then
apply, which the outer loop computes internally but never exposes. These two functions are
that split, reusing the same partialled fit and the same inner alpha search.

Not vendored: run_ridge_cross_val_pixel (per-pixel, wraps the band engine), the
best-subset family, and the OLS combination family -- the last of which lives in lm_model.py
already.
"""
import numpy as np
import pandas as pd
import xarray as xr
from sklearn import linear_model
from sklearn.preprocessing import StandardScaler


# ===========================================================================
# VENDORED VERBATIM from RIDGE_linear.py -- do not edit in place; re-extract
# if the upstream copy changes.
# ===========================================================================


def summarize_data(pillow_data, aso_tseries, melt_dates=None):
    """
        Processes pillow data to match ASO time series using nearest date.
        Input:
            pillow_data - list object of xarray dataarray objects representing
                          timeseries of each snow pillow.
            aso_tseries - dataarray object representing ASO time series of
                          basin averaged SWE (mm)
            melt_dates  - optional np.datetime64 array of melt-season flight
                          dates (from cfg seasonal_dates).  When provided a
                          binary phase indicator row is inserted after the
                          pillow rows and before the target row:
                              0 = accumulation date, 1 = melt date.
                          Pass None (default) for phase-only or melt-only runs;
                          the indicator will have zero variance and be ignored
                          by the variance filter in add_parameter.
        Output:
            full_data - numpy 2D array (n_features+1) x n_dates.
                        Rows 0..n_pils-1 : pillow values nearest to each date.
                        Row  n_pils      : phase indicator (only if melt_dates
                                           provided, else absent).
                        Last row         : ASO SWE target values.

        UPDATE vs original
        ------------------
        Added optional melt_dates parameter to inject a phase indicator feature
        for season models, allowing the regression to learn different pillow–SWE
        slopes for accumulation vs melt without requiring two separate models.
    """
    n_extra   = 1 if melt_dates is not None else 0
    full_data = np.zeros((len(pillow_data) + 1 + n_extra, len(aso_tseries)))

    # precompute melt date strings for O(1) lookup
    _melt_set = {str(md)[:10] for md in melt_dates} if melt_dates is not None else set()

    for i, t in enumerate(aso_tseries):
        for j, p in enumerate(pillow_data):
            full_data[j, i] = p.sel(time=t.date, method="nearest")

        if melt_dates is not None:
            full_data[len(pillow_data), i] = 1.0 if str(t.date.values)[:10] in _melt_set else 0.0

        ## set the aso observations in the last row ##
        full_data[-1, i] = t.values

    return full_data


def _ridge_partialled_fit_predict(P_train, y_train, P_pred, alpha,
                                  phase_train=None, phase_pred=None):
    """
    Fit ridge regression on standardized pillow columns (P_train -> y_train)
    with the L2 penalty applied only to the pillow coefficients. The
    intercept (always) and 'phase' (when supplied) are treated as fixed,
    unpenalized covariates via exact partialling (Frisch-Waugh-Lovell):

      1. Let Z = [1] (M4) or [1, phase] (M5).
      2. Partial Z out of P_train and y_train by OLS projection (np.linalg.lstsq),
         giving residuals P_resid, y_resid.
      3. Ridge-fit P_resid -> y_resid with fit_intercept=False, so sklearn's
         penalty applies only to the pillow coefficients (`beta`).
      4. Recover the fixed-covariate coefficients `gamma` (intercept, and
         phase's coefficient if present) by OLS of (y_train - P_train @ beta)
         on Z.

    This is an exact solution of the jointly-penalized least-squares problem
    (zero penalty on Z, alpha on the pillow block) — not an approximation —
    because ridge regression's normal equations are linear, so partialling
    out the unpenalized block via OLS and ridge-fitting the residual is
    algebraically identical to solving the joint problem directly (a
    standard extension of the Frisch-Waugh-Lovell theorem to quadratic
    penalties). Prediction on new rows (P_pred/phase_pred) is then just the
    ordinary additive evaluation Z_pred @ gamma + P_pred @ beta — no further
    transformation of P_pred is needed beyond whatever scaler the caller
    already applied.

    Parameters
    ----------
    P_train, P_pred : 2D arrays (n_samples, n_pillows), already standardized
        by the caller using a scaler fit on P_train's source data only.
    y_train : 1D array (n_samples,)
    alpha : float, ridge penalty strength for the pillow block.
    phase_train, phase_pred : 1D arrays or None, raw (unstandardized) 0/1
        phase indicator aligned with P_train/P_pred. Pass None for M4.

    Returns
    -------
    y_pred : 1D array (len(P_pred),) predictions.
    beta   : 1D array (n_pillows,) penalized pillow coefficients.
    gamma  : 1D array (1,) or (2,) unpenalized coefficients — [intercept]
             (M4) or [intercept, phase_coef] (M5).
    """
    n_train = P_train.shape[0]
    if phase_train is not None:
        Z_train = np.column_stack([np.ones(n_train), phase_train])
    else:
        Z_train = np.ones((n_train, 1))

    # Partial Z out of the pillow columns and the target via OLS projection.
    P_fit_coefs, *_ = np.linalg.lstsq(Z_train, P_train, rcond=None)
    P_resid = P_train - Z_train @ P_fit_coefs

    y_fit_coefs, *_ = np.linalg.lstsq(Z_train, y_train, rcond=None)
    y_resid = y_train - Z_train @ y_fit_coefs

    ridge = linear_model.Ridge(alpha=alpha, fit_intercept=False)
    ridge.fit(P_resid, y_resid)
    beta = ridge.coef_

    # Recover the unpenalized (intercept [, phase]) coefficients.
    gamma, *_ = np.linalg.lstsq(Z_train, y_train - P_train @ beta, rcond=None)

    n_pred = P_pred.shape[0]
    if phase_pred is not None:
        Z_pred = np.column_stack([np.ones(n_pred), phase_pred])
    else:
        Z_pred = np.ones((n_pred, 1))

    y_pred = Z_pred @ gamma + P_pred @ beta
    return y_pred, beta, gamma


def _select_ridge_alpha_loyo(P_train, y_train, years_train, alphas, phase_train=None):
    """
    Inner LOWYO ridge-alpha search, confined to one outer training fold.

    For each inner-held-out water year found in `years_train`, fits a fresh
    StandardScaler (on the inner-training rows only) and, for every
    candidate alpha, an inner ridge model via _ridge_partialled_fit_predict()
    on the remaining inner-training years. Squared errors on the
    inner-held-out year are pooled across all inner folds per alpha; the
    alpha with the lowest pooled (aggregated) RMSE is returned.

    The outer held-out year is not present in `years_train` and therefore
    never participates in this search, in the scaler fits, or in any
    candidate model fit.

    Returns
    -------
    float : selected alpha, one of the values in `alphas`. Falls back to the
        middle of the grid if no inner fold could be formed (e.g. only one
        distinct water year in the outer training fold).
    """
    inner_years = np.unique(years_train)

    sq_err_by_alpha = np.zeros(len(alphas))
    n_by_alpha      = np.zeros(len(alphas))

    for inner_y in inner_years:
        inner_test_mask  = years_train == inner_y
        inner_train_mask = ~inner_test_mask

        if inner_train_mask.sum() == 0 or inner_test_mask.sum() == 0:
            continue

        P_inner_train_raw = P_train[inner_train_mask]
        P_inner_test_raw  = P_train[inner_test_mask]
        y_inner_train     = y_train[inner_train_mask]
        y_inner_test      = y_train[inner_test_mask]

        phase_inner_train = phase_train[inner_train_mask] if phase_train is not None else None
        phase_inner_test  = phase_train[inner_test_mask]  if phase_train is not None else None

        # Inner scaler: fit on inner-training rows only (never the inner or
        # outer held-out rows).
        inner_scaler  = StandardScaler().fit(P_inner_train_raw)
        P_inner_train = inner_scaler.transform(P_inner_train_raw)
        P_inner_test  = inner_scaler.transform(P_inner_test_raw)

        for a_idx, alpha in enumerate(alphas):
            y_pred_inner, _, _ = _ridge_partialled_fit_predict(
                P_inner_train, y_inner_train, P_inner_test, alpha,
                phase_train=phase_inner_train, phase_pred=phase_inner_test,
            )
            sq_err_by_alpha[a_idx] += np.sum((y_pred_inner - y_inner_test) ** 2)
            n_by_alpha[a_idx]      += len(y_inner_test)

    valid = n_by_alpha > 0
    if not valid.any():
        # Degenerate outer fold (e.g. a single distinct training water
        # year) — no inner fold could be formed. Fall back to the middle
        # of the candidate grid rather than leaving alpha unregularized.
        return alphas[len(alphas) // 2]

    rmse_by_alpha = np.full(len(alphas), np.inf)
    rmse_by_alpha[valid] = np.sqrt(sq_err_by_alpha[valid] / n_by_alpha[valid])

    return alphas[np.argmin(rmse_by_alpha)]


def run_ridge_cross_val_swe_band(swe_tseries: xr.DataArray,
                                 insitu_obs_ds: xr.Dataset,
                                 band_label: str = '',
                                 melt_dates=None,
                                 alphas=np.logspace(-4, 4, 50),
                                 showOutput_: bool = False) -> tuple:
    """
    Run outer LOWYO ridge regression (M4/M5) to predict elevation-band mean
    SWE from all eligible insitu pillows simultaneously (no stepwise
    selection). Mirrors run_cross_val_swe_band()'s outer LOWYO fold
    structure and feature-matrix construction (same summarize_data() /
    NaN-drop-pillow logic), but replaces greedy OLS selection with a ridge
    fit on all eligible pillows, with alpha chosen via
    _select_ridge_alpha_loyo() using only the outer training fold.
    'phase' (when melt_dates is supplied) is always included and is never
    penalized — see _ridge_partialled_fit_predict().

    Returns
    -------
    pred_da       : xr.DataArray, dim 'date', predicted SWE (mm).
    station_names : list[str], all eligible pillow names (comma-joined),
                    repeated once per water year (ridge does not select a
                    subset, so this is identical across years/folds).
    errors        : list[float], outer LOWYO RMSE (mm) per water year.
    alphas_sel    : list[float], ridge alpha selected per water year.
    years         : np.ndarray, water years corresponding to the above.
    beta_da, phase_coef_da, intercept_da, scaler_mean_da, scaler_scale_da :
        (diagnostic only; do not affect pred_da/errors/alphas_sel above)
        beta_da/scaler_mean_da/scaler_scale_da have dims (year, pillow);
        phase_coef_da/intercept_da have dims (year,). beta is the ridge
        coefficient for a 1-SD change in that pillow's value WITHIN that
        outer training fold (i.e. on the standardized scale that fold's own
        outer_scaler was fit on -- not a raw mm/mm slope, and not directly
        comparable across folds without scaler_mean/scaler_scale). gamma's
        intercept and phase coefficient are already in raw mm (phase itself
        is never standardized).
    """
    start_wy = int(swe_tseries.date.dt.year.values.min())
    end_wy   = int(swe_tseries.date.dt.year.values.max())
    years    = np.unique(swe_tseries.date.dt.year.values)
    n_dates  = swe_tseries.shape[0]

    if swe_tseries.isnull().any():
        print(f'[{band_label}] Warning: NaN values in target — returning NaN predictions.')
        pred_da = xr.DataArray(
            np.full(n_dates, np.nan), dims=['date'],
            coords={'date': swe_tseries.date}, name='swe_yhat',
        )
        # Pillow identity isn't known yet at this point; use an empty pillow
        # coordinate (see run_cross_val_swe_band's identical handling).
        nan_year        = xr.DataArray(np.full(len(years), np.nan), dims=['year'], coords={'year': years})
        nan_year_pillow = xr.DataArray(
            np.full((len(years), 0), np.nan), dims=['year', 'pillow'],
            coords={'year': years, 'pillow': np.array([], dtype=object)},
        )
        return (pred_da, [np.nan] * len(years), [np.nan] * len(years), [np.nan] * len(years), years,
                nan_year_pillow, nan_year, nan_year, nan_year_pillow, nan_year_pillow)

    # Feature matrix — identical construction to the OLS (M1-M3) path.
    summary_data = summarize_data(
        [insitu_obs_ds[s] for s in insitu_obs_ds.data_vars],
        swe_tseries,
        melt_dates=melt_dates,
    )
    all_pils = np.array(
        list(insitu_obs_ds.data_vars) + (['phase'] if melt_dates is not None else [])
    )

    # Drop pillows with any missing data across all dates (same rule as M1-M3).
    keep           = ~np.isnan(summary_data).any(axis=1)
    remaining_data = summary_data[keep]
    remaining_pils = all_pils[keep[:-1]]

    phase_match = np.where(remaining_pils == 'phase')[0]
    phase_row   = int(phase_match[0]) if phase_match.size else None

    pillow_rows  = [i for i in range(len(remaining_pils)) if remaining_pils[i] != 'phase']
    pillow_names = [remaining_pils[i] for i in pillow_rows]

    P_all = remaining_data[pillow_rows, :].T  # (n_dates, n_pillows)
    P_all[~np.isfinite(P_all)] = 0
    y_all = remaining_data[-1, :]
    phase_all = remaining_data[phase_row, :] if phase_row is not None else None
    years_all = swe_tseries.date.dt.year.values

    preds, errors, alphas_sel = [], [], []
    beta_lst, phase_coef_lst, intercept_lst = [], [], []
    scaler_mean_lst, scaler_scale_lst = [], []
    for y in range(start_wy, end_wy + 1):
        test_mask = years_all == y
        if test_mask.sum() == 0:
            if showOutput_: print(f'No flights for: {y}')
            continue
        train_mask = ~test_mask

        P_train, P_test = P_all[train_mask], P_all[test_mask]
        y_train, y_test = y_all[train_mask], y_all[test_mask]
        years_train      = years_all[train_mask]
        phase_train = phase_all[train_mask] if phase_all is not None else None
        phase_test  = phase_all[test_mask]  if phase_all is not None else None

        # Inner LOWYO alpha search — confined to the outer training fold;
        # the outer held-out year (y) is excluded from P_train/years_train
        # and therefore never enters scaling or alpha selection.
        alpha_sel = _select_ridge_alpha_loyo(
            P_train, y_train, years_train, alphas, phase_train=phase_train,
        )

        # Outer scaler: fit on the outer-training fold only, then applied
        # to both the outer-training data and the outer held-out year.
        outer_scaler   = StandardScaler().fit(P_train)
        P_train_scaled = outer_scaler.transform(P_train)
        P_test_scaled  = outer_scaler.transform(P_test)

        y_pred, beta, gamma = _ridge_partialled_fit_predict(
            P_train_scaled, y_train, P_test_scaled, alpha_sel,
            phase_train=phase_train, phase_pred=phase_test,
        )

        if showOutput_:
            fold_rmse = np.sqrt(np.mean((y_pred - y_test) ** 2))
            print(f"{y}: alpha={alpha_sel:.4g}, RMSE={fold_rmse:.2f}")

        preds.extend(y_pred.tolist())
        errors.append(float(np.sqrt(np.mean((y_pred - y_test) ** 2))))
        alphas_sel.append(float(alpha_sel))

        # Diagnostic-only: beta/gamma/scaler are already fully computed
        # above for this fold's prediction; just retain them (nothing about
        # fitting, alpha selection, or prediction changes).
        beta_lst.append(beta)
        intercept_lst.append(float(gamma[0]))
        phase_coef_lst.append(float(gamma[1]) if len(gamma) > 1 else np.nan)
        scaler_mean_lst.append(outer_scaler.mean_)
        scaler_scale_lst.append(outer_scaler.scale_)

    preds = np.array(preds)
    preds[preds < 0] = 0
    station_names = [', '.join(pillow_names)] * len(years)

    pred_da = xr.DataArray(
        preds, dims=['date'],
        coords={'date': swe_tseries.date},
        name='swe_yhat',
    )

    beta_da = xr.DataArray(
        np.array(beta_lst), dims=['year', 'pillow'],
        coords={'year': years, 'pillow': pillow_names},
    )
    scaler_mean_da = xr.DataArray(
        np.array(scaler_mean_lst), dims=['year', 'pillow'],
        coords={'year': years, 'pillow': pillow_names},
    )
    scaler_scale_da = xr.DataArray(
        np.array(scaler_scale_lst), dims=['year', 'pillow'],
        coords={'year': years, 'pillow': pillow_names},
    )
    phase_coef_da = xr.DataArray(np.array(phase_coef_lst), dims=['year'], coords={'year': years})
    intercept_da  = xr.DataArray(np.array(intercept_lst), dims=['year'], coords={'year': years})

    return (pred_da, station_names, errors, alphas_sel, years,
            beta_da, phase_coef_da, intercept_da, scaler_mean_da, scaler_scale_da)


def run_ridge_swe_band_cross_validation(ds_A_tseries, insitu_obs_ds, melt_dates=None,
                                        alphas=np.logspace(-4, 4, 50), showOutput=False):
    """
    Ridge-regression analogue of run_swe_band_cross_validation() (M4/M5):
    runs run_ridge_cross_val_swe_band() across every elevation band in
    ds_A_tseries. All eligible pillows are used simultaneously (no stepwise
    selection); the L2 penalty strength (alpha) is chosen per elevation
    band / outer water-year fold via an inner LOWYO search confined to that
    fold's training years (see _select_ridge_alpha_loyo).

    Pass melt_dates for M5 (phase forced in as an additive, unpenalized
    covariate); pass None (default) for M4 (no phase) — same calling
    convention as run_swe_band_cross_validation().

    Output
    ------
    ds_agg_tseries  : xr.Dataset, dims (date, elev), var 'swe_yhat'
                      Same layout as run_swe_band_cross_validation().
    ds_agg_stations : xr.Dataset, dims (elev, year), var 'pillow_sel'
                      All eligible pillow names per band (ridge does not
                      select a subset).
    ds_agg_errors   : xr.Dataset, dims (elev, year), var 'cross_error'
                      Outer LOWYO RMSE (mm) per elevation band / water year.
    ds_agg_alpha    : xr.Dataset, dims (elev, year), var 'alpha_sel'
                      Ridge alpha selected by the inner LOWYO search, per
                      elevation band / outer water-year fold.
    ds_agg_coefs    : xr.Dataset, diagnostic only (does not affect
                      ds_agg_tseries/ds_agg_stations/ds_agg_errors/
                      ds_agg_alpha above).
                      dims (elev, year, pillow) for vars 'pillow_beta',
                      'scaler_mean', 'scaler_scale'; dims (elev, year) for
                      vars 'phase_coef', 'intercept'. 'pillow' coordinate
                      matches the eligible pillow names for that elevation
                      band (identity via named coordinate, not position).
                      'pillow_beta' is the ridge coefficient for a 1-SD
                      change in that pillow's value WITHIN that outer
                      training fold's own standardization (i.e. not a raw
                      mm/mm slope, and not directly comparable across folds
                      without 'scaler_mean'/'scaler_scale', which are kept
                      separately rather than used to unstandardize here).
                      'intercept'/'phase_coef' are already in raw mm, since
                      neither the intercept nor phase is standardized.
    """
    elev_labels = ds_A_tseries.elev.values

    pred_das, station_das, error_das, alpha_das = [], [], [], []
    beta_das, phase_coef_das, intercept_das = [], [], []
    scaler_mean_das, scaler_scale_das = [], []
    for elev in elev_labels:
        swe_tseries = ds_A_tseries['aso_swe'].sel(elev=elev)

        (pred_da, station_names, errors, alphas_sel, years,
         beta_da, phase_coef_da, intercept_da, scaler_mean_da, scaler_scale_da) = run_ridge_cross_val_swe_band(
            swe_tseries,
            insitu_obs_ds,
            band_label=str(elev),
            melt_dates=melt_dates,
            alphas=alphas,
            showOutput_=showOutput,
        )

        pred_das.append(pred_da)
        station_das.append(xr.DataArray(station_names, dims=['year'], coords={'year': years}))
        error_das.append(xr.DataArray(errors, dims=['year'], coords={'year': years}))
        alpha_das.append(xr.DataArray(alphas_sel, dims=['year'], coords={'year': years}))
        beta_das.append(beta_da)
        phase_coef_das.append(phase_coef_da)
        intercept_das.append(intercept_da)
        scaler_mean_das.append(scaler_mean_da)
        scaler_scale_das.append(scaler_scale_da)

    elev_index = pd.Index(elev_labels, name='elev')

    ds_agg_tseries  = xr.concat(pred_das, dim=elev_index).transpose('date', 'elev').to_dataset()
    ds_agg_stations = xr.concat(station_das, dim=elev_index).to_dataset(name='pillow_sel')
    ds_agg_errors   = xr.concat(error_das, dim=elev_index).to_dataset(name='cross_error')
    ds_agg_alpha    = xr.concat(alpha_das, dim=elev_index).to_dataset(name='alpha_sel')

    ds_agg_coefs = xr.Dataset({
        'pillow_beta':   xr.concat(beta_das, dim=elev_index).transpose('elev', 'year', 'pillow'),
        'phase_coef':    xr.concat(phase_coef_das, dim=elev_index),
        'intercept':     xr.concat(intercept_das, dim=elev_index),
        'scaler_mean':   xr.concat(scaler_mean_das, dim=elev_index).transpose('elev', 'year', 'pillow'),
        'scaler_scale':  xr.concat(scaler_scale_das, dim=elev_index).transpose('elev', 'year', 'pillow'),
    })

    return ds_agg_tseries, ds_agg_stations, ds_agg_errors, ds_agg_alpha, ds_agg_coefs


# ===========================================================================
# ADDED -- forward prediction (fit on all years, then apply)
# ===========================================================================

def _build_band_matrix(swe_tseries, insitu_obs_ds, melt_dates=None):
    """
    Feature matrix for one elevation band, identical in construction to
    run_ridge_cross_val_swe_band()'s -- same summarize_data() call, same
    drop-pillows-with-any-NaN rule, same phase handling. Factored out so fit
    and cross-validation cannot drift apart.

    Returns (P, y, phase, pillow_names, years) where P is (n_dates, n_pillows).
    """
    summary_data = summarize_data(
        [insitu_obs_ds[s] for s in insitu_obs_ds.data_vars],
        swe_tseries,
        melt_dates=melt_dates,
    )
    all_pils = np.array(
        list(insitu_obs_ds.data_vars) + (['phase'] if melt_dates is not None else [])
    )

    keep           = ~np.isnan(summary_data).any(axis=1)
    remaining_data = summary_data[keep]
    remaining_pils = all_pils[keep[:-1]]

    phase_match = np.where(remaining_pils == 'phase')[0]
    phase_row   = int(phase_match[0]) if phase_match.size else None

    pillow_rows  = [i for i in range(len(remaining_pils)) if remaining_pils[i] != 'phase']
    pillow_names = [str(remaining_pils[i]) for i in pillow_rows]

    P = remaining_data[pillow_rows, :].T
    P[~np.isfinite(P)] = 0
    y = remaining_data[-1, :]
    phase = remaining_data[phase_row, :] if phase_row is not None else None
    years = swe_tseries.date.dt.year.values
    return P, y, phase, pillow_names, years


def fit_ridge_band(swe_tseries: xr.DataArray,
                   insitu_obs_ds: xr.Dataset,
                   melt_dates=None,
                   alphas=np.logspace(-4, 4, 50)) -> dict:
    """
    Fit ONE ridge model on every available water year, for forward prediction.

    This is the operational counterpart to run_ridge_cross_val_swe_band(), which
    holds each water year out in turn and returns cross-validated predictions --
    a skill estimate, not a forecast. Here nothing is held out: alpha is still
    chosen by the same inner LOWYO search (so it is not tuned on data the model
    will be judged against), but the final fit uses all years.

    Returns a model dict consumable by predict_ridge_band():
        pillow_names : list[str]  column order the model expects, in order
        beta         : (n_pillows,) penalised coefficients, standardized scale
        gamma        : (1,) [intercept] or (2,) [intercept, phase_coef], raw mm
        scaler       : fitted StandardScaler (pillow columns only)
        alpha        : selected ridge penalty
        has_phase    : bool
        n_train      : number of flight dates the fit saw
        train_years  : sorted water years present
    """
    P, y, phase, pillow_names, years = _build_band_matrix(
        swe_tseries, insitu_obs_ds, melt_dates=melt_dates
    )
    if P.shape[1] == 0 or not np.isfinite(y).all():
        return {'pillow_names': pillow_names, 'beta': None, 'gamma': None,
                'scaler': None, 'alpha': np.nan, 'has_phase': phase is not None,
                'n_train': int(P.shape[0]), 'train_years': sorted(set(years.tolist()))}

    alpha  = _select_ridge_alpha_loyo(P, y, years, alphas, phase_train=phase)
    scaler = StandardScaler().fit(P)
    P_std  = scaler.transform(P)

    # fit and predict on the same rows: we want the coefficients, not the predictions.
    _, beta, gamma = _ridge_partialled_fit_predict(
        P_std, y, P_std, alpha, phase_train=phase, phase_pred=phase
    )
    return {'pillow_names': pillow_names, 'beta': beta, 'gamma': gamma,
            'scaler': scaler, 'alpha': float(alpha), 'has_phase': phase is not None,
            'n_train': int(P.shape[0]), 'train_years': sorted(set(years.tolist()))}


def predict_ridge_band(model: dict,
                       insitu_obs_ds: xr.Dataset,
                       dates,
                       phase=None) -> np.ndarray:
    """
    Apply a fitted band model to arbitrary dates -- typically every day of a
    target water year.

    insitu_obs_ds must contain every pillow in model['pillow_names']; ridge
    carries a coefficient per pillow, so unlike the OLS combination approach
    there is no dropping a missing one. Feed it the IMPUTED observations.

    A pillow present but NaN on a given date yields NaN for that date rather
    than silently contributing zero -- the caller should see the gap.

    phase : required iff the model was fitted with a phase term; 0/1 per date.
    """
    if model.get('beta') is None:
        return np.full(len(pd.to_datetime(dates)), np.nan)

    dates = pd.to_datetime(dates)
    missing = [p for p in model['pillow_names'] if p not in insitu_obs_ds.data_vars]
    if missing:
        raise KeyError(f'pillows absent from insitu_obs_ds: {missing}')

    cols = [insitu_obs_ds[p].sel(time=dates, method='nearest').values
            for p in model['pillow_names']]
    P = np.column_stack(cols).astype(float)

    bad = ~np.isfinite(P).all(axis=1)          # any missing pillow -> no prediction
    P_filled = np.where(np.isfinite(P), P, 0.0)
    P_std = model['scaler'].transform(P_filled)

    if model['has_phase']:
        if phase is None:
            raise ValueError('model was fitted with a phase term; pass phase=')
        Z = np.column_stack([np.ones(len(dates)), np.asarray(phase, float)])
    else:
        Z = np.ones((len(dates), 1))

    yhat = Z @ model['gamma'] + P_std @ model['beta']
    yhat[bad] = np.nan
    return yhat
