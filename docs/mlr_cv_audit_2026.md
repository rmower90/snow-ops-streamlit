# MLR pillow-selection cross-validation audit — 2026

Frozen record of a methodological audit of the linear-regression snow-pillow selection
workflow in `scripts/lm_model.py`. The audit is **complete**; the statistical redesign it
points to was **deliberately not implemented**. See *Direction* at the end.

---

## 1. The workflow as it exists

Per `(imputation mode, elevation band)`, `run_cross_val_selection2()` runs:

```
CV #1   cross_val_loyo_pred_select()  — leave-one-YEAR-out. Per fold, fit_best_model() does
        forward selection. Each fold returns its own selected pillow list.
          |
          v
pool    station_lst = list(set(sum(best_stations, [])))   — union across folds
          |
          v
CV #2   identify_best_stations()  — enumerates combinations(station_lst, 1..5), scores each
        with cross_val_loo() (leave-one-FLIGHT-out), keeps the argmax adjusted R2
          |
          v
final   cross_val_loyo_pred_select() re-run on the winning combination
        -> predictions_validation2 -> RMSE_MM reported in summary_dict
```

Scale, FRIANT: 31–35 ASO flights across 8–9 years (~3.9 flights/year); candidate universe
10–19 pillows depending on availability and imputation mode.

---

## 2. Problems identified

**P1 — CV #1 stopping rule uses the held-out year.**
In `fit_best_model()`, `add_parameter()` chooses *which* pillow enters using training data
only (clean), but the loop exit is driven by
`error = RMSE(lm.predict(x_hat[:, new_parameters]), y_hat)` where `y_hat` is the held-out
year's ASO. So the nominal test year determines *how far* selection proceeds. The returned
`best_error` is the held-out RMSE of a model whose size that same held-out data chose; it is
printed per fold and consumed nowhere else.

**P2 — CV #2 ranks with leave-one-FLIGHT-out.**
Several ASO flights occur per water year and snowpack is strongly autocorrelated within a
season, so holding out one flight leaves ~3 near-duplicate siblings from the same year in
training. The ranking statistic is therefore optimistic.

**P3 — Final validation is post-selection.**
`predictions_validation2` blocks whole years for the *coefficients*, so it is better than a
training RMSE. But the combination being evaluated was chosen using all years. `RMSE_MM` is
consequently not an independent estimate of the whole selection procedure.

**P4 — Model-complexity criterion is unresolved.**
Separate from the leakage: how far should forward selection proceed, and how should
combinations be ranked? Phase 1 shows the current rule strongly constrains the candidate
universe, but does **not** establish that minimum inner-CV RMSE is the right replacement.

---

## 3. Phase 0 — flight-wise vs year-wise ranking

**Design.** Reproduce the exact candidate set production enumerated (CV #1 -> `station_lst`
-> `combinations(1..5)`), then score every one of those *same* combinations both ways:
`cross_val_loo` (flight-blocked, production) and `cross_val_loyo` (year-blocked, fixed
feature set, no feature selection inside the fold). Today's prediction is computed once per
combination and used as a control — it must be identical under both, and was.

Cases: FRIANT `predict NaNs`, band `total`, 2025-04-01 (31 combinations) and 1983-04-01 (15).

| | WY2025 | WY1983 |
|---|---|---|
| optimism, adj R² (median flight − year) | +0.0475 | +0.0428 |
| optimism, RMSE (median year − flight) | +20.9 mm | +18.6 mm |
| RMSE ratio year/flight (median) | 1.31× | 1.26× |
| rank-1 changed | **yes** | **yes** |
| Spearman (adj R²) | +0.789 | **+0.250** |
| top-10 overlap | 9/10 | 6/10 |
| top-10 adj R² separation, flight → year | 0.0090 → **0.0517** | 0.0149 → **0.0552** |
| top-10 prediction spread, flight → year | 43.9 → 43.9 mm | 207.3 → **273.0** mm |

Rank-1 models:

| | flight-wise | year-wise |
|---|---|---|
| WY2025 | `KSP,LLE,SWM` adjR² .9645 RMSE 50.6 → 347.7 mm | `LLE,MHP,SWM` adjR² .9593 RMSE 54.5 → 359.6 mm |
| WY1983 | `WWC,KSP,PSR,STL` adjR² .9392 RMSE 62.2 → 835.1 mm | `PSR,STL` adjR² .9162 RMSE 76.6 → 821.8 mm |

Re-scoring the flight-wise winner year-wise: WY2025 .9645 → **.9241** (RMSE 50.6 → 75.1);
WY1983 .9392 → **.8611** (62.2 → 95.2). The flight-wise winner is specifically the
combination that benefits most from within-season leakage.

Jackknife (best combination excluding each pillow):

| | flight-wise | year-wise |
|---|---|---|
| WY2025 | base 347.7, span 38.1 mm, worst LLE +21.2 | base 359.6, span 34.0 mm, worst SWM −18.9 |
| WY1983 | base 835.1, span 161.9 mm, worst STL +148.6 | base 821.8, span 273.0 mm, worst PSR −138.7 |

---

## 4. Phase 1 — CV #1 stopping rule

**Design.** Per outer year, replay the training-derived entry sequence using production's
`add_parameter()` verbatim, then compare two stopping rules along that *identical* sequence:
the current rule (stop when the outer year's RMSE stops improving) versus minimum **inner
leave-one-year-out CV RMSE** computed entirely within the outer training years. The outer
year's ASO is touched once per rule, to score the final prediction.

| | WY2025 (8 outer years, 19 features) | WY1983 (9 outer years, 10 features) |
|---|---|---|
| mean model size, current | **1.75** | **1.89** |
| mean model size, inner-CV | **6.75** | **4.78** |
| folds where selection changed | 8/8 (100%) | 8/9 (89%) |
| `station_lst` current → inner | **5 → 16** (Jaccard 0.31) | **4 → 9** (Jaccard 0.44) |
| `only-current` pillows | none | none |
| pooled outer RMSE, current (optimistic) | 89.99, R² .9010 | 85.80, R² .8977 |
| pooled outer RMSE, inner (honest) | 91.95, R² .8952 (+2.2%) | 90.40, R² .8920 (+5.4%) |
| outer prediction change, mean / max | 33.6 / 68.6 mm | 31.4 / 89.1 mm |

**The leaky rule selects far simpler models, not more complex ones.** Each fold has only 2–6
test flights, so the held-out RMSE is very noisy and the loop exits the first time noise
makes an addition look unhelpful — five of eight WY2025 folds stop at one pillow. This is
erratic *premature stopping*, not overfitting-by-leakage.

`only-current` being empty in both cases means the current rule purely **under-selects**: it
never nominates a pillow the honest rule would not.

**Cost consequence.** `station_lst` 5 → 16 turns `identify_best_stations` from
`C(5,1..5)` = 31 combinations into `C(16,1..5)` = 6,884 — a **220×** increase. At the
measured ~197 ms/combination that is ~22 min per (band, mode) instead of ~6 s, roughly
6 hours per date across 8 bands × 2 modes. Not viable daily without per-water-year caching.

---

## 5. Duplicate-predictor defect (confirmed, not fixed)

`add_parameter()` initialises `corr = np.zeros(x.shape[1])`, leaves `corr[i] = 0` for any `i`
already in `param`, then does `best_param = np.argmax(corr)` and appends **without a
membership check**. Once every predictor is selected, every entry is 0, `argmax` returns
index **0**, and index 0 is appended again. Reproduced in isolation on a 5-feature problem:

```
step 5: [1, 0, 2, 3, 4]
step 6: [1, 0, 2, 3, 4, 0]        <-- duplicate
step 9: [1, 0, 2, 3, 4, 0, 0, 0, 0]
```

Structurally identical to the historical Tuolumne output `[4, 2, 3, 1, 0, 0, 0, 0, 0]`.

The loop does not self-terminate: a duplicated column is collinear, so the refit changes
predictions by ~4e-15, `error <= best_error` remains true (note `<=`, not `<`), and appending
continues until `len(parameters) >= max_params`. `max_params` defaults to `x.shape[0]-1`,
which is `n_TRAIN_OBS - 1` (~24–31 here) rather than the intended `n_features` — likely a
separate typo, and it is why the run is bounded at all.

**Current impact is low.** `station_lst` is built with `set(...)`, which dedupes, and the
collinear refit is numerically identical, so no downstream number is corrupted. The cost is
wasted iterations and a misleading per-fold feature list. It was not reached in either Phase
0 or Phase 1 (universes of 10 and 19 exceeded the 8-step cap used in the Phase 1 replay).

**Smallest safe fix, if ever needed** — two guarded additions, deliberately *not* applied:

```python
# add_parameter: only fires in the case that currently yields a duplicate
best_param = np.argmax(corr)
if best_param in param:
    return param.copy()

# fit_best_model: exit when the sequence stops growing
prev = len(parameters)
new_parameters = add_parameter(parameters, x, y)
if len(new_parameters) == prev:
    break
```

---

## 6. What the audit supports / does not support

**Supported**

- P1 and P2 are real and confirmed by direct code trace and reproduction.
- Year-blocked ranking raises RMSE ~1.26–1.31× and changes rank 1 in both tested cases.
- The apparent near-equivalence of top candidate models is largely an artefact: year-blocked
  scoring widens top-10 adj R² separation 3.7–5.7×.
- The current stopping rule under-selects severely (mean 1.8 vs 4.8–6.8 pillows) and shrinks
  the candidate universe 2–3×.
- The **prediction spread** across plausible combinations is robust — unchanged in WY2025 and
  larger in WY1983. Near-equal-scoring combinations really do give materially different basin
  estimates.
- The jackknife/exposure *magnitude* survives year-blocking (~20% of the estimate in the
  sparse case), though the identity of the critical station changes.
- The duplicate-predictor path is reachable and explains the historical Tuolumne output.

**Not supported / unresolved**

- That minimum inner-CV RMSE is the correct replacement stopping rule. No complexity penalty
  was applied; k = 7–8 predictors on ~30 observations is a lot, and a 1-SE rule or
  regularised regression would likely choose smaller sets. Untested.
- That a larger `station_lst` yields better *final* predictions. Phase 1 measured CV #1 only;
  the downstream effect on `identify_best_stations`, the winning combination, and `RMSE_MM`
  was not run.
- Any statement that one model is "almost as good as" another on adj R² grounds. That
  argument rested on the flight-blocked statistic.
- Sparse-year sensitivity as a clean result. All folds within a date share the same
  training-year count, so there was no within-date variation to test.
- The Phase 1 WY2025 divergence is a **lower bound**: four of eight folds selected exactly
  `MAX_STEPS = 8`, so the inner optimum may lie beyond the cap.
- Scope throughout: FRIANT, `predict NaNs`, band `total`, two dates. Not USCASJ, not
  `drop NaNs`, not per-elevation-band.

---

## 7. Pragmatic design vs unbiased validation

These are not simply coding mistakes. The algorithm was built for very sparse basins (~6 ASO
years), where holding out a whole year for model *selection* as well as evaluation can leave
too little data to select on at all. Leave-one-flight-out and the held-out-error stopping
rule are defensible **operational** compromises in that regime.

What they are not is a basis for unbiased generalisation claims. The distinction to carry
forward: the existing numbers are usable as *relative* guidance for network design, and are
not usable as absolute out-of-sample skill or as evidence of statistical equivalence between
candidate networks.

---

## 8. Direction

**Publication.** A nested, year-blocked CV framework: an outer leave-one-year-out loop with
the *entire* selection procedure — CV #1, pooling, CV #2 — inside it, so each outer year is
predicted once by a model that never saw it. The variation in selected network across outer
folds is itself an output: it measures selection stability. Not implemented here.

**Operational.** The daily/historical prediction system is moving to a simpler pretrained
lookup-table approach on a separate branch. The nested redesign is explicitly *not* being
applied to the operational system now.

---

## 9. Reproducing the audit

Scripts (`scripts/experiments/`):

| file | purpose |
|---|---|
| `phase0_cv_scoring_diagnostic.py` | flight-wise vs year-wise scoring of identical candidate sets |
| `phase1_stopping_rule_diagnostic.py` | current vs inner-year-CV stopping rule along the same entry sequence |
| `pillow_selection_table.py` | full candidate table with adj R² + prediction; `--wide` enumerates all available pillows |
| `mlr_ab_harness.py` | A/B instrument for `lm_model.py` changes (determinism + fidelity) |
| `mlr_ab_fidelity.py` | checks harness output against production `_combination.csv` |

Invocations:

```bash
python phase0_cv_scoring_diagnostic.py    FRIANT 2025 2025-04-01
python phase0_cv_scoring_diagnostic.py    FRIANT 1983 1983-04-01
python phase1_stopping_rule_diagnostic.py FRIANT 2025 2025-04-01
python phase1_stopping_rule_diagnostic.py FRIANT 1983 1983-04-01
python pillow_selection_table.py          FRIANT 2025 2025-04-01 [--wide]
```

Outputs, under `/home/rossamower/work/aso/data_test/` (untracked, outside all production
paths): `phase0_cv_scoring/`, `phase1_stopping_rule/`, `pillow_selection/`, `mlr_ab/`.

`cross_val_loyo()` in `scripts/lm_model.py` is the year-blocked scorer the diagnostics
import. It is the **only** change this audit makes to production code, and it is purely
additive — nothing calls it outside the diagnostics, so production behaviour is unchanged.

A `score_mode='flight'|'year'` switch was briefly threaded through `identify_best_stations`,
`run_cross_val_selection2`, `run_mlr_train_predict` and `train_all_mlr_models` during Phase 0.
It was **reverted before this commit**: no diagnostic used it (they call `cross_val_loo` /
`cross_val_loyo` directly), and a dormant switch that silently changes the published model if
flipped is a liability. To rebuild it, dispatch on
`cross_val_loo if score_mode == 'flight' else cross_val_loyo` at the two scoring call sites
inside `identify_best_stations` and thread the keyword up through the four signatures above.

Environment: conda env `bor`. Note that `conda activate bor` sets `CONDA_PREFIX` correctly
but the shell profile prepends another env to `PATH`, so invoke `$CONDA_PREFIX/bin/python`
explicitly when running interactively.
