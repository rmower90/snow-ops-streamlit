# Pending fixes

Defects found while running the WY2026 model experiments, **deliberately not applied yet**.

Every item here changes numbers. Applying any of them mid-experiment would mean arm 3 differs
from arms 1 and 2 by more than the model, so they are parked until the comparison is done.

Each entry records where a working fix already exists, since several were already solved on
the separate HPC and only need porting back.

---

## A. Correctness — these change results

### A1. Imputation compares adjusted R² across different datasets

`preprocessing.imputation_w_pillows`, three-level forward donor search.

Level 1 scores single donors on the **whole overlapping record**; level 2 scores pairs on
**2013 onward only** (`train_df_[train_df_.index.year >= 2013]`). The stopping test compares
them directly:

```python
if second_corr_adjr2 > best_corr_adjr2:      # level 2 vs level 1 — different n, different span
```

Measured across all 750 USCASJ pillow pairs: mean adjusted R² is **0.702** on the full record
against **0.850** on 2013+, and 75% of pairs score higher on the shorter record — a **+0.147**
mean uplift that owes nothing to the extra donor. A second predictor nearly always clears a
bar inflated that much, so the search almost always advances. Level 3 compares like with like
(2013+ vs 2013+), and a third predictor with any signal usually nudges adjusted R² up.

Net effect: the search runs to three donors nearly every time. The stopping rule is not
deciding anything — `pillow_impute_threePils` is descriptive, not a design choice.

**Fix:** apply the same `>= 2013` filter at level 1. Already fixed on the other HPC.
**Blast radius:** every predict-NaNs result in every arm.

### A2. `fit_best_model` selects complexity on the held-out fold

`lm_model.fit_best_model`.

```python
while (error <= best_error) and (len(parameters) < max_params):
    ...
    error = np.sqrt(np.mean((lm.predict(x_hat[:, new_parameters]) - y_hat)**2))
```

`x_hat`/`y_hat` are the test fold. Predictors keep being added for as long as **test** RMSE
improves, so the number of predictors is chosen by looking at the answer. The comment above
the loop claims adjusted R², which is what it was evidently meant to do.

Live on the operational path:
`train_all_mlr_models → run_cross_val_selection2 → cross_val_loyo_pred_select →
compute_linear(x, y, x_hat, y_hat) → fit_best_model`.
(`cross_val_loo` passes no `y_hat` and takes the plain-OLS branch — that route is unaffected.)

`max_params` also defaults to `x.shape[0] - 1`, so a 5-date melt record permits 4 predictors.

**Fix exists** in `scripts/experiments/ridge_linear.py`'s upstream source: training adjusted
R² as the criterion, `max_params = max(1, min(3, n_train // 3))`, and a
degrees-of-freedom guard. Test RMSE is computed once after selection, for reporting only.
**Blast radius:** reported CV skill is optimistic; selected station sets may change.

### A3. `add_parameter` can append a duplicate predictor

`lm_model.add_parameter`.

```python
corr = np.zeros(x.shape[1])          # every column starts at 0
for i in range(x.shape[1]):
    if i in param: pass              # already-selected stay at 0
    else:          corr[i] = r2
best_param = np.argmax(corr)         # argmax over ALL columns, selected included
```

When no candidate beats 0 — every candidate non-finite, or a tie at zero — `np.argmax` returns
index 0, which may already be in `param`. The duplicate makes the design matrix singular and
still counts toward `max_params`.

Flagged in `docs/mlr_cv_audit_2026.md` (branch `mlr-cv-audit`) as "confirmed, not fixed".

**Fix exists** in the ridge source: restrict to candidates (`max(candidates, key=...)`) and
return unchanged when none remain.

### A4. `saveImputeCSV=False` still reads a stale cache

`preprocessing.imputation_w_pillows` and `imputation_w_snowmodel`.

```python
if not os.path.exists(fpath) or (saveImputeCSV == False):
    ... build fresh ...
    if saveImputeCSV: df.to_csv(fpath)

if os.path.exists(fpath):            # unconditional
    df = pd.read_csv(fpath)          # discards the fresh build
```

`saveImputeCSV=False` correctly forces a rebuild and correctly skips the write — then reads
any pre-existing file back over the result. It means "don't write" but delivers "use someone
else's anyway". A fresh working directory hides it; a second run in the same directory
returns the first run's table regardless of what changed.

**Fix:** only read back when the build was skipped.

---

## B. Already fixed on `model-experiments` — land on `main`

Neither is experiment scaffolding; both are production defects.

**B1. `preprocessing.py` — `makedirs` race.** Check-then-create on the imputation directory.
The nightly chain submits six jobs at once against a shared directory, so this fires the
first time anyone stands up a basin with no `imputation/` yet. It killed a job during arm 2.

**B2. `mlr_prediction.py` — `realpath` for git provenance.** The script runs through a symlink
in `bin/`, whose directory is not a git repo, so `abspath` made every manifest record commit,
branch and dirty as `null`.

---

## C. Robustness and hygiene

**C1. Shell scripts call bare `python`.** `copy_generate_html.sh` and `run_prediction.sh`
activate a conda environment then invoke `python`, which resolves by `PATH` rather than from
the environment just activated. Caused three separate failures in one session — a missing
`pyarrow`, a missing `xarray`, and a `proj.db` version mismatch. Cron survives on a clean
`PATH`; anything else does not. Fix: call the explicit interpreter path.

**C2. Config drift.** `load_aso_metadata` reads `/home/rossamower/work/aso/configs/`, not the
repo. The repo copy was seven months stale before 2026-10-07 and will drift again — a
snapshot was committed, not a link. Fix: symlink the runtime config tree into the repo the way
`pipeline/` and `scripts/` are.

**C3. QA persistence rule misses short extreme outliers.** `build_majority_qa_ds` masks only
when a majority persists ≥3 consecutive days. KUP read 13,107 mm on 2026-06-20/21 — about 12x
the largest legitimate value in the basin — and all three methods flagged it, but it lasted
two days and survived into WY2026. Fix: a magnitude escape hatch that bypasses the
persistence requirement.

**C4. Water-year hardcoding.** Eleven files plus the region config. Documented separately in
`wy2027_rollover.md`; listed here only so this file is a complete index.

---

## Suggested order, once experiments are done

1. **B1, B2** — already written and tested, no effect on results
2. **C1** — environment only, no effect on results
3. **A4** — narrow, and makes the rest safer to verify
4. **A1** — then regenerate the arms, since every predict-NaNs number moves
5. **A2, A3** — together; both touch greedy selection and both have fixes upstream
6. **C2, C3** — independent, any time

A1 through A3 all invalidate the current arm comparison. If they are applied, all arms should
be regenerated together rather than piecemeal.
