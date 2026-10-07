# Prediction versions

One row per prediction run that we intend to compare or cite. The point is to be able to
answer "what made this number" months later without archaeology.

Each run also writes a machine-readable `run_manifest.json` beside its outputs, recording
input paths with content hashes, model settings, environment overrides, and the git commit.
This table is the human index over those.

---

## Registry

| ID | Test pillows | Training pillows | Model | Output | Run | Finding |
|---|---|---|---|---|---|---|
| `wy2026-baseline` | blended (82,3), 19 manual QA windows | `pillow_wy_1980_2025_qa1.nc` | combinations 1–5, argmax adj R² | `COMMON_MASK/`, `SNOWMODEL_IMPUTE/` | 2026-10-07 | Published WY2026 record, 2025-10-01 → 2026-09-30. The reference. |
| `wy2026-s82` | **bulk sensor 82**, KUP window only | unchanged | unchanged | `COMMON_MASK_s82/`, `SNOWMODEL_IMPUTE_s82/` | *not yet run* | — |
| `wy2026-s82-ridge` | bulk sensor 82, KUP window only | unchanged | **ridge, inner CV, all pillows** | `COMMON_MASK_s82_ridge/`, `SNOWMODEL_IMPUTE_s82_ridge/` | *not yet run* | — |

All runs: season model, predict-NaNs, 7 elevation bands, both imputation strategies,
`N_ENSEMBLE=10`, basin USCASJ, WY2026.

---

## Why training data is held constant

The sensor-82 experiment varies **test** data only. On the 35 ASO flight dates the model
actually fits to, the blended and bulk-82 training records differ on 6 of 736 cells — five
of those by under 3 mm. Swapping training data would not move the fit; it would only change
imputation-donor availability, which is a different question.

Full accounting: [`training_data_sensor82_aso_comparison.md`](training_data_sensor82_aso_comparison.md).

---

## How to run a version

Every knob defaults to the operational value, so an unset environment reproduces the
baseline exactly.

### QA the test observations

```bash
source /opt/miniforge3/etc/profile.d/conda.sh && conda activate bor
cd /home/rossamower/bin/insitu/qa

INSITU_QA_CONFIG_NAME=USCASJ_s82 \
INSITU_QA_RAW_FILE=sensor82/pillow_wy_2026_sensor82.nc \
INSITU_QA_SUFFIX=_s82 \
/home/rossamower/work/aso/conda/envs/bor/bin/python -u insitu_qa.py USCASJ 2026 10 1
```

Writes `processed/USCASJ_insitu_obs_daily_wy_2026_s82.nc` plus suffixed QA tables and plot
directories. **The suffix is not optional** — without it this overwrites the file
`mlr_prediction.py` reads and the baseline was produced from.

### Predict

```bash
cd /home/rossamower/bin/mlr_prediction
MLR_TEST_OBS_FILE=processed/USCASJ_insitu_obs_daily_wy_2026_s82.nc \
MLR_TEST_OBS_RAW_FILE=sensor82/pillow_wy_2026_sensor82.nc \
MLR_STACK_SUFFIX=_s82 \
MLR_N_ENSEMBLE=10 \
sbatch --export=ALL run_prediction.sh USCASJ 2026 0 0 0 1   # season, pillow imputation
sbatch --export=ALL run_prediction.sh USCASJ 2026 0 0 0 0   # season, snowmodel imputation
```

---

## What the suffix isolates

`MLR_STACK_SUFFIX` must cover every artifact a run writes *or reads back*, not just the
predictions. It currently keys:

| Artifact | Path |
|---|---|
| Predictions, ensemble, manifest | `{mlrPred_dir}/{stack}{SUFFIX}/{model}/` |
| **Imputation cache** | `{mlrPred_dir}/imputation{SUFFIX}/` |

The imputation cache is the non-obvious one. Tables are named
`pillow_impute_*_wy{YYYY}.csv` — keyed by water year and nothing else — and the read-back is
unconditional: once the file exists it is never rebuilt, however the inputs changed. A
shared directory therefore cannot corrupt an earlier run's table, but it will silently feed
a *new* run the *old* one.

That is precisely how FRIANT v2–v6 ended up with their "predict NaNs" half imputed from qa1
while the "drop NaNs" half tracked the newer data. The historic runner was fixed by keying
the cache to the QA rung; the daily runner carried the same defect until this experiment
surfaced it.

**If you add another artifact that is cached and read back, key it by the suffix too.**

---

## Environment knobs

| Variable | Script | Default |
|---|---|---|
| `INSITU_QA_CONFIG_NAME` | `insitu_qa.py` | the basin's own region config |
| `INSITU_QA_RAW_FILE` | `insitu_qa.py` | `raw/{basin}_insitu_obs_daily_wy_2026.nc` |
| `INSITU_QA_SUFFIX` | `insitu_qa.py` | `''` — appended to every output |
| `MLR_TEST_OBS_FILE` | `mlr_prediction.py` | `processed/{basin}_insitu_obs_daily_wy_2026.nc` |
| `MLR_TEST_OBS_RAW_FILE` | `mlr_prediction.py` | `raw/{basin}_insitu_obs_daily_wy_2026.nc` |
| `MLR_STACK_SUFFIX` | `mlr_prediction.py` | `''` — appended to the output directory |
| `MLR_N_ENSEMBLE` | `insitu_qa_domain.sh` | `10` |

---

## Note on scope

Only the **daily** driver (`mlr_prediction.py`) is used for these experiments. The FRIANT
historic runner is a different shape: it splits train and test from one continuous record
rather than consuming a separate test file, trains on qa6 rather than qa1, and has its
SnowModel imputation commented out — so it cannot produce the second half of this
comparison.
