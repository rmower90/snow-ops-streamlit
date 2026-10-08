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
| `wy2026-s82` | **bulk sensor 82**, KUP window only | unchanged | unchanged | `COMMON_MASK_s82/`, `SNOWMODEL_IMPUTE_s82/` | 2026-10-07 | Season only. Helps under SnowModel imputation, hurts under pillow imputation — see below. |
| `wy2026-s82-ridge` | bulk sensor 82, KUP window only | unchanged | **ridge, all pillows, alpha by inner LOWYO** | `COMMON_MASK_s82_ridge/` | 2026-10-08 | Beats every OLS arm mid-season, much worse at the shoulders. Mean MAE 40.4. |
| `wy2026-s82-ridge-phase` | as above | unchanged | ridge + accumulation/melt indicator | `COMMON_MASK_s82_ridge_phase/` | 2026-10-08 | Helps the shoulders, hurts mid-season. Net worse: 46.2. |

All runs: season model, predict-NaNs, 7 elevation bands, both imputation strategies,
`N_ENSEMBLE=10`, basin USCASJ, WY2026.

---

## Results so far

Validated against the WY2026 ASO flights, band-mean SWE, season model, predict-NaNs. MAE in
mm against the observed flight.

| Flight | pillow-imp baseline | pillow-imp s82 | snowmodel-imp baseline | **snowmodel-imp s82** |
|---|---|---|---|---|
| 2026-01-27 | 37.7 | 58.0 | 35.2 | **25.5** |
| 2026-03-28 | 32.5 | 33.6 | 22.9 | **22.0** |
| 2026-04-29 | 29.5 | 39.4 | 27.3 | **27.1** |
| mean | 33.2 | 43.7 | 28.5 | **24.9** |

Two patterns, consistent across all three flights:

**Imputation strategy matters more than the sensor change.** SnowModel imputation beats
pillow imputation in every arm. The sensor switch moves MAE by 1–3 mm; the imputation choice
moves it by about 10 mm.

**Sensor-82 interacts with imputation strategy.** It improves SnowModel imputation on all
three flights and degrades pillow imputation on all three. A plausible reading: the
sensor-82 QA retains 34% more pillow values (7,347 vs 5,489) because it carries 1 exclusion
window instead of 19, and regression against SnowModel grids tolerates those marginal values
better than pillow-to-pillow donor search does.

Three flights, one basin, one water year — directional, not settled. The two remaining
WY2026 flights (03-03, 05-18) have not been checked. WY2026 ASO was binned directly from the
raw GeoTIFFs, since those flights were never ingested into `ASO_50M_SWE_TSERIES.nc`.

---

## Arm 3 — ridge

MAE (mm) against the five WY2026 ASO flights, band-mean, season model, predict-NaNs.

| Flight | OLS s82 | OLS-sm s82 | RIDGE | RIDGE+phase |
|---|---|---|---|---|
| 2026-01-27 | 58.0 | **25.5** | 70.0 | 60.3 |
| 2026-03-03 | 30.3 | 33.1 | **31.5** | 62.4 |
| 2026-03-28 | 33.6 | 22.0 | **20.3** | 33.8 |
| 2026-04-29 | 39.4 | 27.1 | 22.8 | **21.3** |
| 2026-05-18 | 41.8 | 38.2 | 57.4 | 53.0 |
| **mean** | 40.6 | **29.2** | 40.4 | 46.2 |

**Ridge wins three of five flights outright** — all mid-season — and is essentially tied with
the OLS arm using the same pillow imputation (40.4 vs 40.6). It fails badly at the season
edges, January 27 and May 18, which is what drags its mean.

**OLS with SnowModel imputation remains best overall at 29.2.** Ridge has not beaten it in
any configuration tried. Note ridge used *pillow* imputation, so the like-for-like column is
`OLS s82`; it has not yet been given SnowModel-imputed inputs.

### The phase indicator

Adding an unpenalised accumulation/melt term helps the shoulders (Jan 27: 70.0 -> 60.3,
May 18: 57.4 -> 53.0) but badly hurts mid-season (Mar 3: 31.5 -> 62.4), and is worse on
average.

The mechanism is boundary ambiguity. Benefit correlates with distance from the melt onset
(+0.53 across 35 band-flight pairs): flights within 30 days of the boundary get **worse** by
21.8 mm on average, flights 60-120 days away get **better** by 22.4 mm.

A binary indicator models two well-separated regimes well and a continuum badly. Earlier
cross-basin work, where ridge-plus-phase performed best, used **one accumulation and one melt
flight per year** -- unambiguous, balanced classes. This run used all 35 flights spanning
January to July, where many sit near an uncertain boundary. The two findings are consistent
rather than contradictory.

WY2026 onset came from the date of peak SnowModel SWE per band -- causal, so usable in real
time. Training-side phase came from `melt_threshold.csv`, which stops at WY2025 and carries
duplicate `(water_year, elev_bin)` rows (100 where 9 x 8 = 72). The two sides therefore use
different onset definitions; acceptable for a test, not for a result.

### Next

Test one accumulation and one melt training flight per year, for both OLS-sensor-82 and
ridge. Less training data, but possibly less noise -- and it is the configuration under which
ridge-plus-phase previously won.

---

## Known defect affecting all arms

`imputation_w_pillows` compares adjusted R² across two different datasets — single donors are
scored on the whole record, pairs on 2013 onward only — so its forward search almost always
runs to three donors regardless of whether the third helps. Measured uplift from the dataset
change alone is +0.147 mean adjusted R² across 750 USCASJ pillow pairs.

This affects the predict-NaNs half of **every arm equally**, so it does not bias the
comparison between them. It is deliberately left unfixed until the arms are complete; fixing
it mid-sequence would mean arm 3 differed from arms 1 and 2 by more than the model.

Full detail and the rest of the backlog: [`pending_fixes.md`](pending_fixes.md).

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
