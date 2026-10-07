# Snow-Ops

Operational basin snow-water-equivalent (SWE) prediction for water
supply forecasting, built around Airborne Snow Observatories (ASO) lidar flights and snow pillows.

---

## What this does

ASO flights measure basin snowpack with high accuracy, but they are expensive and flown only
a handful of times per season — five flights over the San Joaquin in water year 2026. Snow
pillows report daily but at a point, and a pillow is not a basin.

This system bridges that gap. It fits **multiple linear regression models that predict
basin-mean SWE, by elevation band, from the snow pillows that best explain the ASO
record** — then evaluates those models every day against live pillow observations. The
result is a daily basin SWE estimate between flights, anchored to the flights themselves.

Supporting datasets (SnowModel, SNODAS, UA SWE) provide independent comparison and fill
gaps when a pillow drops out.

### Method, briefly

1. **Train** on paired history — daily pillow SWE (WY1980–2025) against ASO basin SWE
   (35 flights, 2017–2025), per elevation band.
2. **Select pillows** by exhaustive search: score every combination of candidate pillows
   taken 1 through 5, keep the best by adjusted R². The top 10 combinations form an
   ensemble, so each prediction carries a spread rather than a single number.
3. **Predict daily** from QA'd live pillow observations, in three temporal regimes
   (full season, accumulation, melt) and under two strategies for filling missing pillows.
4. **Publish** basin SWE in mm and acre-feet, by elevation band.

Prediction quality depends on pillow data quality, so observations pass through a
three-method automated QA stage plus manual review before they reach a model.

---

## Status

| | |
|---|---|
| Operational basin | **USCASJ** (San Joaquin) |
| Configured basins | 12 across California and Colorado |
| Last complete water year | **WY2026** — 2025-10-01 to 2026-09-30 |
| Daily predictions | **paused** following WY2026 close-out |

Restarting is a water-year rollover, not simply re-enabling cron — see
[`docs/wy2027_rollover.md`](docs/wy2027_rollover.md).

---

## Repository layout

| Path | Contents |
|---|---|
| [`pipeline/`](pipeline/) | Ingest, processing and orchestration — the nightly cron chain. **Start here**: [`pipeline/README.md`](pipeline/README.md) |
| `scripts/` | Analysis layer — QA methods, MLR models, plotting, prediction drivers |
| `src/snow_ops/` | Installable package; the pretrained MLR lookup library |
| `config/regions/` | Per-basin configuration: geometry, elevation bands, pillow lists, QA exclusions |
| `docs/` | Decision records and operational plans |
| `data/basins/` | Published per-basin outputs consumed by the web app |
| `pngs/` | Additional figures |
| `public/` | Generated HTML and figures |
| `app.py`, `pages/` | Streamlit application |
| `notebooks/` | Exploratory analysis |

Scripts in `pipeline/` and `scripts/` are symlinked into the operational `bin/` tree and
invoked from there by cron, so the repository is the single source of truth for code that
runs in production.

---

## Documentation

| Document | Covers |
|---|---|
| [`pipeline/README.md`](pipeline/README.md) | Data flow by scale, the nightly chain, QA and MLR in detail, known failure modes |
| [`docs/wy2027_rollover.md`](docs/wy2027_rollover.md) | Restarting after a pause; rolling a completed year into training |
| [`docs/cdec_sensor_selection.md`](docs/cdec_sensor_selection.md) | Which CDEC sensor to ingest, and why — a decision record |
| [`docs/mlr_cv_audit_2026.md`](docs/mlr_cv_audit_2026.md) | Cross-validation audit of pillow selection (branch `mlr-cv-audit`) |

---

## Data sources

| Dataset | Provider | Scale | Role |
|---|---|---|---|
| ASO 50 m SWE | Airborne Snow Observatories | Basin | Training target |
| CDEC snow pillows | CA Dept. of Water Resources | Point | Predictor, daily |
| SNOTEL | NRCS | Point | Predictor, daily |
| HRRR | NOAA | Western US | SnowModel forcing |
| SnowModel | Liston & Elder | Basin | Physical model; gap-filling and QA reference |
| SNODAS | NSIDC | CONUS | Independent comparison |
| UA SWE | University of Arizona | CONUS | Independent comparison |

---

## Running it

The pipeline runs unattended via four chained cron jobs (download → process → QA and
predict → publish), spanning midnight. Operational detail, environment requirements and
the stages that need manual intervention after a pause are in
[`pipeline/README.md`](pipeline/README.md).

The Streamlit app is served from `app.py` and reads only from `data/basins/`.

---

## A note on provenance

Several decisions behind published numbers were, until recently, recorded nowhere —
recoverable only from email or a colleague's memory. `docs/` exists to stop that happening
again. Where a document records a judgement rather than a fact, it says so, and says what
the judgement rested on.
