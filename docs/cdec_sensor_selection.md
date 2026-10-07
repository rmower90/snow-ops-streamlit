# CDEC sensor selection for pillow SWE — decision record

**Status as of 2026-10-06:** decided, not yet implemented.

**Decision:** fetch CDEC sensor **82 (SWE_ADJ) only**. Do **not** backfill with sensor 3
(raw SWE) when 82 is missing. Days 82 does not report stay NaN and are handled downstream
by the existing imputation/QA machinery, not by silently substituting a different sensor.

The authority for this is an email thread, not anything in this repository. This file
exists because the reasoning was otherwise unrecorded.

---

## 1. The two sensors

`scripts/pillow_api.py`:

```python
SENSOR_COLUMNS = {82: "SWE_ADJ", 3: "SWE"}
```

- **82 — SWE_ADJ**: the adjusted/revised series.
- **3 — SWE**: the raw series.

## 2. What was wrong with the legacy behavior

`load_pillow_api()` has defaulted to `sensor_priority=(82, 3)` since the beginning. The
problem is not the preference order — it is that the two series are combined
**element-wise**. From the function's own docstring:

> Every listed sensor is fetched, then combined ELEMENT-WISE -- for each day the value
> comes from the first sensor that reported that day, so one station's series can mix
> sensors over time.

So a single station's record can silently alternate between adjusted and raw values day to
day, with no flag marking the switch. Any discontinuity that introduces is indistinguishable
from a real change in snowpack. That element-wise blending is the defect; "82 with 3 as
fallback" was never a deliberate data-quality choice, just an unexamined default.

## 3. How the investigation went

The path to the decision reversed direction partway through, which is worth recording so
the intermediate position is not mistaken for the conclusion.

**2026-08-05 — hypothesis favored sensor 3.** `scripts/experiments/sensor_ab_download.py`
was written. Its background section states the premise that motivated the whole exercise:

> A CDEC data specialist advised that sensor 3 (raw SWE) is the series that actually
> receives QA/QC, so it -- not 82 -- should be the one we ingest.

Note this is the *starting hypothesis*, not a result. The script was built to test it, and
was deliberately structured as three arms because a straight prod-vs-sensor-3 diff would
confound two separate effects — production was built incrementally (each day fetched ~24h
later) while a bulk refetch pulls values CDEC has since revised:

| Arm | `sensor_priority` | Isolates |
|---|---|---|
| `PROD82` | — (frozen copy of production) | what the pipeline actually consumed |
| `BULK82` | `(82,)` | `BULK82 − PROD82` = revision / latency effect |
| `TEST` | `(3,)` | `TEST − BULK82` = pure sensor effect |

Both fetched arms are pure single-sensor, so any blended policy can be reconstructed
offline without re-hitting CDEC.

**2026-08-18 — redownload landed on 82 only.**
`notebooks/experiments/redownload_historic_pillow_data.ipynb` ran two full FRIANT
redownloads of WY1980–2025, writing to `FRIANT/raw/`:

| File | Written | Arm |
|---|---|---|
| `pillow_wy_1980_2025.nc` | 2026-03-07 | original, default `(82, 3)` |
| `pillow_wy_1980_2025_redownload_82_3.nc` | 2026-08-18 18:47 | blended |
| `pillow_wy_1980_2025_redownload_82.nc` | 2026-08-18 21:13 | **82 only** |

The 82-only download sits under a markdown cell reading `## sensor 82 only` and is the last
download in the notebook. `FRIANT/raw/post_v5_focused_qa.csv` was written one minute later,
so QA was run against that output.

**2026-08-19 — tooling committed.** `55338eb "CDEC sensor A/B investigation: parameterize
sensor selection"` added the `sensor_priority` parameter. It exposed the knob; it did not
change the default.

**2026-10 — confirmed by email.** 82 only, no fallback to 3. This supersedes the August 5
hypothesis.

## 4. What production actually does today

Nothing above has been wired in. The current state does **not** reflect the decision.

- `scripts/pillow_api.py` still defaults to `sensor_priority=(82, 3)`.
- No production script passes `sensor_priority`. A repo-wide grep finds it only in
  `scripts/experiments/sensor_ab_download.py` and the redownload notebook.
- `FRIANT/raw/pillow_wy_1980_2025_redownload_82.nc` was never promoted into `processed/`,
  and nothing reads it. It is a terminal artifact.

Both prediction paths therefore train on blend-derived data:

| Path | Training data | Lineage |
|---|---|---|
| FRIANT historic (`scripts/mlr_prediction_historic.py`, `OBS_QA_FILE`) | `FRIANT/processed/pillow_wy_1980_2025_qa6.nc` | QA ladder on top of the 2026-03-07 blended download |
| USCASJ daily 2026 (`scripts/mlr_prediction.py`) | `USCASJ/processed/pillow_wy_1980_2025_qa1.nc` | the 2026-03-07 blended download, no ladder |

USCASJ inference uses `USCASJ/processed/USCASJ_insitu_obs_daily_wy_2026.nc` (with `raw/`
loaded alongside), which comes through the daily download path — also the blended default.

Two things worth flagging independently of the sensor question:

1. The two paths train on **different QA rungs of the same record** — `qa6` for FRIANT
   historic, `qa1` for USCASJ daily. `USCASJ/processed/` contains only `qa1`; the ladder
   was built for FRIANT alone.
2. Changing the sensor policy invalidates every rung of the FRIANT QA ladder, because the
   manual QA was performed against blended values.

## 5. FRIANT QA ladder, for reference

The model versions in `data/mlr_prediction/FRIANT/README.md` map onto the `qa*` pillow
files in `data/insitu/FRIANT/processed/`:

| Version | Change | Artifact |
|---|---|---|
| v0 | baseline, raw pillow data, step-wise regression | — |
| v1 | first manual QA/QC pass; switch to combination pillow selection | `pillow_wy_1980_2025_qa1.nc` |
| v2 | second manual QA/QC pass (removed 36 flight-date cells in 2017–2025) | `..._qa2.nc` |
| v3 | linear interpolation of gaps ≤ 3 consecutive values | `..._qa3.nc`, `qa3_imputation_log.csv` |
| v4 | force regression y-intercept to 0 — **model change, not data** | no `qa4` file |
| v5 | fill missing zeros (see `pillow_manual_qa_friant.ipynb`) | `..._qa5.nc`, `qa5_zerofill_log.csv` |
| v6 | filter incomplete per-water-year series to reduce pillow-switch jumps | `..._qa6.nc`, `manual_qa_log_v6.csv` |

The v6 manual review covered every water year 1980–2025
(`manual_qa_reviewed_years_v6.csv`), removing whole-season blocks where warranted — e.g.
KSP WY1981, all 366 values.

Caveat recorded at `scripts/mlr_prediction_historic.py` (the `OBS_QA_FILE` comment block):
the imputation tables under `FRIANT/imputation/` were written during the v1 run from qa1
data and reused unconditionally through v2–v6, so the "predict NaNs" half of those runs was
imputed from qa1 while the "drop NaNs" half tracked the newer data. The cache tag is now
derived from the filename so changing the rung re-keys it, but the previously written tables
should be confirmed regenerated.

## 6. To implement the decision

1. Change the default in `scripts/pillow_api.py` to `sensor_priority=(82,)`, so the
   element-wise blend cannot be reached by omission.
2. Decide what the daily download path does when 82 is absent for a station-day — leave
   NaN and let imputation handle it, rather than reintroducing a fallback.
3. Rebuild the historic record from an 82-only download for **both** sites.
   `FRIANT/raw/pillow_wy_1980_2025_redownload_82.nc` already exists and may serve; USCASJ
   has no 82-only equivalent.
4. Re-run the QA ladder against the 82-only record. The existing `qa2`–`qa6` corrections
   were made against blended values and cannot be carried over unexamined.
5. Resolve the qa1-vs-qa6 split between the two prediction paths while the record is being
   rebuilt.
6. Re-run the imputation tables rather than reusing cached ones.

## 7. Provenance of this document

Written 2026-10-06. Sections 1, 2, 4, 5 and the file inventory in 3 are read directly from
the code, the data directories, and git history. The 2026-08-05 hypothesis is quoted from
the `sensor_ab_download.py` docstring. The final decision in section 3 and the header comes
from an email thread outside this repository and is recorded here on that basis. No result
artifacts from the `PROD82`/`BULK82`/`TEST` A/B were found on disk; if the arms were
compared, the comparison was not written down.
