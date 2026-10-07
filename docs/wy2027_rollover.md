# WY2027 rollover / restart

Written 2026-10-07, when daily USCASJ predictions were paused after closing out WY2026.
Nothing here has been done. It is a plan, not a record.

**The short version:** restarting is not "uncomment the crontab." The pipeline is pinned to
WY2026 in code rather than config, so firing it up unchanged in a new water year would
append WY2027 dates to WY2026 files or fail outright. Budget this as a rollover, not a
restart.

---

## 1. State at the pause

Daily predictions stopped after WY2026 was completed and committed (`cc3d6d6`).

| Stage | End date |
|---|---|
| HRRR raw | 2026-10-05 |
| CDEC pillow raw | 2026-10-05 |
| CDEC pillow processed (QA'd) | **2026-09-30** |
| `hrrr_correlated_test_2026.zarr` | **2026-09-30** |
| Predictions (published) | **2026-09-30** |

Oct 1–5 2026 raw data is on disk but unused: those days belong to WY2027. The crontab's
four jobs were left uncommented at the pause — **comment them out** if that has not
happened, or the chain will keep running a WY2026 pipeline against WY2027 dates.

---

## 2. Part A — make the water year a parameter

Every one of these is a literal today. This is the bulk of the work, and it is the same
work that adding new basins needs, so the two are worth doing together.

| File | What is hardcoded |
|---|---|
| `pipeline/domain_process/insitu_qa_domain.sh` | `2026` in all six `sbatch run_prediction.sh` calls |
| `pipeline/domain_process/process_data_domains.sh` | `SM_START_DATE="2025-10-01"` (drives SnowModel `max_iter`) |
| `pipeline/domain_process/insitu_qa_domain.sh` | same `SM_START_DATE` |
| `pipeline/domain_process/backup_pre_exclude.sh` | eight `*_2026` paths |
| `scripts/mlr_prediction.py` | `pillow_wy_1980_2025_qa1.nc`, `hrrr_correlated_train_2017_2025_dowy.zarr`, `hrrr_correlated_test_2026.zarr`, `{site}_insitu_obs_daily_wy_2026.nc` |
| `scripts/insitu_qa.py` | same train/test files, plus `insitu_qa_simple_wy_2026.csv`, `qa_viz_2026/`, `qa_method_diagnostics_2026/` |
| `scripts/snowmodel_basin_process.py` | `snowmodel_swe_fpaths(..., [2026], ...)`, `hrrr_correlated_test_2026.zarr`, `hrrr_correlated_grids_2017_2025_dowy.csv` |
| `pipeline/snodas/basin_process/snodas_basin_process.py` | `mean_swe_snodas_*_wy2026.csv` |
| `pipeline/uaswe/basin_process/uaswe_basin_process.py` | `mean_swe_uaswe_*_wy2026.csv` |
| `scripts/mlr_prediction_historic.py` | `OBS_QA_FILE = 'pillow_wy_1980_2025_qa6.nc'` |
| `configs/regions/USCASJ.yaml` | `aso_years: end: 2025` |

Note the config above is the **runtime** one at `/home/rossamower/work/aso/configs/`, not the
repo copy — those two have drifted and only the runtime one is read. Reconciling them is
still outstanding; see §6.

Suggested shape: thread `(basin, water_year)` through, defaulting the water year to the one
containing "yesterday". That removes the rollover as a recurring task rather than deferring
it a year.

---

## 3. Part B — roll the training data forward

Training is currently WY1980–2025 (insitu) and WY2017–2025 (ASO / SnowModel). WY2026 is now
a completed year and should move from "test" to "train" before WY2027 is predicted.

### B1. Insitu pillows

Current: `insitu/USCASJ/processed/pillow_wy_1980_2025_qa1.nc` — 1979-10-01 .. 2025-10-01.

Append the completed WY2026 record to produce `pillow_wy_1980_2026_qa1.nc`. The WY2026 side
is `processed/USCASJ_insitu_obs_daily_wy_2026.nc` (365 days, QA applied). Same 30 pillows,
same variable names, so it is an `xr.concat` along `time`.

**Seam:** the historic file's last day is 2025-10-01, which is WY2026 day 1. Take the
historic side through **2025-09-30** and let the WY2026 file own the rest. Verify
afterwards that there are no duplicate timestamps and the only day-step present is 1.

Decide which QA rung to carry forward. FRIANT's ladder runs qa1→qa6 and the historic runner
uses qa6, while USCASJ's daily runner trains on qa1 — that asymmetry is unresolved and
worth settling here rather than inheriting it.

### B2. ASO flights

Current: `data/aso/USCASJ/ASO_50M_SWE_TSERIES.nc` and `_SPATIAL.nc` — 35 flight dates,
2017-01-29 .. 2025-05-09.

**The five WY2026 flights are on disk but not ingested.** They exist only as raw GeoTIFFs:

```
data/aso/USCASJ/wy_2026/raw/ASO_50M_SWE_USCASJ_20260127.tif
                            ..._20260303.tif
                            ..._20260328.tif
                            ..._20260429.tif
                            ..._20260518.tif
```

Ingest them into both TSERIES and SPATIAL (35 → 40 dates), then set `aso_years.end: 2026`
in the region config. The published `data/basins/USCASJ/aso/ASO_50M_SWE_TSERIES.nc` also
carries 35 and will need republishing.

The SWE-volume page already reads the raw tifs directly, which is why it reports five ASO
dates for WY2026 while the training file knows nothing about them. Do not take that page as
evidence the flights are ingested.

### B3. SnowModel / HRRR correlated grids

Current:
- `hrrr_correlated_train_2017_2025_dowy.zarr` — 30 pillows, 3288 days
- `hrrr_correlated_train_2017_2025.zarr` — **28** pillows, 3288 days (note the difference)
- `hrrr_correlated_test_2026.zarr` — 30 pillows, 365 days
- `hrrr_correlated_grids_2017_2025[_dowy].csv` — the per-pillow correlated grid lookup

Concatenate the WY2026 test zarr onto the training zarr → `..._train_2017_2026_dowy.zarr`,
3288 + 365 = 3653 days. Check whether the correlated-grid CSVs need regenerating with the
extra year or are intended to stay fixed at the 2017–2025 fit; that choice changes what the
model is conditioned on and should be deliberate.

Resolve the 30-vs-28 pillow discrepancy between the two training zarrs before relying on
either.

### B4. The same step, next year

For WY2028, repeat B1–B3 folding WY2027 in: insitu → `pillow_wy_1980_2027_*`, ASO → 2017–2027,
correlated train → `2017_2027`. If Part A is done properly the paths follow from the water
year and this becomes a single scripted step rather than a hand edit.

---

## 4. Part C — WY2027 test scaffolding

New-year artifacts that must exist before the nightly chain can run:

- `hrrr_correlated_test_2027.zarr` — written by `snowmodel_basin_process.py` at the end of
  the SnowModel job; it will create this once the year is parameterized
- `{site}_insitu_obs_daily_wy_2027.nc` (raw and processed)
- `mean_swe_{snodas,uaswe,snowmodel}_*_wy2027.csv`
- `qa/insitu_qa_{simple,detail}_wy_2027.csv`, `qa/qa_viz_2027/`, `qa/qa_method_diagnostics_2027/`
- `exclude_pillows_ranges` in the region config reset for WY2027 — the 19 WY2026 windows are
  dated 2025-10-01..2026-10-01 and are meaningless in the new year

Also reset the per-stage checklist CSVs, or the todo generators will try to backfill from
WY2026's last date:

```
insitu/USCASJ/USCASJ_insitu_daily_download_checklist.csv
uaswe/USCASJ/USCASJ_uaswe_daily_download_checklist.csv  (+ _provisional_)
snodas/USCASJ/USCASJ_snodas_daily_download_checklist.csv
snowmodel/domains/USCASJ/USCASJ_snowmodel_daily_download_checklist.csv
snodas/CONUS/wus_snodas_daily_download_checklist.csv
uaswe/CONUS/wus_uaswe_daily_download_checklist.csv
hrrr .../wus_hrrr_daily_download_checklist.csv
```

---

## 5. Part D — restart sequence

Most stages catch up on their own from a checklist: CDEC pillows, HRRR raw, SNODAS, UASWE,
the SnowModel basin process, and MLR (which runs a whole water year). **Two do not.**

For every calendar month skipped during the pause, run once each:

```bash
source /opt/miniforge3/etc/profile.d/conda.sh && conda activate sm_nc_process
cd /home/rossamower/bin/hrrr/wus_process/SnowModel_Preproccess/netcdf/sm_wrf_preprocess
python process_vars_one_month.py <YYYY> <MM> <last-day-of-MM>     # one call per month

cd /home/rossamower/bin/hrrr/basin_process
python hrrr_clip_aso_domains.py USCASJ <YYYY> <MM>                 # one call per month
```

`process_vars_one_month.py` writes **one month per run** — the previous-month entry in its
`month_lst` is a lead-in boundary for month-start aggregation, not a second month to
process. This is by design, and it is what silently left September 2026 empty: the nightly
run only ever covers the current month. `hrrr_clip_aso_domains.py` likewise takes one month
per call and the cron driver only ever passes it the current one.

Then uncomment the crontab. The chain spans midnight:

```
22:00 day N    download_daily_conus.sh    HRRR + SNODAS + UASWE
23:00 day N    process_data_domains.sh    CDEC, HRRR clip, SnowModel (sbatch)
02:00 day N+1  insitu_qa_domain.sh        insitu QA + MLR
08:00 day N+1  copy_generate_html.sh      publish
```

The 23:00 → 02:00 gap is what lets the `sbatch`'d SnowModel finish before MLR reads its
output. On the first run after a long pause `max_iter` covers the whole water year to date,
making it the longest SnowModel run of the season against a `--time=02:00:00` SLURM cap.
Check `squeue` that night; if it overruns 02:00, re-run `insitu_qa_domain.sh` by hand once
it lands. Do not let the QA job read `hrrr_correlated_test_*.zarr` while the SnowModel
post-process is writing it with `mode="w"`.

---

## 6. Traps worth knowing

Everything below was hit during the WY2026 close-out.

**Silent success.** The HRRR monthly processing, the SNODAS/UASWE water-year misfiling, and
the SnowModel aggregation crash all exited 0 with nothing useful in the log. Verify by
inspecting output date ranges, never by exit code.

**Water-year boundaries.** Fixed in `760b5d4`, `2733135`, `97a79a0`, but the pattern is
worth re-checking anywhere new: downloaders derived the water year from the *run* date while
basin clips read a hardcoded `wy_2026`. The pair only misbehaves when a run crosses Oct 1 —
precisely a rollover.

**Environment, not code.** Two failures came from the wrong interpreter:
- `conda activate bor` sets `CONDA_PREFIX` but bare `python` may still resolve to
  `~/bin/miniconda_envs/jomey_hrrr/bin/python`, which has no `pyarrow` — this silently broke
  the publish step. `copy_generate_html.sh` still calls bare `python`.
- pyproj reads `PROJ_DATA`, not `PROJ_LIB`, and the base `/opt/miniforge3` ships a `proj.db`
  at layout version 3 where pyproj wants ≥ 4.

Safe invocation for anything in `bor`:

```bash
source /opt/miniforge3/etc/profile.d/conda.sh && conda activate bor
/home/rossamower/work/aso/conda/envs/bor/bin/python <script>      # explicit interpreter
```

**Config drift.** The repo's `config/regions/USCASJ.yaml` and the runtime
`/home/rossamower/work/aso/configs/regions/USCASJ.yaml` differ: only the runtime one carries
`manual_qa_only` and the 19 `exclude_pillows_ranges`, and only the runtime one is read.
Reconcile before relying on the repo copy for anything.

**Sensor policy.** WY2026 was produced on the legacy blended `sensor_priority=(82, 3)`
download. The decision is sensor 82 only — see `docs/cdec_sensor_selection.md`. It is **not
implemented**; `load_pillow_api` still defaults to `(82, 3)`. A rollover is a natural point
to make the switch, but doing so breaks comparability with the WY2026 baseline, so decide
deliberately.

---

## 7. Verification checklist

After a rollover, confirm by date range rather than by job status:

- [ ] `hrrr_correlated_test_<WY>.zarr` spans the full year to date
- [ ] `processed/{site}_insitu_obs_daily_wy_<WY>.nc` matches the zarr's time axis (an exact
      `.sel` is done on it, so the obs file must be a superset — a mismatch raises)
- [ ] published predictions reach yesterday
- [ ] `mean_swe_{snodas,uaswe,snowmodel}_*_wy<WY>.csv` all reach yesterday
- [ ] HRRR processed **and** basin-clipped for every month in the year so far
- [ ] no files in a `wy_<WY+1>` directory that belong to `<WY>`
