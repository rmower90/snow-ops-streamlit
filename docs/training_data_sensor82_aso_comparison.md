# Training data: blended (82,3) vs bulk sensor-82, on ASO flight dates

Written 2026-10-07, before the sensor-82 prediction experiment.

**Question.** The WY2026 models were trained on a pillow record built from the blended CDEC
download (`sensor_priority=(82,3)`) with manual QA applied. We now hold an independent bulk
download of sensor 82 alone. If we switch training data, does the signal the model fits to
actually change?

**Answer: no, not meaningfully.** Of the 736 pillow-date cells where both records carry a
value on an ASO flight date, **6 differ** — five of them by under 3 mm.

---

## Method

Compared on the 35 ASO flight dates only, since those are the dates the MLR actually fits
to. All 35 are present in both records and both carry the same 30 pillows.

| | |
|---|---|
| **Old** (operational training) | `insitu/USCASJ/processed/pillow_wy_1980_2025_qa1.nc` — blended (82,3), manual QA |
| **New** (candidate) | `insitu/FRIANT/raw/pillow_wy_1980_2025_redownload_82.nc` — bulk sensor 82, no QA |
| ASO dates | `aso/USCASJ/ASO_50M_SWE_TSERIES.nc` — 35 flights, 2017-01-29 .. 2025-05-09 |

The bulk-82 file was verified pure sensor-82 structurally rather than by metadata (it
carries no attrs): every finite value in it is present and identical in the blended
redownload, which is the signature of 82 with no sensor-3 backfill.

---

## Result

35 flights x 30 pillows = 1,050 possible cells.

| | Cells |
|---|---|
| Present in old | 753 |
| Present in bulk 82 | 762 (**+9**) |
| Present in both | 736 |
| **Value differs** | **6** (0.8% of overlap) |
| Only old — lost on a switch | 17 |
| Only new — gained | 26 |
| **Total cells changed** | **49** of ~770 |

Note the dominant difference is **presence, not value**. That matters more than it sounds:
pillow availability determines which combinations are eligible in the selection search on a
given flight date.

### The six differing values

```
2018-04-23  BGP   old=  108.0  bulk82=  105.4    +2.5
2021-02-27  DPO   old=  228.6  bulk82=  226.9    +1.7
2021-02-27  WWC   old=  280.9  bulk82=  282.6    -1.7
2024-03-26  STL   old=  635.0  bulk82=  632.1    +2.9
2017-04-30  TUM   old= 1024.1  bulk82= 1022.6    +1.5
2024-06-29  FLV   old=    0.0  bulk82=   66.3   -66.3   <-- the only material one
```

Five are revision-scale noise. **FLV on 2024-06-29** is the exception and is worth a look:
the old record holds exactly 0.0 in late June while bulk-82 reports 66.3 mm. An exact zero
in melt season is either a genuine melt-out or a zero-fill artifact; the two readings imply
different things about that pillow's behaviour.

### Per pillow

Sorted by cells changed. `bias` is mean(old - bulk82) where both are present.

| pillow | n_old | n_new | both | only_old | only_new | n_diff | mean_abs | max_abs | bias |
|---|---|---|---|---|---|---|---|---|---|
| WWC | 27 | 24 | 24 | 3 | 0 | 1 | 0.1 | 1.7 | -0.1 |
| STL | 30 | 28 | 28 | 2 | 0 | 1 | 0.1 | 2.9 | 0.1 |
| BGP | 29 | 29 | 29 | 0 | 0 | 1 | 0.1 | 2.5 | 0.1 |
| DPO | 25 | 25 | 25 | 0 | 0 | 1 | 0.1 | 1.7 | 0.1 |
| FLV | 18 | 18 | 18 | 0 | 0 | 1 | 3.7 | 66.3 | -3.7 |
| TUM | 35 | 35 | 35 | 0 | 0 | 1 | 0 | 1.5 | 0 |
| GEM | 18 | 11 | 11 | 7 | 0 | 0 | 0 | 0 | -0 |
| TMR | 35 | 31 | 31 | 4 | 0 | 0 | 0 | 0 | 0 |
| MHP | 27 | 31 | 26 | 1 | 5 | 0 | 0 | 0 | 0 |
| KUB | 14 | 20 | 14 | 0 | 6 | 0 | 0 | 0 | 0 |
| BSH | 14 | 19 | 14 | 0 | 5 | 0 | 0 | 0 | 0 |
| AGP | 3 | 5 | 3 | 0 | 2 | 0 | 0 | 0 | 0 |
| TNY | 32 | 34 | 32 | 0 | 2 | 0 | 0 | 0 | 0 |
| UBC | 27 | 29 | 27 | 0 | 2 | 0 | 0 | 0 | 0 |
| BCB | 19 | 20 | 19 | 0 | 1 | 0 | 0 | 0 | -0 |
| DAN | 34 | 35 | 34 | 0 | 1 | 0 | 0 | 0 | 0 |
| KUP | 21 | 22 | 21 | 0 | 1 | 0 | 0 | 0 | 0 |
| SWM | 34 | 35 | 34 | 0 | 1 | 0 | 0 | 0 | 0 |
| CHM | 31 | 31 | 31 | 0 | 0 | 0 | 0 | 0 | 0 |
| GRM | 32 | 32 | 32 | 0 | 0 | 0 | 0 | 0 | -0 |
| GRV | 26 | 26 | 26 | 0 | 0 | 0 | 0 | 0 | 0 |
| HNT | 31 | 31 | 31 | 0 | 0 | 0 | 0 | 0 | 0 |
| KSP | 35 | 35 | 35 | 0 | 0 | 0 | 0 | 0 | 0 |
| LLE | 35 | 35 | 35 | 0 | 0 | 0 | 0 | 0 | 0 |
| MAM | 0 | 0 | 0 | 0 | 0 | 0 |  |  |  |
| PSR | 35 | 35 | 35 | 0 | 0 | 0 | 0 | 0 | 0 |
| SLK | 35 | 35 | 35 | 0 | 0 | 0 | 0 | 0 | 0 |
| SNF | 0 | 0 | 0 | 0 | 0 | 0 |  |  |  |
| STR | 29 | 29 | 29 | 0 | 0 | 0 | 0 | 0 | 0 |
| VLC | 22 | 22 | 22 | 0 | 0 | 0 | 0 | 0 | -0 |
Losses concentrate in **GEM (-7 of 18 flight dates), TMR (-4), WWC (-3), STL (-2)**; gains
in **KUB (+6), BSH (+5), MHP (+5)**. GEM is the one to watch — it loses nearly 40% of its
flight-date coverage.

### Per ASO date

Dates with any change; the other 9 flights are identical in both records.

| date | n_old | n_new | delta | only_old | only_new | n_diff |
|---|---|---|---|---|---|---|
| 2017-04-30 | 21 | 21 | 0 | 0 | 0 | 1 |
| 2018-04-23 | 22 | 22 | 0 | 2 | 2 | 1 |
| 2018-06-01 | 21 | 22 | 1 | 2 | 3 | 0 |
| 2019-06-09 | 22 | 20 | -2 | 2 | 0 | 0 |
| 2019-07-04 | 22 | 20 | -2 | 2 | 0 | 0 |
| 2019-07-14 | 22 | 20 | -2 | 2 | 0 | 0 |
| 2020-05-04 | 24 | 23 | -1 | 1 | 0 | 0 |
| 2020-05-23 | 24 | 23 | -1 | 1 | 0 | 0 |
| 2020-06-08 | 24 | 23 | -1 | 1 | 0 | 0 |
| 2021-02-27 | 23 | 23 | 0 | 0 | 0 | 2 |
| 2021-04-01 | 21 | 22 | 1 | 0 | 1 | 0 |
| 2021-05-03 | 21 | 22 | 1 | 0 | 1 | 0 |
| 2022-02-07 | 23 | 24 | 1 | 0 | 1 | 0 |
| 2022-04-17 | 22 | 22 | 0 | 1 | 1 | 0 |
| 2022-04-30 | 22 | 22 | 0 | 1 | 1 | 0 |
| 2023-05-24 | 18 | 20 | 2 | 0 | 2 | 0 |
| 2023-06-25 | 17 | 20 | 3 | 0 | 3 | 0 |
| 2024-01-29 | 21 | 21 | 0 | 1 | 1 | 0 |
| 2024-02-25 | 22 | 23 | 1 | 0 | 1 | 0 |
| 2024-03-26 | 21 | 22 | 1 | 0 | 1 | 1 |
| 2024-04-30 | 21 | 22 | 1 | 0 | 1 | 0 |
| 2024-05-21 | 21 | 22 | 1 | 0 | 1 | 0 |
| 2024-06-29 | 20 | 20 | 0 | 1 | 1 | 1 |
| 2025-02-26 | 20 | 23 | 3 | 0 | 3 | 0 |
| 2025-03-25 | 20 | 21 | 1 | 0 | 1 | 0 |
| 2025-04-28 | 21 | 22 | 1 | 0 | 1 | 0 |
Availability never moves more than +/-3 pillows on any single flight.

---

## Implication

**Keep the existing training data.** The fit sees the same signal either way, so switching
buys nothing and costs the manual QA effort already invested.

This also removes a confound. The two records differ in *two* ways — sensor policy and QA
level (the old one has had manual passes, the bulk download has none). Had they disagreed,
we could not have said which cause was responsible without further work. They agree, so the
question does not arise.

It follows that a sensor-82 prediction experiment is really a test of the **test** data, not
the training data. That is the cleaner question anyway.

---

## Caveats

**This holds on ASO dates, not across the record.** Over all ~16,800 days the bulk-82 file
carries **11,536 fewer values (-3.5%)** than the operational training file. The fit is
unaffected because it only sees flight dates — but **imputation donors are drawn from the
broader record**, and the three-donor search draws on 2013 onward. Switching training data
would therefore change imputation behaviour even though it would not change the fit. A
further argument for leaving it alone.

**Equal values are not equal provenance.** Agreement on ASO dates does not mean the manual
QA was unnecessary; it means the QA and the sensor switch do not conflict where the model
fits. A full-season comparison for selected pillows would say more about which record is
better overall, and has not been done.

**Source tables:** `data/insitu/USCASJ/sensor82/aso_date_training_comparison_{by_pillow,by_date}.csv`
