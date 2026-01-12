import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt

import requests
import io
from urllib.parse import urlencode

from metloom.pointdata import CDECPointData
from metloom.variables import CdecStationVariables
from metloom_patches import apply_cdec_patches

from shapely.geometry import Point,Polygon
import geopandas as gpd
import os
import time

import ulmo
import copy



def load_pillow_api(buff_geog, start_date, end_date, plot=True, daily_agg="mean", debug=False):
    """
    Download SWE (prefers SWE_ADJ=82, fallback SWE=3) for stations in buff_geog.
    CSV-first, then JSON; robust datetime parsing; Series-safe selection.

    Returns:
      output: list[xr.DataArray] per station (daily, inches→mm, NaNs kept)
      cdec_locations: GeoDataFrame with station metadata (+ 'kind')
      delete_cdec_lst: list of skipped stations
      station_kind: dict {station_id: 'pillow' | 'snow_course' | 'unknown'}
    """

    # ---- dates
    start_date = pd.to_datetime(start_date, errors="coerce")
    end_date   = pd.to_datetime(end_date,   errors="coerce")
    if pd.isna(start_date) or pd.isna(end_date) or end_date < start_date:
        raise ValueError(f"Bad date range: start={start_date}, end={end_date}")
    full_idx = pd.date_range(start_date, end_date, freq="D")

    # ---- patches / discovery
    apply_cdec_patches()
    points = CDECPointData.points_from_geometry(
        buff_geog, [CdecStationVariables.SWE], snow_courses=False
    )
    cdec_locations = points.to_dataframe().sort_values("id")
    try:
        cdec_locations = cdec_locations.set_crs("EPSG:4326", allow_override=True)
    except Exception:
        pass

    # ========================== helpers ==========================
    def _clean_values(s):
        return pd.to_numeric(s, errors="coerce").replace({-9999: np.nan, -999: np.nan, -99: np.nan})

    def _series_daily_from_df(df, value_col, time_index, how="mean"):
        """Aggregate to daily Series on full_idx."""
        if df is None or len(df) == 0:
            return pd.Series(index=full_idx, dtype=float)
        idx = pd.to_datetime(time_index, errors="coerce")
        df2 = pd.DataFrame({"val": _clean_values(df[value_col])}, index=idx)
        df2 = df2[~df2.index.isna()].dropna(subset=["val"])
        if df2.empty:
            return pd.Series(index=full_idx, dtype=float)
        day = df2.index.floor("D")
        out = (df2["val"].groupby(day).first() if how == "first" else df2["val"].groupby(day).mean())
        return out.reindex(full_idx)

    def _series_usable(s):
        return (s is not None) and isinstance(s, pd.Series) and (s.notna().sum() > 0)

    def first_nonempty_series(series_list):
        for s in series_list:
            if _series_usable(s):
                return s
        return None

    # -------- raw CSV (EXACTLY like your probe) --------
    def fetch_csv_raw(station, sensor, start, end, dur="D"):
        if requests is None:
            return None
        base = "https://cdec.water.ca.gov/dynamicapp/req/CSVDataServlet"
        params = {
            "Stations": str(station).upper(),
            "SensorNums": str(sensor),     # 82=SWE_ADJ, 3=SWE
            "dur_code": dur,               # "D" or "H"
            "Start": pd.to_datetime(start).strftime("%Y-%m-%d"),
            "End":   pd.to_datetime(end).strftime("%Y-%m-%d"),
        }
        url = f"{base}?{urlencode(params)}"
        r = requests.get(url, timeout=30)
        try:
            r.raise_for_status()
        except Exception:
            return None
        # pandas can sniff CDEC CSV (your probe showed this works)
        try:
            # pandas.compat is not always present; try it first to match your probe
            df = pd.read_csv(pd.compat.StringIO(r.text))
        except Exception:
            df = pd.read_csv(io.StringIO(r.text))
        return df

    
    def parse_cdec_datetime_from_csv(df):
        """
        Parse CDEC CSV timestamps into a pandas Series of datetimes.

        Handles:
        - 'OBS DATE' in either 'YYYYMMDD HHMM' or 'YYYYMMDDHHMM'
        - separate 'DATE' + 'TIME' columns (any punctuation; ints/strings)
        - 'DATE' only (YYYYMMDD)
        """

        # keep case, just trim whitespace in headers
        df = df.rename(columns={c: str(c).strip() for c in df.columns})

        # 1) Prefer 'OBS DATE'
        if "OBS DATE" in df.columns:
            s = df["OBS DATE"].astype(str).str.strip()

            # try spaced form first (fast path)
            t = pd.to_datetime(s, format="%Y%m%d %H%M", errors="coerce")

            # where parse failed, normalize to digits then compact form
            need = t.isna()
            if need.any():
                s_digits = s[need].str.replace(r"\D", "", regex=True)
                # if only YYYYMMDD present, append "0000"; then trim/pad to 12 chars
                s_digits = s_digits.where(s_digits.str.len() != 8, s_digits + "0000")
                s_digits = s_digits.str.slice(0, 12).str.pad(12, side="right", fillchar="0")
                t2 = pd.to_datetime(s_digits, format="%Y%m%d%H%M", errors="coerce")
                t = t.where(~need, t2)

            if t.notna().any():
                return t

        # 2) DATE + TIME (separate columns)
        if ("DATE" in df.columns) and ("TIME" in df.columns):
            ds = df["DATE"].astype(str).str.replace(r"\D", "", regex=True).str.zfill(8)
            ts = df["TIME"].astype(str).str.replace(r"\D", "", regex=True).str.zfill(4)
            t  = pd.to_datetime(ds + ts, format="%Y%m%d%H%M", errors="coerce")
            if t.notna().any():
                return t

        # 3) DATE only
        if "DATE" in df.columns:
            ds = df["DATE"].astype(str).str.replace(r"\D", "", regex=True).str.zfill(8)
            t  = pd.to_datetime(ds, format="%Y%m%d", errors="coerce")
            if t.notna().any():
                return t

        # 4) last resort: any column with date/time-ish strings
        for c in df.columns:
            if "date" in str(c).lower() or "time" in str(c).lower():
                t = pd.to_datetime(df[c], errors="coerce")
                if t.notna().any():
                    return t

        # nothing parseable → return NaT series of matching length
        return pd.Series(pd.NaT, index=np.arange(len(df)))

    def fetch_csv_series(station_id, sensor_num, dur_code="D"):
        """
        Fetch CDEC CSV for (station, sensor) and return a daily Series on full_idx.
        Mirrors your working probe for reading, then uses the hardened datetime parser.
        """

        base = "https://cdec.water.ca.gov/dynamicapp/req/CSVDataServlet"
        params = {
        "Stations": str(station_id).upper(),
        "SensorNums": str(sensor_num),   # 82=SWE_ADJ, 3=SWE
        "dur_code": dur_code,            # "D" or "H"
        "Start": start_date.strftime("%Y-%m-%d"),
        "End":   end_date.strftime("%Y-%m-%d"),
        }
        url = f"{base}?{urlencode(params)}"

        try:
            r = requests.get(url, timeout=30)
            r.raise_for_status()
        except Exception:
            return None

        # EXACTLY like your probe: let pandas sniff the full CSV
        try:
            df = pd.read_csv(pd.compat.StringIO(r.text))
        except Exception:
            df = pd.read_csv(io.StringIO(r.text))

        if df is None or df.empty:
            return None

        # normalize headers (trim only)
        df = df.rename(columns={c: str(c).strip() for c in df.columns})

        # choose value column
        val_col = next((c for c in ("VALUE", "Value", "value", "VAL", "Data", "DATA") if c in df.columns), None)
        if val_col is None:
            num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
            val_col = num_cols[-1] if num_cols else None
        if val_col is None:
            return None

        # parse datetime
        t = parse_cdec_datetime_from_csv(df)
        if t.isna().all():
            return None

        # clean values (convert '---' etc. to NaN)
        vals = pd.to_numeric(df[val_col], errors="coerce").replace({-9999: np.nan, -999: np.nan, -99: np.nan})

        # daily aggregate on caller's full_idx
        day = t.dt.floor("D")
        grp = pd.DataFrame({"val": vals, "day": day}).dropna(subset=["val", "day"])
        if grp.empty:
            return None
        agg = (grp.groupby("day")["val"].first() if daily_agg == "first" else grp.groupby("day")["val"].mean())
        ser = agg.reindex(full_idx)

        # DEBUG: count how many days survived
        if debug:
            print(f"[DEBUG] {station_id} s{sensor_num} {dur_code}: raw_rows={len(df)}, "
                f"parsed_ts={t.notna().sum()}, numeric_vals={pd.to_numeric(df[val_col], errors='coerce').notna().sum()}, "
                f"daily_nonnull={ser.notna().sum()}")

        return ser if ser.notna().any() else None


    # -------- JSON → daily Series (kept as a fallback) --------
    def fetch_json_series(station_id, sensor_num, dur_code="D"):
        if requests is None:
            return None
        base = "https://cdec.water.ca.gov/dynamicapp/req/JSONDataServlet"
        params = {
            "Stations": str(station_id).upper(),
            "SensorNums": str(sensor_num),
            "dur_code": dur_code,
            "Start": start_date.strftime("%Y-%m-%d"),
            "End":   end_date.strftime("%Y-%m-%d"),
        }
        url = f"{base}?{urlencode(params)}"
        headers = {"User-Agent": "Mozilla/5.0 (compatible; cdec-json/1.0)"}
        try:
            resp = requests.get(url, headers=headers, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            if not isinstance(data, list) or len(data) == 0:
                return None
            df = pd.DataFrame(data)
        except Exception:
            return None

        # datetime + value columns
        t = None
        for dcol in ("datetime","date","obsDate","ObsDate","Date",
                     "MeasurementDate","measurement_date","time","Time","timestamp"):
            if dcol in df.columns:
                t = pd.to_datetime(df[dcol], errors="coerce")
                break
        if t is None:
            return None
        vcol = next((c for c in ("value","Value","val","Val","data","Data") if c in df.columns), None)
        if vcol is None:
            return None
        return _series_daily_from_df(pd.DataFrame({vcol: df[vcol]}), value_col=vcol, time_index=t, how=daily_agg)

    def classify_station(p_obj, series_df):
        """Metadata-first; else heuristic by coverage across months."""
        kind = None
        try:
            meta = p_obj._get_all_metadata()
            sensors = meta.get("sensors", [])
            s = pd.DataFrame(sensors) if not isinstance(sensors, pd.DataFrame) else sensors.copy()
            if not s.empty:
                ren = {}
                for c in s.columns:
                    cl = str(c).lower()
                    if "sensor" in cl and ("num" in cl or "number" in cl): ren[c] = "Sensor Number"
                    if cl.startswith("dur"): ren[c] = "Duration"
                if ren: s = s.rename(columns=ren)
                if ("Sensor Number" in s.columns) and ("Duration" in s.columns):
                    snow_nums = {3,82,18}
                    rows = s.loc[
                        s["Sensor Number"].astype(str).str.extract(r"(\d+)")[0].astype(float).astype(int).isin(snow_nums)
                    ]
                    durs = set(str(x).strip().lower() for x in rows["Duration"].dropna().tolist())
                    if durs:
                        if durs.issubset({"monthly"}): kind = "snow_course"
                        elif durs & {"daily","hourly"}: kind = "pillow"
        except Exception:
            pass
        if kind is None:
            series = series_df["SWE_clean"]
            coverage = series.notna().sum() / max(1, len(series))
            months_with_values = series.groupby(pd.Index(series.index).to_period("M")).apply(lambda s: s.notna().any()).sum()
            if coverage >= 0.15:       kind = "pillow"
            elif months_with_values >= 2: kind = "snow_course"
            else:                        kind = "unknown"
        return kind

    # ============================ main ============================
    output = []
    delete_cdec_lst = []
    station_kind = {}
    skipped = []

    for station in cdec_locations.id.astype(str).values:
        p = CDECPointData(station, station)

        # CSV first (D→H), then JSON (D→H). Pick the first non-empty series.
        s82 = first_nonempty_series([
            fetch_csv_series(station, 82, "D"),
            fetch_csv_series(station, 82, "H"),
            fetch_json_series(station, 82, "D"),
            fetch_json_series(station, 82, "H"),
        ])
        s3 = first_nonempty_series([
            fetch_csv_series(station, 3, "D"),
            fetch_csv_series(station, 3, "H"),
            fetch_json_series(station, 3, "D"),
            fetch_json_series(station, 3, "H"),
        ])

        if not _series_usable(s82) and not _series_usable(s3):
            skipped.append(station)
            print(f"{station}: skipped (no rows from CSV/JSON for sensors 82/3)")
            if debug:
                # one-shot debug: show CSV head for 82 daily to see columns quickly
                probe_df = fetch_csv_raw(station, 82, start_date, end_date, "D")
                if isinstance(probe_df, pd.DataFrame):
                    print(f"[DEBUG] {station} CSV head (82/D):\n{probe_df.head(3)}")
            continue

        # wide daily (no pivot)
        wide = pd.DataFrame(index=full_idx)
        wide["SWE_ADJ"] = s82.reindex(full_idx) if _series_usable(s82) else np.nan
        wide["SWE"]     = s3.reindex(full_idx)  if _series_usable(s3)  else np.nan
        wide["SWE_clean"] = wide["SWE_ADJ"].combine_first(wide["SWE"])

        non_null = int(wide["SWE_clean"].notna().sum())
        print(f"{station}: {non_null} day(s) with SWE (after clean) | ", end="")

        # classify + package (inches→mm)
        kind = classify_station(p, wide)
        station_kind[station] = kind
        da = xr.DataArray(
            (wide["SWE_clean"].to_numpy() * 25.4),
            dims=("time",),
            coords={"time": full_idx},
            name=station
        )
        output.append(da)

        if plot:
            plt.plot(da.time, da, label=f"{station} ({kind})")

    if skipped:
        print(f"\nSkipped {len(skipped)} station(s): {', '.join(skipped[:50])}" + ("..." if len(skipped) > 50 else ""))

    if plot and len(output) > 0:
        plt.legend(ncol=2, fontsize=8)
        plt.xlabel("Date"); plt.ylabel("Snow Water Equivalent [mm]")
        plt.ylim(0, 5000); plt.tight_layout()
        plt.savefig("pillow_plots_api.png", dpi=200)

    cdec_locations = cdec_locations.copy()
    cdec_locations["kind"] = cdec_locations["id"].astype(str).map(station_kind)

    delete_cdec_lst = skipped
    return output, cdec_locations, delete_cdec_lst, station_kind

# =========================
# Snow-course helpers (global)
# =========================

# ---------- robust numeric & time helpers (keep these) ----------
def _sc_to_numeric_series(s):
    import re, numpy as np, pandas as pd
    if not isinstance(s, pd.Series):
        s = pd.Series(s)
    v = pd.to_numeric(s, errors="coerce")
    if v.notna().sum() < max(3, int(0.1 * len(v))):
        ex = s.astype(str).str.extract(r'([-+]?\d*\.?\d+)', expand=False).replace({"": np.nan})
        v = pd.to_numeric(ex, errors="coerce")
    return v.replace({-9999: np.nan, -999: np.nan, -99: np.nan})

def _sc_parse_datetime_from_csv(df):
    import pandas as pd, numpy as np
    df = df.rename(columns={c: str(c).strip() for c in df.columns})
    if "OBS DATE" in df.columns:
        s = df["OBS DATE"].astype(str).str.strip()
        t = pd.to_datetime(s, format="%Y%m%d %H%M", errors="coerce")
        need = t.isna()
        if need.any():
            sd = s[need].str.replace(r"\D", "", regex=True)
            sd = sd.where(sd.str.len() != 8, sd + "0000").str.slice(0, 12).str.pad(12, side="right", fillchar="0")
            t2 = pd.to_datetime(sd, format="%Y%m%d%H%M", errors="coerce")
            t = t.where(~need, t2)
        if t.notna().any(): return t
    if ("DATE" in df.columns) and ("TIME" in df.columns):
        ds = df["DATE"].astype(str).str.replace(r"\D", "", regex=True).str.zfill(8)
        ts = df["TIME"].astype(str).str.replace(r"\D", "", regex=True).str.zfill(4)
        t  = pd.to_datetime(ds + ts, format="%Y%m%d%H%M", errors="coerce")
        if t.notna().any(): return t
    if "DATE" in df.columns:
        ds = df["DATE"].astype(str).str.replace(r"\D", "", regex=True).str.zfill(8)
        t  = pd.to_datetime(ds, format="%Y%m%d", errors="coerce")
        if t.notna().any(): return t
    for c in df.columns:
        if "date" in str(c).lower() or "time" in str(c).lower():
            t = pd.to_datetime(df[c], errors="coerce")
            if t.notna().any(): return t
    import pandas as pd
    return pd.Series(pd.NaT, index=range(len(df)))

def _sc_find_swe_column_and_values(df):
    """
    Choose the SWE/WC column AND return cleaned numeric Series for it.
    Skips obvious non-value columns (date, time, depth, density, unnamed, etc.).
    Recognizes 'W.C.' as Water Content. Returns (column_name, numeric_series) or (None, None).
    """
    import re, pandas as pd

    def norm(name: str) -> str:
        return re.sub(r'[^a-z0-9]+', ' ', str(name).lower()).strip()

    # columns we should never treat as values
    BLOCK_TOKENS = {"date", "measured", "time", "depth", "density",
                    "station", "id", "elev", "elevation", "lat", "lon", "long", "location", "site", "name"}
    BLOCK_PREFIX = ("unnamed",)

    best_name, best_vals, best_score, best_nonnull = None, None, -1, -1

    for col in df.columns:
        disp = str(col).strip()
        n = norm(disp)

        # hard blocks
        if any(tok in n.split() for tok in BLOCK_TOKENS):   # e.g. 'measured date', 'density'
            continue
        if any(n.startswith(pfx) for pfx in BLOCK_PREFIX):  # 'Unnamed: 5', etc.
            continue

        # 'W.C.' is a common "Water Content" label
        is_wc = (n.replace(" ", "") == "wc") or (n in {"w c", "w", "w c."})

        # how numeric is it?
        vals = _sc_to_numeric_series(df[col])
        n_nonnull = int(vals.notna().sum())
        if n_nonnull == 0:
            continue

        # scoring
        if is_wc:
            score = 95
        elif ("snow" in n and "water" in n and ("equiv" in n or "equivalent" in n)) or re.search(r"\bswe\b", n):
            score = 100
        elif ("water" in n and "content" in n):
            score = 90
        elif re.search(r"\bvalue\b", n):
            score = 10
        else:
            score = 1  # generic numeric fallback

        if (score > best_score) or (score == best_score and n_nonnull > best_nonnull):
            best_name, best_vals, best_score, best_nonnull = disp, vals, score, n_nonnull

    return (best_name, best_vals) if best_name is not None else (None, None)


def _sc_series_usable(s):
    import pandas as pd
    return (s is not None) and isinstance(s, pd.Series) and s.notna().any()

def _sc_first_nonempty(series_list):
    for s in series_list:
        if _sc_series_usable(s): return s
    return None

def _sc_monthly_candidate_sensors(station_id):
    """Use CDEC metadata to find monthly SWE-like sensors (don’t rely only on 82/3)."""
    import pandas as pd
    try:
        p = CDECPointData(station_id, station_id)
        meta = p._get_all_metadata()
    except Exception:
        meta = {}
    sensors = meta.get("sensors", [])
    s = pd.DataFrame(sensors) if not isinstance(sensors, pd.DataFrame) else sensors.copy()
    if s is None or s.empty: return []
    ren = {}
    for c in s.columns:
        cl = str(c).lower()
        if "sensor" in cl and ("num" in cl or "number" in cl): ren[c] = "Sensor Number"
        if cl.startswith("dur"): ren[c] = "Duration"
        if "type" in cl: ren[c] = "Sensor Type"
    if ren: s = s.rename(columns=ren)
    if "Sensor Number" not in s.columns or "Duration" not in s.columns: return []
    s["Sensor Number"] = s["Sensor Number"].astype(str).str.extract(r"(\d+)")[0]
    s = s[s["Duration"].astype(str).str.lower().str.contains("month", na=False)]
    if s.empty: return []
    if "Sensor Type" in s.columns:
        swe_like = s["Sensor Type"].astype(str).str.contains(r"SNO|SWE|WATER", case=True, regex=True, na=False)
        s = s.loc[swe_like] if swe_like.any() else s
    nums = list(dict.fromkeys(s["Sensor Number"].tolist()))
    # Try the common ones first; include even if missing from metadata.
    for k in ["82", "3", "76"]:
        if k in nums: nums.remove(k)
        nums.insert(0, k)
    out, seen = [], set()
    for x in nums:
        if x not in seen:
            out.append(x); seen.add(x)
    return out[:8]

# ---------- NEW: snowQuery CSV fallback ----------
def _sc_fetch_snowquery_monthly(station_id, start_dt, end_dt, monthly_agg="first", debug=False):
    """
    Pull monthly snow course readings from CDEC snowQuery.
    Returns {'values': Series, 'times': Series, 'vcol': <name>} or None.
    Handles both wide tables (Water Content / SWE column) and long tables
    (Parameter + Value).
    """
    import io, re
    import pandas as pd

    try:
        import requests
    except Exception:
        return None

    start_dt = pd.to_datetime(start_dt)
    end_dt   = pd.to_datetime(end_dt)

    sd_mdy = start_dt.strftime("%m/%d/%Y")
    ed_mdy = end_dt.strftime("%m/%d/%Y")
    sd_ymd = start_dt.strftime("%Y-%m-%d")
    ed_ymd = end_dt.strftime("%Y-%m-%d")

    base = "https://cdec.water.ca.gov/dynamicapp/snowQuery"
    headers = {"User-Agent": "Mozilla/5.0 (compatible; cdec-snowquery/1.0)"}

    # ---------- 1) Try CSV (several param variants) ----------
    csv_param_sets = [
        {"o": "csv", "s": str(station_id).upper()},
        {"o": "csv", "s": str(station_id).upper(), "d": "1", "span": "All"},
        {"o": "csv", "s": str(station_id).upper(), "sd": sd_mdy, "ed": ed_mdy},
        {"o": "csv", "s": str(station_id).upper(), "Start": sd_ymd, "End": ed_ymd},
        # some backends accept "span=Water Year"
        {"o": "csv", "s": str(station_id).upper(), "d": "1", "span": "Water Year"},
    ]

    df_csv = None
    for params in csv_param_sets:
        try:
            r = requests.get(base, params=params, headers=headers, timeout=30)
            r.raise_for_status()
            txt = (r.text or "").strip()
            looks_html = txt[:256].lstrip().lower().startswith("<")
            if looks_html or not txt or "," not in txt:
                continue
            df = pd.read_csv(io.StringIO(txt))
            if df is not None and not df.empty:
                df_csv = df
                break
        except Exception:
            continue

    def _pick_date_column(df):
        # Prefer 'Date' over 'Measured Date' if both exist
        cols = {str(c).strip(): c for c in df.columns}
        for key in ("Date", "Observation Date", "Obs Date"):
            if key in cols:
                return cols[key]
        # fallback: any column containing 'date'
        for c in df.columns:
            if "date" in str(c).lower():
                return c
        return None


    def _norm(s: str) -> str:
        return re.sub(r'[^a-z0-9]+', ' ', str(s).lower()).strip()

    if df_csv is not None and not df_csv.empty:
        date_col = _pick_date_column(df_csv)
        if date_col is None:
            if debug:
                print(f"[SC snowQuery CSV] {station_id}: no date column; cols={list(df_csv.columns)}")
        else:
            # Wide-table attempt
            vcol, vals = _sc_find_swe_column_and_values(df_csv)
            if vcol is None:
                # Long-table attempt: look for Parameter + Value
                cols = [c.lower() for c in df_csv.columns]
                if "parameter" in cols and "value" in cols:
                    pcol = df_csv.columns[cols.index("parameter")]
                    vcol2 = df_csv.columns[cols.index("value")]
                    # filter for SWE-like parameters
                    sel = df_csv[pcol].astype(str).str.lower().str.contains(
                        r"swe|water\s*content|water\s*equiv", regex=True
                    )
                    sub = df_csv.loc[sel, [date_col, vcol2]].copy()
                    sub[vcol2] = _sc_to_numeric_series(sub[vcol2])
                    t = pd.to_datetime(sub[date_col], errors="coerce")
                    m = t.notna() & sub[vcol2].notna() & (t >= start_dt) & (t <= end_dt)
                    if m.any():
                        return {"values": sub.loc[m, vcol2], "times": t[m], "vcol": vcol2}
                if debug:
                    print(f"[SC snowQuery CSV] {station_id}: no SWE-like column; cols={list(df_csv.columns)}")
            else:
                t = pd.to_datetime(df_csv[date_col], errors="coerce")
                m = t.notna() & (t >= start_dt) & (t <= end_dt)
                if m.any():
                    return {"values": _sc_to_numeric_series(df_csv.loc[m, vcol]),
                            "times": t[m], "vcol": vcol}

    # ---------- 2) HTML fallback ----------
    # Try a couple of HTML variants; site renders tables server-side
    html_param_sets = [
        {"o": "html", "s": str(station_id).upper(), "d": "1", "span": "All"},
        {"s": str(station_id).upper(), "d": "1", "span": "All"},
        {"s": str(station_id).upper()},  # default
    ]

    for params in html_param_sets:
        try:
            r = requests.get(base, params=params, headers=headers, timeout=30)
            r.raise_for_status()
            tables = pd.read_html(r.text)  # needs lxml/html5lib
        except Exception as e:
            if debug:
                print(f"[SC snowQuery HTML] {station_id}: fetch/parse failed ({type(e).__name__})")
            continue

        if not tables:
            continue

        if debug:
            # Print columns for visibility
            for i, tdf in enumerate(tables[:4]):
                print(f"[SC snowQuery HTML] {station_id}: table {i} cols={list(tdf.columns)}")

        best = None; best_rows = -1; best_info = None

        for df in tables:
            if df is None or df.empty:
                continue
            df = df.rename(columns=lambda x: str(x).strip())
            date_col = _pick_date_column(df)
            if date_col is None:
                continue

            # Wide-style first
            vcol, vals = _sc_find_swe_column_and_values(df)
            if vcol is not None:
                t = pd.to_datetime(df[date_col], errors="coerce")
                vals = _sc_to_numeric_series(df[vcol])
                m = t.notna() & vals.notna() & (t >= start_dt) & (t <= end_dt)
                n = int(m.sum())
                if n > best_rows:
                    best = (vals[m], t[m]); best_rows = n; best_info = (date_col, vcol)
                continue

            # Long-style: Parameter + Value
            cols_l = [c.lower() for c in df.columns]
            if "parameter" in cols_l and "value" in cols_l:
                pcol = df.columns[cols_l.index("parameter")]
                vcol2 = df.columns[cols_l.index("value")]
                sel = df[pcol].astype(str).str.lower().str.contains(
                    r"swe|water\s*content|water\s*equiv", regex=True
                )
                sub = df.loc[sel, [date_col, vcol2]].copy()
                sub[vcol2] = _sc_to_numeric_series(sub[vcol2])
                t = pd.to_datetime(sub[date_col], errors="coerce")
                m = t.notna() & sub[vcol2].notna() & (t >= start_dt) & (t <= end_dt)
                n = int(m.sum())
                if n > best_rows:
                    best = (sub.loc[m, vcol2], t[m]); best_rows = n; best_info = (date_col, vcol2)

        if best is not None and best_rows > 0:
            vals, times = best
            if debug:
                dc, vc = best_info
                print(f"[SC snowQuery HTML] {station_id}: picked Date='{dc}', Value='{vc}', rows={best_rows}")
            return {"values": vals, "times": times, "vcol": best_info[1]}

    if debug:
        print(f"[SC snowQuery] {station_id}: no usable table (CSV + HTML both failed)")
    return None


# =========================
# Single-station snow-course (monthly)
# =========================

def load_snow_course_single(station_id, start_date, end_date,
                            monthly_agg="first", plot=True, debug=False):
    import io, numpy as np, pandas as pd, xarray as xr, matplotlib.pyplot as plt
    from urllib.parse import urlencode
    try:
        import requests
    except Exception:
        requests = None

    start_date = pd.to_datetime(start_date); end_date = pd.to_datetime(end_date)
    full_midx = pd.date_range(start_date, end_date, freq="MS")

    def _monthly_aggregate(values, times, how="first"):
        v = _sc_to_numeric_series(values)
        t = pd.to_datetime(times, errors="coerce")
        df = pd.DataFrame({"v": v}, index=t).dropna(subset=["v"])
        if df.empty: return pd.Series(index=full_midx, dtype=float)
        per = df.index.to_period("M")
        if how == "first": agg = df["v"].groupby(per).first()
        elif how == "max": agg = df["v"].groupby(per).max()
        elif how == "min": agg = df["v"].groupby(per).min()
        else: agg = df["v"].groupby(per).mean()
        agg.index = agg.index.to_timestamp("MS")
        return agg.reindex(full_midx)

    def _csv_series(sensor_num):
        if requests is None: return None
        base = "https://cdec.water.ca.gov/dynamicapp/req/CSVDataServlet"
        params = {"Stations": str(station_id).upper(), "SensorNums": str(sensor_num),
                  "dur_code": "M", "Start": start_date.strftime("%Y-%m-%d"),
                  "End": end_date.strftime("%Y-%m-%d")}
        url = f"{base}?{urlencode(params)}"
        try:
            r = requests.get(url, timeout=30); r.raise_for_status()
            df = pd.read_csv(io.StringIO(r.text))
        except Exception:
            return None
        if df is None or df.empty: return None
        df = df.rename(columns={c: str(c).strip() for c in df.columns})
        vcol, vals = _sc_find_swe_column_and_values(df)
        if vcol is None or vals is None: 
            if debug: print(f"[SC CSV] {station_id} s{sensor_num}: SWE-like column not found; cols={list(df.columns)}")
            return None
        t = _sc_parse_datetime_from_csv(df)
        if t.isna().all():
            if debug: print(f"[SC CSV] {station_id} s{sensor_num}: could not parse timestamps")
            return None
        ser = _monthly_aggregate(vals, t, monthly_agg)
        if debug:
            print(f"[SC CSV] {station_id} s{sensor_num}: rows={len(df)} ts_ok={t.notna().sum()} "
                  f"using '{vcol}' numeric_rows={vals.notna().sum()} monthly_nonnull={ser.notna().sum()}")
        return ser if ser.notna().any() else None

    def _json_series(sensor_num):
        if requests is None: return None
        base = "https://cdec.water.ca.gov/dynamicapp/req/JSONDataServlet"
        params = {"Stations": str(station_id).upper(), "SensorNums": str(sensor_num),
                  "dur_code": "M", "Start": start_date.strftime("%Y-%m-%d"),
                  "End": end_date.strftime("%Y-%m-%d")}
        url = f"{base}?{urlencode(params)}"
        headers = {"User-Agent": "Mozilla/5.0 (compatible; cdec-json/1.0)"}
        try:
            resp = requests.get(url, headers=headers, timeout=30); resp.raise_for_status()
            data = resp.json()
        except Exception:
            return None
        if not isinstance(data, list) or len(data) == 0: return None
        df = pd.DataFrame(data)
        t = None
        for dcol in ("datetime","date","obsDate","ObsDate","Date","MeasurementDate",
                     "measurement_date","time","Time","timestamp"):
            if dcol in df.columns:
                t = pd.to_datetime(df[dcol], errors="coerce"); break
        vcol = next((c for c in ("value","Value","val","Val","data","Data") if c in df.columns), None)
        if t is None or vcol is None: return None
        ser = _monthly_aggregate(df[vcol], t, monthly_agg)
        if debug:
            n_num = pd.to_numeric(df[vcol], errors="coerce").notna().sum()
            print(f"[SC JSON] {station_id} s{sensor_num}: rows={len(df)} numeric_in_{vcol}={n_num} "
                  f"monthly_nonnull={ser.notna().sum()}")
        return ser if ser.notna().any() else None

    # try standard monthly sensors
    s82 = _csv_series(82) or _json_series(82)
    s3  = _csv_series(3)  or _json_series(3)

    # if both empty → try metadata monthly sensors and the snowQuery fallback
    best = None; best_nonnull = -1; best_sensor = None
    if not _sc_series_usable(s82) and not _sc_series_usable(s3):
        for sn in _sc_monthly_candidate_sensors(station_id):
            ser = _csv_series(sn) or _json_series(sn)
            if _sc_series_usable(ser):
                nonnull = int(ser.notna().sum())
                if nonnull > best_nonnull:
                    best_nonnull, best, best_sensor = nonnull, ser, sn
        sq = _sc_fetch_snowquery_monthly(station_id, start_date, end_date, monthly_agg, debug=debug)
        if sq is not None:
            ser_sq = _monthly_aggregate(sq["values"], sq["times"], monthly_agg)
            if _sc_series_usable(ser_sq) and ser_sq.notna().sum() > best_nonnull:
                best, best_nonnull, best_sensor = ser_sq, int(ser_sq.notna().sum()), "snowQuery"

    # package
    wide = pd.DataFrame(index=full_midx)
    wide["SWE_ADJ"] = s82.reindex(full_midx) if _sc_series_usable(s82) else np.nan
    wide["SWE"]     = s3.reindex(full_midx)  if _sc_series_usable(s3)  else (best.reindex(full_midx) if _sc_series_usable(best) else np.nan)
    wide["SWE_clean"] = wide["SWE_ADJ"].combine_first(wide["SWE"])

    da = xr.DataArray((wide["SWE_clean"].to_numpy() * 25.4), dims=("time",), coords={"time": full_midx}, name=str(station_id))

    if plot:
        plt.figure(figsize=(8,3)); plt.plot(da.time, da, marker="o", lw=1)
        plt.title(f"{station_id} — Snow course SWE (monthly)"); plt.ylabel("mm"); plt.xlabel("Month"); plt.tight_layout()

    if debug:
        if _sc_series_usable(s82) or _sc_series_usable(s3):
            print(f"[SC PICK] {station_id}: 82{'✓' if _sc_series_usable(s82) else '✗'}, 3{'✓' if _sc_series_usable(s3) else '✗'}")
        elif _sc_series_usable(best):
            print(f"[SC PICK] {station_id}: fallback {best_sensor}✓ (monthly_nonnull={best_nonnull})")
        else:
            print(f"[SC PICK] {station_id}: no usable monthly SWE")

    return da, wide, s82, s3



# =========================
# Batch snow-course (monthly)
# =========================

def load_snow_course_api(buff_geog, start_date, end_date, plot=True, monthly_agg="first", debug=False):
    """
    Download MONTHLY snow-course SWE for stations in buff_geog (CDEC).
    Prefer SWE_ADJ (82) over SWE (3) when both exist; otherwise fall back to the
    best monthly SWE-like sensor from metadata. Units returned in mm.
    """
    import io, numpy as np, pandas as pd, xarray as xr, matplotlib.pyplot as plt
    from urllib.parse import urlencode
    try:
        import requests
    except Exception:
        requests = None

    start_date = pd.to_datetime(start_date, errors="coerce")
    end_date   = pd.to_datetime(end_date,   errors="coerce")
    if pd.isna(start_date) or pd.isna(end_date) or end_date < start_date:
        raise ValueError(f"Bad date range: start={start_date}, end={end_date}")
    full_midx = pd.date_range(start_date, end_date, freq="MS")

    apply_cdec_patches()

    # discover candidate stations
    try:
        points = CDECPointData.points_from_geometry(
            buff_geog, [CdecStationVariables.SWE], snow_courses=True
        )
    except TypeError:
        points = CDECPointData.points_from_geometry(
            buff_geog, [CdecStationVariables.SWE], snow_courses=False
        )
    cdec_locations = points.to_dataframe().sort_values("id")
    try:
        cdec_locations = cdec_locations.set_crs("EPSG:4326", allow_override=True)
    except Exception:
        pass

    # local aggregator captures full_midx
    def _monthly_aggregate(values, times, how="first"):
        v = _sc_to_numeric_series(values)
        t = pd.to_datetime(times, errors="coerce")
        df = pd.DataFrame({"v": v}, index=t).dropna(subset=["v"])
        if df.empty:
            return pd.Series(index=full_midx, dtype=float)
        per = df.index.to_period("M")
        if how == "first":
            agg = df["v"].groupby(per).first()
        elif how == "max":
            agg = df["v"].groupby(per).max()
        elif how == "min":
            agg = df["v"].groupby(per).min()
        else:
            agg = df["v"].groupby(per).mean()
        agg.index = agg.index.to_timestamp("MS")
        return agg.reindex(full_midx)

    def fetch_csv_series_monthly(station_id, sensor_num):
        if requests is None:
            return None
        base = "https://cdec.water.ca.gov/dynamicapp/req/CSVDataServlet"
        params = {
            "Stations": str(station_id).upper(),
            "SensorNums": str(sensor_num),
            "dur_code": "M",
            "Start": start_date.strftime("%Y-%m-%d"),
            "End":   end_date.strftime("%Y-%m-%d"),
        }
        url = f"{base}?{urlencode(params)}"
        try:
            r = requests.get(url, timeout=30); r.raise_for_status()
        except Exception:
            return None
        try:
            df = pd.read_csv(io.StringIO(r.text))
        except Exception:
            return None
        if df is None or df.empty:
            return None
        df = df.rename(columns={c: str(c).strip() for c in df.columns})
        vcol, vals = _sc_find_swe_column_and_values(df)
        if vcol is None or vals is None:
            if debug:
                print(f"[SC CSV] {station_id} s{sensor_num}: SWE-like column not found; cols={list(df.columns)}")
            return None
        t = _sc_parse_datetime_from_csv(df)
        if t.isna().all():
            if debug:
                print(f"[SC CSV] {station_id} s{sensor_num}: could not parse timestamps")
            return None
        ser = _monthly_aggregate(vals, t, monthly_agg)
        if debug:
            print(f"[SC CSV] {station_id} s{sensor_num}: rows={len(df)} ts_ok={t.notna().sum()} "
                  f"using '{vcol}' numeric_rows={vals.notna().sum()} monthly_nonnull={ser.notna().sum()}")
        return ser if ser.notna().any() else None

    def fetch_json_series_monthly(station_id, sensor_num):
        if requests is None:
            return None
        base = "https://cdec.water.ca.gov/dynamicapp/req/JSONDataServlet"
        params = {
            "Stations": str(station_id).upper(),
            "SensorNums": str(sensor_num),
            "dur_code": "M",
            "Start": start_date.strftime("%Y-%m-%d"),
            "End":   end_date.strftime("%Y-%m-%d"),
        }
        url = f"{base}?{urlencode(params)}"
        headers = {"User-Agent": "Mozilla/5.0 (compatible; cdec-json/1.0)"}
        try:
            resp = requests.get(url, headers=headers, timeout=30); resp.raise_for_status()
            data = resp.json()
        except Exception:
            return None
        if not isinstance(data, list) or len(data) == 0:
            return None
        import pandas as pd
        df = pd.DataFrame(data)
        t = None
        for dcol in ("datetime","date","obsDate","ObsDate","Date","MeasurementDate",
                     "measurement_date","time","Time","timestamp"):
            if dcol in df.columns:
                t = pd.to_datetime(df[dcol], errors="coerce"); break
        vcol = next((c for c in ("value","Value","val","Val","data","Data") if c in df.columns), None)
        if t is None or vcol is None:
            return None
        ser = _monthly_aggregate(df[vcol], t, monthly_agg)
        if debug:
            import pandas as pd
            n_num = pd.to_numeric(df[vcol], errors="coerce").notna().sum()
            print(f"[SC JSON] {station_id} s{sensor_num}: rows={len(df)} numeric_in_{vcol}={n_num} "
                  f"monthly_nonnull={ser.notna().sum()}")
        return ser if ser.notna().any() else None

    # metadata prefilter: keep stations that have at least one monthly SWE-like sensor
    def has_overlapping_monthly_swe(station):
        return len(_sc_monthly_candidate_sensors(station)) > 0

    ids_all = cdec_locations["id"].astype(str).tolist()
    valid_ids = [sid for sid in ids_all if has_overlapping_monthly_swe(sid)]
    if debug and len(valid_ids) < len(ids_all):
        dropped = sorted(set(ids_all) - set(valid_ids))
        print(f"[SC DEBUG] Prefilter dropped {len(dropped)} non-monthly SWE stations: {', '.join(dropped[:12])}" + ("..." if len(dropped) > 12 else ""))

    run_ids = valid_ids if len(valid_ids) > 0 else ids_all

    output, skipped, station_kind = [], [], {}

    for station in run_ids:
        candidates = _sc_monthly_candidate_sensors(station)

        s82 = None
        s3  = None
        best = None
        best_nonnull = -1
        best_sensor = None

        # Try metadata monthly sensors (CSV/JSON)
        for sn in candidates:
            ser = fetch_csv_series_monthly(station, sn) or fetch_json_series_monthly(station, sn)
            if _sc_series_usable(ser):
                nonnull = int(ser.notna().sum())
                if sn == "82":
                    s82 = ser
                elif sn == "3":
                    s3 = ser
                if nonnull > best_nonnull:
                    best, best_nonnull, best_sensor = ser, nonnull, sn

        # SnowQuery fallback (only if 82/3 both empty)
        if not _sc_series_usable(s82) and not _sc_series_usable(s3):
            sq = _sc_fetch_snowquery_monthly(station, start_date, end_date, monthly_agg, debug=debug)
            if sq is not None:
                ser_sq = _monthly_aggregate(sq["values"], sq["times"], monthly_agg)
                if _sc_series_usable(ser_sq):
                    ser_sq_n = int(ser_sq.notna().sum())
                    if ser_sq_n > best_nonnull:
                        best, best_nonnull, best_sensor = ser_sq, ser_sq_n, "snowQuery"

        if not _sc_series_usable(s82) and not _sc_series_usable(s3) and not _sc_series_usable(best):
            skipped.append(station)
            if debug:
                print(f"{station}: skipped (no MONTHLY SWE across sensors {candidates})")
            continue

        wide = pd.DataFrame(index=full_midx)
        wide["SWE_ADJ"] = s82.reindex(full_midx) if _sc_series_usable(s82) else np.nan
        wide["SWE"]     = (
            s3.reindex(full_midx) if _sc_series_usable(s3)
            else (best.reindex(full_midx) if _sc_series_usable(best) else np.nan)
        )
        wide["SWE_clean"] = wide["SWE_ADJ"].combine_first(wide["SWE"])

        if debug:
            picked = "82" if _sc_series_usable(s82) else ("3" if _sc_series_usable(s3) else best_sensor or "unknown")
            print(f"{station}: {int(wide['SWE_clean'].notna().sum())} month(s) with SWE (picked sensor {picked})")

        da = xr.DataArray(
            (wide["SWE_clean"].to_numpy() * 25.4),
            dims=("time",), coords={"time": full_midx}, name=str(station)
        )
        output.append(da)
        station_kind[str(station)] = "snow_course"

        if plot:
            plt.plot(da.time, da, marker="o", linestyle="-", label=f"{station} (snow course)")


    if skipped:
        print(f"Skipped {len(skipped)} station(s): {', '.join(skipped[:12])}" + ("..." if len(skipped) > 12 else ""))

    if plot and len(output) > 0:
        plt.legend(ncol=2, fontsize=8)
        plt.xlabel("Month"); plt.ylabel("Snow Water Equivalent [mm]")
        plt.ylim(0, 5000); plt.tight_layout()
        plt.savefig("snow_course_plots_api.png", dpi=200)

    cdec_locations = cdec_locations.copy()
    cdec_locations["kind"] = "snow_course"

    delete_cdec_lst = skipped
    return output, cdec_locations, delete_cdec_lst, station_kind


def snotel_fetch(sitecode, start_date, end_date, variablecode = 'SNOTEL:WTEQ_D'):
    """
    CUAHSI Snotel API and quality control cleaning
    Input:
      sitecode - python string object for snotel code. 
      start_date - datetime object of start date. 
      end_date - datetime object of end date. 
      variablecode - python string for snotel variable code.
    
    Output:
      output - list of dataarrays for snow pillows.
      aso_snotel - locations of snow pillows.
      
    """
    values_df = None
    wsdlurl = 'https://hydroportal.cuahsi.org/Snotel/cuahsi_1_1.asmx?WSDL'
    # try:
    ## start by requesting data from the server ##
    ## print('Requesting Data from Server...')
    site_values = ulmo.cuahsi.wof.get_values(wsdlurl,sitecode, variablecode, start=start_date, end=end_date)
    ## convert to pandas dataframe ##
    values_df = pd.DataFrame.from_dict(site_values['values'])
    ## parse the datetime values to Pandas Timestamp object ##
    ## print('Cleaning Data...') ##
    values_df['datetime'] = pd.to_datetime(values_df['datetime'], utc=False)
    ## set df index to datetime ##
    values_df = values_df.set_index('datetime')
    ## convert values to float and replace -9999 with NaN ##
    values_df['value'] = pd.to_numeric(values_df['value']).replace(-9999,np.nan)
    ## rename values column after variable code ##
    values_df.rename(columns = {'value':variablecode}, inplace = True)
    ## remove lower quality records ##
    values_df = values_df[values_df['quality_control_level_code']=='1']
    ## print sitecode ##
    print(sitecode,end = ' ')
    # except:
    #     print(f'\nCould not process {variablecode}')
    return values_df

def load_snotel_api(buff_geog, 
                    start_date, 
                    end_date,
                    pillow_geog,
                    variable = 'SNOTEL:WTEQ_D',
                    plot = False):
    """
    Call CUAHSI Snotel API for sites within basin of interest.
    Input:
      buff_geog - geopandas object for buffered geometry of aso basin. 
      start_date - datetime object of start date. 
      end_date - datetime object of end date. 
      plot - boolean to plot pillow timeseries.
    
    Output:
      output - list of dataarrays for snow pillows.
      aso_snotel - locations of snow pillows.
      
    """
    
    ## CUASI API ##
    wsdlurl = 'https://hydroportal.cuahsi.org/Snotel/cuahsi_1_1.asmx?WSDL' # snotel url
    ## identify SNOTEL site locations ##
    sites = ulmo.cuahsi.wof.get_sites(wsdlurl)    
    ## convert to pandas dataframe from dictionary and drop na values ##
    sites_df = pd.DataFrame.from_dict(sites, orient='index').dropna()
    sites_df['geometry']=[Point(float(loc['longitude']),float(loc['latitude'])) for loc in sites_df['location']]
    sites_df = pd.concat([sites_df, sites_df["location"].apply(pd.Series)], axis=1)
    sites_df = sites_df.drop(columns='location')
    sites_df = sites_df.astype({'elevation_m':float,'latitude':float,'longitude':float})
    sites_df['name'] = sites_df['name'].str.upper()
    
    ## gathers all snotel sites ##
    sites_gdf_all = gpd.GeoDataFrame(sites_df, crs='EPSG:4326')
    sites_gdf_all
    
    ## intersect snotel with aso basin ##
    aso_snotel = gpd.overlay(sites_gdf_all, buff_geog, how='intersection')
    ## aso snotel names ##
    aso_snotel_names = aso_snotel['code'].to_list()
    ## spatial snotel ##
    # aso_snotel = sites_gdf_all[sites_gdf_all['code'].isin(aso_snotel_names)]
    aso_snotel = sites_gdf_all.loc[sites_gdf_all['code'].isin(aso_snotel_names)].copy()
    ## loop through sites ##
    output = []
    # aso_snotel['id'] = ''
    aso_snotel.loc[:, 'id'] = ''
    id_lst = pillow_geog['id'].to_list()
    print('Downloading snotel data...')
    
    for i in aso_snotel.index:
        snotel_code = i
        site_name = aso_snotel.at[i, 'name']
        network = aso_snotel.at[i, 'network']
        lat = aso_snotel.at[i, 'latitude']
        lon = aso_snotel.at[i, 'longitude']
        abrev,id_lst = create_abrev(site_name,id_lst)
        # aso_snotel.at[i, 'id'] = abrev
        aso_snotel.loc[i, 'id'] = abrev
        
        # try:
        swe_snotel = snotel_fetch(snotel_code, 
                                 start_date,
                                 end_date,
                                 variable)
                                 
            ## create xarray datarray from timeseries (convert from inches to milimeters)##
        da = xr.DataArray(swe_snotel[variable].values* 0.0254 * 1000,
                     dims = ("time",), 
                     coords = {"time":swe_snotel.reset_index()['datetime'].values},
                     name = abrev)
            ## plotting ##
        if plot:
            plt.plot(da.time, da, label=site_name) 

            ## append dataArray into list ##
        output.append(da)
        # except:
        #     print(f'Could not process: {snotel_code}')
    
                                 
    if plot:
        plt.legend()
        plt.xlabel("Date")
        plt.ylabel("Snow Water Equivalent [mm]")
        plt.ylim(0,5000)
        plt.savefig("snotel_plots_api.png")
        
    ## clean up snotel geopandas ##
    return output,aso_snotel


def create_abrev(input_s,lsttoExclude):
    """
    Given an input string for the name of an observation station and a list of 
    other observation abreviations, this function will create a new abbreviation
    that is unique.
    Input:
      input_s - string object representing name of an observation station. 
      lsttoExclude - python list of strings for abreviations of stations.
    
    Output:
      tot - string representing 3 character abbreviation for observation.
      lsttoExclude - python list of strings for abreviations of stations.
      
    """
    ## instantiate variables ##
    abrev_ = []
    count = 0
    new_str = copy.deepcopy(input_s)
    blah = True
    tot = ""
    
    ## loop through to append capitilized characters ##
    for i in range(0,len(input_s)):
        if input_s[i].isupper():
            if count < 3:
                abrev_.append(input_s[i].upper())
                count += 1
                new_str = remove_character(new_str,i)
                
    ## append additional characters ##
    if len(abrev_) == 1:
        for i in range(len(abrev_),3):
            idx = random.randint(0,len(new_str)-1)
            abrev_.append(new_str[idx].upper())
            new_str = remove_character(new_str,idx)
    if len(abrev_) == 2:
        abrev_.append(new_str[-1].upper())
    for i in abrev_:
        tot = tot + i 
        
    ## check to make sure selected abbreviation is unique compared to the list 
    ##   of abbreviations. If not, randomly select other characters until 
    ##   abbreviation is unique. ##
    while blah == True:
        if tot in lsttoExclude:
            idx = random.randint(0,len(input_s)-1)
            if input_s[idx] != ' ':
                tot = tot[0:2] + input_s[idx].upper()
        else:
            blah = False
            
    lsttoExclude.append(tot)
    
    return tot,lsttoExclude

def remove_character(str_in,index):
    """
    Given an input string for the name of an observation station and a list of 
    other observation abreviations, this function will create a new abbreviation
    that is unique.
    Input:
      str_in - string object representing name of an observation station. 
      index - python integer representing index of character to remove.
    
    Output:
      str_out - python string with character removed.      
    """
    str_out = ""
    for i in range(len(str_in)):
        if i != index:
            str_out = str_out + str_in[i]
    return str_out

def load_lle_geography():
    """
    Load Levit Lake Geography for pillow download.
    Input:
    Output:
      lle_polygon: geodataframe.
    """
    levit_lat,levit_lon = 38.275940,-119.612808

    l_lat_min = levit_lat - 0.01
    l_lat_max = levit_lat + 0.01

    l_lon_min = levit_lon - 0.01
    l_lon_max = levit_lon + 0.01

    lat_point_list = [l_lat_min, l_lat_max, l_lat_max, l_lat_min, l_lat_min]
    lon_point_list = [l_lon_min, l_lon_min, l_lon_max, l_lon_max, l_lon_min]

    polygon_geom = Polygon(zip(lon_point_list, lat_point_list))
    lle_polygon = gpd.GeoDataFrame(index=[0], crs='epsg:4326', geometry=[polygon_geom]) 
    return lle_polygon


def combine_basin_obs(basin_geog, pillow_geog, snotel_geog, pillow_data, snotel_data):

    # ensure we’re not modifying a view
    pillow_geog = pillow_geog.copy()
    snotel_geog = snotel_geog.copy()

    # append snotel data
    for da in snotel_data:
        pillow_data.append(da)

    # add columns safely
    pillow_geog.loc[:, 'latitude'] = pillow_geog.geometry.y
    pillow_geog.loc[:, 'longitude'] = pillow_geog.geometry.x

    # only if Z exists (3D points). Some geometries may be 2D.
    try:
        pillow_geog.loc[:, 'elevation_ft'] = pillow_geog.geometry.z
        pillow_geog.loc[:, 'elevation_m'] = pillow_geog['elevation_ft'] * 0.3048
    except Exception:
        pillow_geog.loc[:, 'elevation_ft'] = np.nan
        pillow_geog.loc[:, 'elevation_m'] = np.nan

    combo_df = pd.concat([
        pillow_geog[['name','id','datasource','elevation_m','latitude','longitude']]
            .rename(columns={'datasource': 'network'}),
        snotel_geog.reset_index()[['name','id','network','elevation_m','latitude','longitude']]
    ]).reset_index()

    combo_gdf = gpd.GeoDataFrame(
        combo_df,
        geometry=gpd.points_from_xy(x=combo_df.longitude, y=combo_df.latitude),
        crs="EPSG:4326"
    )

    combo_gdf['network'] = combo_gdf['network'].astype(str).str.lower()
    combo_gdf.drop(columns='index', inplace=True, errors='ignore')

    return pillow_data, combo_gdf
