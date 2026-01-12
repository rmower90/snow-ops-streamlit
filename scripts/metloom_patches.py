# metloom_patches.py

# load_data.py
import re
import logging
import requests
import numpy as np
import pandas as pd
import geopandas as gpd
from io import StringIO

log = logging.getLogger(__name__)

def apply_cdec_patches(patch_pandas: bool = False):
    """
    Apply robust monkey-patches to metloom's CDECPointData to tolerate
    CDEC's JS/DataTables HTML, column name drift, and empty sensor responses.

    Call this BEFORE importing/using CDECPointData elsewhere:
        from load_data import apply_cdec_patches
        apply_cdec_patches()
        from metloom.pointdata.cdec import CDECPointData
    """
    from metloom.pointdata.cdec import CDECPointData  # import here to control timing

    # 1) two ArcGIS services / layers
    _ARCGIS_ENDPOINTS = [   
    "https://services.gis.ca.gov/arcgis/rest/services/AtmosphereClimate/CDEC_Stations/MapServer/{layer}/query",
    "https://ferix.water.ca.gov/arcgis/rest/services/ferix/cdecStations/MapServer/{layer}/query",
    ]
    _orig_sensor_response_to_df = CDECPointData._sensor_response_to_df

    def _arcgis_lookup_station_latlon(sta):
        params_base = { 
        "where": f"STA='{sta}'",
        "outFields": "STA,Station_Name,Latitude,Longitude,Elevation,STATION_NAME,LATITUDE,LONGITUDE,ELEVATION",
        "returnGeometry": "false",
        "f": "json",
        }
        for url in _ARCGIS_ENDPOINTS:
            for layer in (1, 0):
                try:    
                    r = requests.get(url.format(layer=layer), params=params_base, timeout=12)
                    r.raise_for_status()
                    js = r.json()
                    feats = js.get("features", [])
                    if not feats:
                        continue
                    attrs = feats[0].get("attributes", {}) or {}
                    # accept both title-case and ALLCAPS field variants
                    lat = attrs.get("Latitude", attrs.get("LATITUDE"))
                    lon = attrs.get("Longitude", attrs.get("LONGITUDE"))
                    elev = attrs.get("Elevation", attrs.get("ELEVATION"))
                    if lat is not None and lon is not None:
                        return float(lat), float(lon), (float(elev) if elev is not None else None)
                except Exception:
                    continue
        return None, None, None

    # --------- helpers ---------
    # def _safe_read_html_all(text: str):
    #     """Try multiple parsers; return [] instead of raising."""
    #     for flavor in (None, "html5lib"):
    #         try:
    #             tables = pd.read_html(text, flavor=flavor, displayed_only=False)
    #             if tables:
    #                 return tables
    #         except Exception:
    #             continue
    #     return []
    def _safe_read_html_all(text: str):
        """Try multiple parsers; return [] instead of raising."""
        sio = StringIO(text)  # pandas wants file-like, not raw html string
        for flavor in (None, "html5lib"):
            try:
                # need a fresh handle each time (StringIO gets consumed)
                tables = pd.read_html(StringIO(text), flavor=flavor, displayed_only=False)
                if tables:
                    return tables
            except Exception:
                continue
        return []
    def _normalize_location_df(df: pd.DataFrame) -> pd.DataFrame:
        """Map variants to ['Latitude','Longitude','Elevation'] if present."""
        if df is None or df.empty:
            return df
        mapping = {}
        for col in df.columns:
            cname = str(col).strip()
            if re.search(r"^(lat|latitude)\b", cname, flags=re.I):
                mapping[col] = "Latitude"
            elif re.search(r"^(lon|long|longitude)\b", cname, flags=re.I):
                mapping[col] = "Longitude"
            elif re.search(r"^elev|elevation", cname, flags=re.I):
                mapping[col] = "Elevation"
        return df.rename(columns=mapping) if mapping else df

    def _normalize_sensors_df(df: pd.DataFrame) -> pd.DataFrame:
        """Map variants to ['Sensor Number','Duration'] if present."""
        if df is None or df.empty:
            return df
        mapping = {}
        for col in df.columns:
            c = str(col).strip()
            if re.search(r"sensor\s*(num|no|number)", c, flags=re.I):
                mapping[col] = "Sensor Number"
            elif re.search(r"^duration|dur\b", c, flags=re.I):
                mapping[col] = "Duration"
        return df.rename(columns=mapping) if mapping else df

    def _regex_location_from_html(html_text: str):
        """
        Fallback: pull lat/lon/elev from staMeta raw HTML if no parseable table.
        Returns (lat, lon, elev_ft or None).
        """
        lat = lon = elev_ft = None
        m = re.search(
            r"Latitude\s*([+-]?\d+(?:\.\d+)?)\D+Longitude\s*([+-]?\d+(?:\.\d+)?)",
            html_text, flags=re.I
        )
        if m:
            lat = float(m.group(1))
            lon = float(m.group(2))
        e = re.search(r"Elevation\s*([0-9,]+(?:\.\d+)?)\s*ft", html_text, flags=re.I)
        if e:
            elev_ft = float(e.group(1).replace(",", ""))
        return lat, lon, elev_ft

    # --------- patched methods ---------
    def _get_all_metadata_safe(self):
        """
        Fetch staMeta HTML safely; try metloom's parser; always return a dict with:
          - 'sensors': DataFrame or empty DataFrame
          - 'location': DataFrame or empty DataFrame
          - '_raw_html': original HTML string for regex fallback
        """
        if getattr(self, "_raw_metadata", None) is None:
            url = self.META_URL + f"?station_id={self.id}"
            try:
                r = requests.get(url, timeout=15, headers={"User-Agent": "Mozilla/5.0"})
                r.raise_for_status()
                tables = _safe_read_html_all(r.text)

                parsed = {"sensors": pd.DataFrame(), "location": pd.DataFrame()}
                try:
                    if tables:
                        p = self._parse_meta_page(tables)
                        if isinstance(p, dict):
                            parsed["sensors"] = p.get("sensors", pd.DataFrame())
                            parsed["location"] = p.get("location", pd.DataFrame())
                except Exception:
                    # fall through to empty parsed
                    pass

                # normalize shapes/types and columns
                if isinstance(parsed["sensors"], list):
                    parsed["sensors"] = pd.DataFrame(parsed["sensors"])
                if not isinstance(parsed["sensors"], pd.DataFrame):
                    parsed["sensors"] = pd.DataFrame()
                parsed["sensors"] = _normalize_sensors_df(parsed["sensors"])

                if not isinstance(parsed["location"], pd.DataFrame):
                    parsed["location"] = pd.DataFrame()
                parsed["location"] = _normalize_location_df(parsed["location"])

                parsed["_raw_html"] = r.text  # keep for regex fallback
                self._raw_metadata = parsed

            except Exception as e:
                log.warning("CDEC staMeta fetch failed for %s: %s", getattr(self, "id", "?"), e)
                self._raw_metadata = {"sensors": pd.DataFrame(),
                                      "location": pd.DataFrame(),
                                      "_raw_html": ""}

        return self._raw_metadata

    def _get_metadata_robust(self):
        """
        Build shapely Point from metadata:
          1) Use 'location' table if available (Latitude/Longitude/Elevation).
          2) Else regex staMeta HTML for lat/lon/elev.
          3) Else last-resort dummy (0,0) point so downstream code doesn't crash.
        """
        data = self._get_all_metadata()
        loc = data.get("location", pd.DataFrame())

        # Case 1: use parsed table
        if isinstance(loc, pd.DataFrame) and not loc.empty:
            row = loc.iloc[0]
            lon_key = next((k for k in row.index if str(k).lower().startswith(("long", "lon"))), None)
            lat_key = next((k for k in row.index if str(k).lower().startswith(("lat",))), None)

            if lon_key is not None and lat_key is not None:
                try:
                    lon = self._parse_str_deg(row[lon_key])
                    lat = self._parse_str_deg(row[lat_key])
                except Exception:
                    lon = float(row[lon_key]); lat = float(row[lat_key])
                elev_key = next((k for k in row.index if re.search(r"^elev|elevation", str(k), flags=re.I)), None)
                if elev_key is not None and pd.notna(row[elev_key]):
                    try:
                        z = float(str(row[elev_key]).replace(",", ""))
                        return gpd.points_from_xy([lon], [lat], z=[z])[0]
                    except Exception:
                        return gpd.points_from_xy([lon], [lat])[0]
                return gpd.points_from_xy([lon], [lat])[0]

        # Case 2) ArcGIS fallback
        # lat, lon, elev_ft = _arcgis_lookup_station_latlon(self.id)
        # if (lat is not None) and (lon is not None):
        #     if elev_ft is not None:
        #         return gpd.points_from_xy([lon], [lat], z=[elev_ft])[0]
        #     return gpd.points_from_xy([lon], [lat])[0]
        
        lat, lon, elev_ft = _arcgis_lookup_station_latlon(self.id)
        if (lat is not None) and (lon is not None):
            if elev_ft is not None:
                return gpd.points_from_xy([lon], [lat], z=[elev_ft])[0]
            return gpd.points_from_xy([lon], [lat])[0]
        
        # Case 3: regex fallback
        lat, lon, elev_ft = _regex_location_from_html(data.get("_raw_html", "") or "")
        if lat is not None and lon is not None:
            if elev_ft is not None:
                return gpd.points_from_xy([lon], [lat], z=[elev_ft])[0]
            return gpd.points_from_xy([lon], [lat])[0]

        # Case 4: last-resort dummy point (prevents hard crash)
        log.warning("CDEC location missing for %s; using (0,0) placeholder geometry.", getattr(self, "id", "?"))
        return gpd.points_from_xy([0.0], [0.0])[0]
    

    def _sensor_response_to_df_lenient(self, response_data, sensor, final_columns, resample_duration=None):
    # Build a plain DataFrame first (no geometry dependency)
        if not response_data:
            df = gpd.GeoDataFrame(columns=final_columns)
            df.index = pd.DatetimeIndex([], name="date")
            return df

        df = pd.DataFrame(response_data)
        # fix sentinel, map date cols, etc. (metloom does more; keep minimal here)
        df.replace(-9999.0, np.nan, inplace=True)

        # prefer obsDate if present; else 'date'
        time_col = "obsDate" if "obsDate" in df.columns else ("date" if "date" in df.columns else None)
        if time_col:
            df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
            df = df.set_index(time_col)

        # Add required columns if missing so merges don't fail
        for col in ("sensor_number", "duration", "measurement_date"):
            if col not in df.columns:
                df[col] = pd.NA

        # Try to attach geometry, but don't block if metadata missing
        geom = None
        try:
            geom = self.metadata
        except Exception:
            pass
        gdf = gpd.GeoDataFrame(df, geometry=[geom] * len(df) if geom is not None else None)
        return gdf

    def _check_snowcourse_sensors_robust(self):
        """
        Robust snow-course cadence check. Returns [] if insufficient info
        (causing is_only_snow_course to NOT filter the station out).
        """
        data = self._get_all_metadata()
        sensors = data.get("sensors", None)

        if sensors is None:
            return []
        if isinstance(sensors, list):
            sensors_df = pd.DataFrame(sensors)
        elif isinstance(sensors, pd.DataFrame):
            sensors_df = sensors.copy()
        else:
            sensors_df = pd.DataFrame()

        if sensors_df.empty:
            return []

        sensors_df = _normalize_sensors_df(sensors_df)
        if "Sensor Number" not in sensors_df.columns or "Duration" not in sensors_df.columns:
            return []

        snow_sensors = {18, 3, 82}  # CDEC snow course sensor numbers
        manual_check = []
        for _, row in sensors_df.iterrows():
            try:
                num = int(str(row["Sensor Number"]).strip())
            except Exception:
                continue
            if num in snow_sensors:
                d = str(row.get("Duration", "")).strip().lower()
                manual_check.append(d == "monthly")
        return manual_check

    # Optional guard: empty response_data from JSON API -> empty GeoDataFrame
    _orig_sensor_response_to_df = CDECPointData._sensor_response_to_df
    def _sensor_response_to_df_guard(self, response_data, *args, **kwargs):
        if not response_data:
            cols = ["date", "value", "sensor_number", "duration", "measurement_date", "geometry"]
            return gpd.GeoDataFrame(columns=cols, geometry="geometry")
        try:
            return _orig_sensor_response_to_df(self, response_data, *args, **kwargs)
        except Exception as e:
            log.warning("CDEC _sensor_response_to_df failed for %s: %s", getattr(self, "id", "?"), e)
            cols = ["date", "value", "sensor_number", "duration", "measurement_date", "geometry"]
            return gpd.GeoDataFrame(columns=cols, geometry="geometry")
        
    

    # --------- apply patches ---------
    CDECPointData._get_all_metadata = _get_all_metadata_safe
    CDECPointData._get_metadata = _get_metadata_robust
    CDECPointData._check_snowcourse_sensors = _check_snowcourse_sensors_robust
    CDECPointData._sensor_response_to_df = _sensor_response_to_df_guard
    CDECPointData._sensor_response_to_df = _sensor_response_to_df_lenient

    # (Optional) make pd.read_html globally forgiving
    if patch_pandas and not hasattr(pd, "_read_html_orig"):
        pd._read_html_orig = pd.read_html  # type: ignore[attr-defined]
        def _patched_read_html(*a, **k):
            try:
                return pd._read_html_orig(*a, **k)
            except Exception:
                return []
        pd.read_html = _patched_read_html  # type: ignore[assignment]

    log.info("Applied CDECPointData resilience patches.")
