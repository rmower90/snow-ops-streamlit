# plotting.py
import os 
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Dict


def sheet_for_heatmap(sheet,end_date,area_dict):
    plot_cols = sheet.columns.to_list()
    plot_cols.remove('Model Type')
    plot_cols.remove('Training Infer NaNs')
    plot_cols.remove('Prediction QA')
    # clean table.
    new_sheet = sheet[plot_cols]
    new_sheet = new_sheet.set_index('Date')
    new_sheet = new_sheet[['Total','>12k','11k-12k','10k-11k','9k-10k','8k-9k','7k-8k','<7k']]
    new_sheet = new_sheet[new_sheet.index >= np.datetime64(end_date + timedelta(days=-14))]
    for col in new_sheet.columns:
        new_sheet[col] = new_sheet[col] / area_dict[col] * 12
    return new_sheet


def prediction_by_elevation_heatmap2(sheet1,sheet2,sheet3,sheet4,sheet5,sheet6,end_date,area_dict,aso_site_name,
                                     savePlot = True,showPlot = False):
    """
        Plot heatmap of last two week predictions by elevation.
        Input:
            sheet - geopandas dataframe for predictions.  
        Output:
            plot of basins.
    """
    # list of elevations.
    new_sheet1 = sheet_for_heatmap(sheet1,end_date,area_dict)
    new_sheet2 = sheet_for_heatmap(sheet2,end_date,area_dict)
    new_sheet3 = sheet_for_heatmap(sheet3,end_date,area_dict)
    new_sheet4 = sheet_for_heatmap(sheet4,end_date,area_dict)
    new_sheet5 = sheet_for_heatmap(sheet5,end_date,area_dict)
    new_sheet6 = sheet_for_heatmap(sheet6,end_date,area_dict)
    all_sheets = [new_sheet1,new_sheet2,new_sheet3,new_sheet4,new_sheet5,new_sheet6]

    # plotting.
    all_max = 0.0
    all_min = 1000.0
    for i in range(0,len(all_sheets)):
        max_ = all_sheets[i].max().max()
        min_ = all_sheets[i].min().min()
        if max_ > all_max:
            all_max = max_
        if min_ < all_min:
            all_min = min_

    fig,ax = plt.subplots(3,2,dpi = 250,sharex = True,sharey = True,constrained_layout = True)
    for i in range(0,len(all_sheets)):
        im = ax[i//2,i%2].imshow(all_sheets[i].T,cmap = 'inferno',vmax = all_max,vmin = all_min)
        if i%2 == 0:
            ax[i//2,i%2].set_yticks(ticks = np.arange(0,len(all_sheets[i].columns)), labels = all_sheets[i].columns)
        if i//2 == 2:
            ax[i//2,i%2].set_xticks(ticks = np.arange(0,len(all_sheets[i]))[::2], labels = all_sheets[i].index.strftime('%m-%d')[::2],rotation = 45)
    ax[0,0].set_title('Drop NaNs QA = 1',fontweight = 'bold')
    ax[0,1].set_title('Predict NaNs QA = 1',fontweight = 'bold')
    ax[1,0].set_title('Drop NaNs QA = 2',fontweight = 'bold')
    ax[1,1].set_title('Predict NaNs QA = 2',fontweight = 'bold')
    ax[2,0].set_title('Drop NaNs QA = 3',fontweight = 'bold')
    ax[2,1].set_title('Predict NaNs QA = 3',fontweight = 'bold')
    plt.suptitle('Model Predictions by Elevation',fontweight = 'bold',fontsize = 16)
    fig.subplots_adjust(right=0.90)
    cbar_ax = fig.add_axes([0.48, 0.10, 0.02, 0.8])
    fig.colorbar(im, cax=cbar_ax,label = 'SWE [in]')
    #plt.tight_layout()
    if savePlot:
        plt.savefig(f'./data/predictions/{aso_site_name}/historic/elevation_comparison.png',dpi = 250)
    if showPlot:
        plt.show()
    return



def create_prediction_plot(sheet1,sheet2,sheet3,sheet4,sheet5,sheet6,aso_site_name,
                           end_date,area_dict,basinID = 'SBF',showPlot = True,savePlot = True):
    """
        Plot heatmap of last two week predictions by elevation.
        Input:
            sheet - geopandas dataframe for predictions.  
        Output:
            plot of basins.
    """
    # instantiate figure.
    fig,ax = plt.subplots(1,2,figsize = (10,5),sharex = False,gridspec_kw = {'width_ratios': [4,1]})
    # ax[0] = plot_predicted_comparisons2(sheet1,sheet2,sheet3,sheet4,sheet5,sheet6,aso_site_name,end_date,ax[0],
    #                            basinID = 'SBF',showPlot = True,savePlot = True)
    ax[0] = plot_predicted_comparisons2(sheet1,sheet2,sheet3,sheet4,sheet5,sheet6,aso_site_name,end_date,ax[0],
                               basinID = 'SBF',showPlot = True,savePlot = True)
    # ax[0] = prediction_by_elevation_heatmap2(sheet1,end_date,area_dict,ax[0])
    ax[1] = prediction_by_elevation_heatmap2(sheet1,end_date,area_dict,ax[1])
    plt.tight_layout()
    plt.show()


def plot_predicted_comparisons3(sheet1,sheet2,sheet3,sheet4,sheet5,sheet6,aso_site_name,end_date,ax_,
                               basinID = 'SBF',showPlot = True,savePlot = True):
    

def plot_predicted_comparisons2(sheet1,sheet2,sheet3,sheet4,sheet5,sheet6,aso_site_name,end_date,ax_,
                               basinID = 'SBF',showPlot = True,savePlot = True):
    """
        Plot basin location with state lines.
        Input:
            shape_geog - geopandas shape of basin in geographic coordinates.
            states_fpath - python string for relative filepath to US States 
                           shapefile.
            proj_crs - python string for projected crs epgs.
            showPlot - boolean to produce plot.  
        Output:
            plot of basins.
    """
    # read in other model predictions.
    data_other = pd.read_csv('https://snow.water.ca.gov/service/plotly/data/download?dash=fcast_resources&file=csv/wsfr_snow.csv')
    data_other['DATE'] = pd.to_datetime(data_other['DATE'])
    # slice to basin.
    sj_other = data_other[data_other['STA_ID'] == basinID]
    sj_all = copy.deepcopy(sj_other)
    # slice dates.
    sj_other = sj_other[(sj_other['DATE'] >= np.datetime64(end_date + timedelta(days=-14))) & (sj_other['DATE'] <= np.datetime64(sheet2.Date.iloc[-1]))]
    # sj_other = sj_other[(sj_other['DATE'] >= np.datetime64('2024-11-15')) & (sj_other['DATE'] <= np.datetime64(sheet2.Date.iloc[-1]))]

    # drop missing columns.
    sj_other = sj_other.dropna(axis = 1,how = 'all')

    # divide values by 1000.
    col_names = sj_other.columns.to_list()
    # remove columns names.
    col_names.remove('DATE')
    col_names.remove('STA_ID')
    col_names.remove('STATION_NAME')
    for col in ['ISNOBAL_DWR_SWE_AF','SNODAS_SWE_AF','SNOW17_SWE_AF','SWANN_UA_SWE_AF']:
        sj_other[col] = sj_other[col]/1000
    # create day of year index.
    sj_other['day_of_year'] = sj_other['DATE'].dt.dayofyear
    sj_other = sj_other.set_index('day_of_year')

    # create median dataframe.
    snodas_df = sj_all[['DATE','SNODAS_SWE_AF']]
    snodas_df = snodas_df[snodas_df['DATE'] < np.datetime64(end_date + timedelta(days=-14))]
    snodas_df['day_of_year'] = snodas_df['DATE'].dt.dayofyear
    median_df = snodas_df.groupby('day_of_year')[['SNODAS_SWE_AF']].median() / 1000
    median_df = median_df.rename(columns = {'SNODAS_SWE_AF':'SNODAS_MEDIAN'})
    # merge median
    sj_other = pd.merge(sj_other,median_df,how = 'left',left_on = 'day_of_year',right_on = 'day_of_year')
    # clean up sheet data.
    sheet1['Date'] = pd.to_datetime(sheet1['Date'])
    sheet2['Date'] = pd.to_datetime(sheet2['Date'])
    sheet3['Date'] = pd.to_datetime(sheet3['Date'])
    sheet4['Date'] = pd.to_datetime(sheet4['Date'])
    sheet5['Date'] = pd.to_datetime(sheet5['Date'])
    sheet6['Date'] = pd.to_datetime(sheet6['Date'])

    nan_df_1 = sheet1[['Date','Total','Prediction QA','Training Infer NaNs']]
    nan_df_2 = sheet3[['Date','Total','Prediction QA','Training Infer NaNs']]
    nan_df_3 = sheet5[['Date','Total','Prediction QA','Training Infer NaNs']]

    pred_df_1 = sheet2[['Date','Total','Prediction QA','Training Infer NaNs']]
    pred_df_2 = sheet4[['Date','Total','Prediction QA','Training Infer NaNs']]
    pred_df_3 = sheet6[['Date','Total','Prediction QA','Training Infer NaNs']]

    df_nan__ = pd.concat([nan_df_1,nan_df_2,nan_df_3])
    df_pred__ = pd.concat([pred_df_1,pred_df_2,pred_df_3])
    df_nan__['Total'] = df_nan__['Total']/1000
    df_pred__['Total'] = df_pred__['Total']/1000
    df_sns = pd.concat([df_nan__,df_pred__])
    df_sns = df_sns.rename(columns = {'Training Infer NaNs':'MLR Prediction'})

    df_sns = df_sns[(df_sns['Date'] >= np.datetime64(end_date + timedelta(days=-14)))]
    # plotting.
    # fig,ax = plt.subplots(dpi = 150)
    g = sns.lineplot(ax=ax_,
                 x = 'Date',
                 y = 'Total',
                 hue = 'MLR Prediction',
                 style='MLR Prediction', 
                 markers=['D', 'X'],
                 data=df_sns,
                 dashes = False,
                 legend = True)
    first_legend = ax_.legend(title = 'MLR Prediction',loc = 'upper left',title_fontproperties={'weight': 'bold'},)



    l1 = ax_.plot(sj_other['DATE'],sj_other['ISNOBAL_DWR_SWE_AF'],color = 'C2',linestyle = '--',linewidth = 1)
    l2 = ax_.plot(sj_other['DATE'],sj_other['SNODAS_SWE_AF'],color = 'C3',linestyle = '--',linewidth = 1)
    l3 = ax_.plot(sj_other['DATE'],sj_other['SNOW17_SWE_AF'],color = 'C4',linestyle = '--',linewidth = 1)
    l4 = ax_.plot(sj_other['DATE'],sj_other['SWANN_UA_SWE_AF'],color = 'C5',linestyle = '--',linewidth = 1)
    l5 = ax_.plot(sj_other['DATE'],sj_other['SNODAS_MEDIAN'],color = 'black',linestyle = '-',linewidth = 2)

    # Make sure to pass single Line2D objects rather than 1-element lists:
    line1 = l1[0]
    line2 = l2[0]
    line3 = l3[0]
    line4 = l4[0]
    line5 = l5[0]

    # Second legend
    second_legend = ax_.legend(
    handles=[line1, line2, line3, line4,line5],
    labels=['ISNOBAL', 'SNODAS', 'SNOW17', 'SWANN','SNODAS_MEDIAN'],
    loc='lower right',
    title = 'Other Products',
    title_fontproperties={'weight': 'bold'},  # makes the legend title bold
    )

    # Add *both* legends
    ax_.add_artist(first_legend)
    ax_.add_artist(second_legend)
    ax_.set_ylabel('Basin Mean SWE [thousand acre-feet]',fontweight = 'bold')
    # plt.xlabel('Date',fontweight = 'bold')
    ax_.set_title('San Joaquin Total Basin Mean SWE Prediction',fontweight = 'bold')
    ax_.set_xticks(rotation = 45)
    # plt.ylim([0,1000])
    plt.tight_layout()
    if savePlot:
        plt.savefig(f'./prediction_comparison.png',dpi = 150)
    if showPlot:
        plt.show()


    return ax_

def plot_pillow_qa_timeline(
    ds_raw: "xr.Dataset",
    pillow: str,
    df_simple: pd.DataFrame,
    ds_qa: "xr.Dataset" = None,
    methods=("static", "voting", "snowmodel"),
    majority_k=2,
    start=None,
    end=None,
    title=None,
    saveFIG: bool = False,
    figDIR: str = None
):

    # --- pull SWE series ---
    raw = ds_raw[pillow].to_series()
    raw.index = pd.to_datetime(raw.index).normalize()

    qa = None
    if ds_qa is not None:
        qa = ds_qa[pillow].to_series()
        qa.index = pd.to_datetime(qa.index).normalize()

    # apply start/end to series
    if start is not None:
        start = pd.to_datetime(start).normalize()
        raw = raw.loc[start:]
        if qa is not None:
            qa = qa.loc[start:]
    if end is not None:
        end = pd.to_datetime(end).normalize()
        raw = raw.loc[:end]
        if qa is not None:
            qa = qa.loc[:end]

    # --- flags for this pillow ---
    d = df_simple.copy()
    d["time"] = pd.to_datetime(d["time"]).dt.normalize()
    d["pillow"] = d["pillow"].astype(str).str.strip()
    d["method"] = d["method"].astype(str).str.strip()
    d["flag"] = d["flag"].fillna(0).astype(int)

    d = d[(d["pillow"] == pillow) & (d["method"].isin(methods))]

    if start is not None:
        d = d[d["time"] >= start]
    if end is not None:
        d = d[d["time"] <= end]

    # pivot to time x method, force all methods present
    flags = d.pivot_table(index="time", columns="method", values="flag", aggfunc="max")
    flags = flags.reindex(columns=list(methods)).fillna(0).astype(int)

    # --- build a CLEAN daily index for plotting ---
    # Use the min/max across available series so bands line up with days
    tmin = raw.index.min()
    tmax = raw.index.max()
    if qa is not None:
        tmin = min(tmin, qa.index.min())
        tmax = max(tmax, qa.index.max())
    if len(flags.index) > 0:
        tmin = min(tmin, flags.index.min())
        tmax = max(tmax, flags.index.max())

    idx = pd.date_range(tmin, tmax, freq="D")

    raw = raw.reindex(idx)
    if qa is not None:
        qa = qa.reindex(idx)
    flags = flags.reindex(idx).fillna(0).astype(int)

    # widen single-day flags for plotting visibility
    flags_plot = flags.copy()
    for c in flags_plot.columns:
        s = flags_plot[c]
        # mark day before and after a single isolated flag
        single = (s == 1) & (s.shift(1, fill_value=0) == 0) & (s.shift(-1, fill_value=0) == 0)
        flags_plot.loc[single.shift(1, fill_value=False), c] = 1
        flags_plot.loc[single.shift(-1, fill_value=False), c] = 1


    # majority
    # maj = (flags_plot.sum(axis=1) >= majority_k).astype(int)
    maj_true = (flags.sum(axis=1) >= majority_k).astype(int)   # computed from true flags
    maj_plot = (flags_plot.sum(axis=1) >= majority_k).astype(int)  # optional, if you want widened majority too


    # manual removal flag
    if qa is not None:
        manual_removed = ((raw.notna()) & (qa.isna())).astype(int)
    else:
        manual_removed = pd.Series(0, index=idx)

    # --- plotting ---
    fig, ax = plt.subplots(figsize=(13, 5))
    ax.plot(raw.index, raw.values, label="Raw SWE", linewidth=1.8)
    if qa is not None:
        ax.plot(qa.index, qa.values, label="Manual QA SWE", linewidth=2.2, linestyle="--")

    # y-scale helpers for the bands
    y0, y1 = ax.get_ylim()
    yr = y1 - y0
    band_h = 0.06 * yr
    gap = 0.01 * yr
    base = y0 + 0.02 * yr

    # manual removed band
    ax.fill_between(
        idx,
        base,
        base + band_h * 0.6,
        where=manual_removed.astype(bool).values,
        alpha=0.25,
        label="Manual removed",
        step="post",
    )
    cur_base = base + band_h * 0.6 + gap

    # method bands
    for m in methods:
        ax.fill_between(
            idx,
            cur_base,
            cur_base + band_h,
            where=flags_plot[m].astype(bool).values,
            alpha=0.25,
            label=f"{m} flag",
            step="post",
        )
        cur_base += band_h + gap

    # majority band
    ax.fill_between(
        idx,
        cur_base,
        cur_base + band_h * 1.2,
        where=maj_plot.astype(bool).values,
        alpha=0.35,
        label=f"Majority (≥{majority_k}/{len(methods)})",
        step="post",
    )

    ax.set_ylabel("SWE (mm)")
    ax.set_xlabel("Date")
    if title is None:
        title = f"{pillow} — Raw vs Manual QA + Method Flags"
    ax.set_title(title)
    ax.legend(ncol=3, loc="upper left")
    ax.grid(True, alpha=0.2)
    plt.tight_layout()

    if saveFIG:
        if figDIR is None:
            figDIR = "./qa_viz_2026/"
        if not os.path.exists(figDIR):
            os.makedirs(figDIR)
        plt.savefig(f"{figDIR}{pillow}.png", dpi=150)
        plt.close()
    else:
        plt.show()

    if qa is not None:
        return {"raw": raw, "qa": qa, "flags": flags_plot, "majority": maj_true, "manual_removed": manual_removed}
    else:
        return {"raw": raw, "flags": flags_plot, "majority": maj_true, "manual_removed": manual_removed}


def plot_qa_method_diagnostics(
    ds_raw,
    df_detail: pd.DataFrame,
    pillow: str,
    ds_qa: "xr.Dataset" = None,
    qa_method: str = "static",   # "static" | "voting" | "snowmodel"
    pil_corr: dict  = None,
    top_k: int = 5,
    start=None,
    end=None,
    show_manual_removed: bool = True,
    title: str = None,
    pred_ds=None,               # optional: xarray Dataset with swed_best/second/third (time,pil)
    majority_df_simple=None,    # optional: df_simple (to compute majority flags and show on strip)
    majority_k: int = 2,
    methods_for_majority=("static", "voting", "snowmodel"),
    saveFIG: bool = False,
    figDIR: str = None
):
    """
    General QA diagnostic plot for one pillow + one method.

    Requires:
      ds_raw[pillow] exists
      ds_qa[pillow] optional (manual QA)
      df_detail includes rows with columns at least: time, pillow, method, flag, reason_code/reason_detail (optional)

    Method-specific (if available, will be plotted):
      - static (peer-median): peer_median_mm, thr_mm, resid_mm
      - voting: diff_mm and dq_lo/dq_hi envelope bounds (or similar)
      - snowmodel: yhat_mm and thr_mm (and optionally pred_ds swed_best/second/third)
    """

    qa_method = str(qa_method).strip().lower()
    pillow = str(pillow).strip()

    # --- series ---
    raw = ds_raw[pillow].to_series()
    raw.index = pd.to_datetime(raw.index).normalize()

    qa = None
    if ds_qa is not None and pillow in ds_qa:
        qa = ds_qa[pillow].to_series()
        qa.index = pd.to_datetime(qa.index).normalize()

    # slice start/end
    if start is not None:
        start = pd.to_datetime(start).normalize()
        raw = raw.loc[start:]
        if qa is not None:
            qa = qa.loc[start:]
    if end is not None:
        end = pd.to_datetime(end).normalize()
        raw = raw.loc[:end]
        if qa is not None:
            qa = qa.loc[:end]

    # build daily plotting index
    tmin, tmax = raw.index.min(), raw.index.max()
    if qa is not None and len(qa.index) > 0:
        tmin = min(tmin, qa.index.min())
        tmax = max(tmax, qa.index.max())
    idx = pd.date_range(tmin, tmax, freq="D")
    raw = raw.reindex(idx)
    if qa is not None:
        qa = qa.reindex(idx)

    manual_removed = pd.Series(0, index=idx)
    if show_manual_removed and qa is not None:
        manual_removed = ((raw.notna()) & (qa.isna())).astype(int)

    # --- detail rows for this method/pillow ---
    d = df_detail.copy()
    d["time"] = pd.to_datetime(d["time"]).dt.normalize()
    d["pillow"] = d["pillow"].astype(str).str.strip()
    d["method"] = d["method"].astype(str).str.strip().str.lower()

    d = d[(d["pillow"] == pillow) & (d["method"] == qa_method)].sort_values("time")
    if start is not None:
        d = d[d["time"] >= start]
    if end is not None:
        d = d[d["time"] <= end]

    d = d.set_index("time").reindex(idx)

    # flags + reasons
    final_flag = d["flag"].fillna(0).astype(int) if "flag" in d.columns else pd.Series(0, index=idx)
    reason_code = d["reason_code"].astype(str) if "reason_code" in d.columns else pd.Series("", index=idx)
    reason_detail = d["reason_detail"].astype(str) if "reason_detail" in d.columns else pd.Series("", index=idx)

    # --- optional: majority flags computed from df_simple ---
    maj = None
    if majority_df_simple is not None:
        s = majority_df_simple.copy()
        s["time"] = pd.to_datetime(s["time"]).dt.normalize()
        s["pillow"] = s["pillow"].astype(str).str.strip()
        s["method"] = s["method"].astype(str).str.strip().str.lower()
        s["flag"] = s["flag"].fillna(0).astype(int)
        s = s[(s["pillow"] == pillow) & (s["method"].isin([m.lower() for m in methods_for_majority]))]
        if start is not None:
            s = s[s["time"] >= start]
        if end is not None:
            s = s[s["time"] <= end]
        flags_wide = s.pivot_table(index="time", columns="method", values="flag", aggfunc="max")
        flags_wide = flags_wide.reindex(columns=[m.lower() for m in methods_for_majority]).fillna(0).astype(int)
        flags_wide = flags_wide.reindex(idx).fillna(0).astype(int)
        maj = (flags_wide.sum(axis=1) >= majority_k).astype(int)

    # --------------------------
    # Helper: fetch first existing col in candidates
    def _col(*cands):
        for c in cands:
            if c in d.columns:
                return c
        return None

    # method-specific series
    peer_med = thr = yhat = diff = dq_lo = dq_hi = None

    if qa_method == "static":
        peer_col = _col("peer_median_mm", "peer_med_mm", "peer_median", "peer_med")
        thr_col  = _col("thr_mm", "peer_thr_mm", "thr", "peer_thr")
        if peer_col is not None:
            peer_med = d[peer_col].astype(float)
        if thr_col is not None:
            thr = d[thr_col].astype(float)

    elif qa_method == "snowmodel":
        yhat_col = _col("yhat_mm", "yhat")
        thr_col  = _col("thr_mm", "thr")
        if yhat_col is not None:
            yhat = d[yhat_col].astype(float)
        if thr_col is not None:
            thr = d[thr_col].astype(float)

    elif qa_method == "voting":
        diff_col = _col("diff_mm", "dswe_mm", "delta_mm")
        lo_col   = _col("dq_lo", "diff_q_lo", "q_lo_mm")
        hi_col   = _col("dq_hi", "diff_q_hi", "q_hi_mm")
        if diff_col is not None:
            diff = d[diff_col].astype(float)
        if lo_col is not None:
            dq_lo = d[lo_col].astype(float)
        if hi_col is not None:
            dq_hi = d[hi_col].astype(float)

    # correlated voters (optional)
    voters = []
    voter_series = {}
    if pil_corr is not None:
        voters = [v for v in pil_corr.get(pillow, []) if v in ds_raw]
        voters = voters[:top_k]
        for v in voters:
            s = ds_raw[v].to_series()
            s.index = pd.to_datetime(s.index).normalize()
            voter_series[v] = s.reindex(idx)

    # --------------------------
    # Figure layout
    # rows:
    # 0: raw/qa + method overlays (peer band or yhat band)
    # 1: method-specific (voters or diff envelope or swed_best/second/third)
    # 2: flag strip (manual removed, method flag, majority optional)
    nrows = 3
    height_ratios = [3, 2, 1]

    fig, axes = plt.subplots(
        nrows=nrows, ncols=1, figsize=(14, 8), sharex=True,
        gridspec_kw={"height_ratios": height_ratios},
    )

    ax0, ax1, axf = axes

    # --- panel 0: target series ---
    ax0.plot(idx, raw.values, label="Raw SWE", linewidth=1.8)
    if qa is not None:
        ax0.plot(idx, qa.values, label="Manual QA SWE", linewidth=2.2, linestyle="--")

    # static: peer band
    if qa_method == "static" and peer_med is not None and thr is not None:
        ax0.plot(idx, peer_med.values, label="peer median", linewidth=1.5)
        ax0.fill_between(idx, (peer_med - thr).values, (peer_med + thr).values, alpha=0.2, label="peer ± thr")

    # snowmodel: yhat band
    if qa_method == "snowmodel" and yhat is not None and thr is not None:
        ax0.plot(idx, yhat.values, label="snowmodel yhat", linewidth=1.5)
        ax0.fill_between(idx, (yhat - thr).values, (yhat + thr).values, alpha=0.2, label="yhat ± thr")

    # mark final flags
    ax0.scatter(idx[final_flag.astype(bool)], raw[final_flag.astype(bool)],
                s=18, zorder=5, label=f"{qa_method} flag")

    ax0.set_ylabel("SWE (mm)")
    ax0.grid(True, alpha=0.2)

    if title is None:
        title = f"{pillow} — {qa_method} diagnostics"
    ax0.set_title(title)
    ax0.legend(loc="upper left", ncol=4)

    # --- panel 1: method-specific context ---
    if qa_method == "static" and voters:
        for v, s in voter_series.items():
            ax1.plot(idx, s.values, linewidth=1.1, label=v)
        ax1.set_ylabel("Peer SWE (mm)")
        ax1.legend(loc="upper left", ncol=6, fontsize=9)
        ax1.grid(True, alpha=0.2)

    elif qa_method == "voting":
        ax1.axhline(0.0, linewidth=1.0, alpha=0.4)
        if diff is not None:
            ax1.plot(idx, diff.values, linewidth=1.4, label="ΔSWE (diff_mm)")
        if dq_lo is not None and dq_hi is not None:
            ax1.fill_between(idx, dq_lo.values, dq_hi.values, alpha=0.2, label="ΔSWE envelope")
        ax1.set_ylabel("ΔSWE (mm/day)")
        ax1.grid(True, alpha=0.2)
        ax1.legend(loc="upper left", ncol=3)

    elif qa_method == "snowmodel" and pred_ds is not None:
        # pred_ds expected variables swed_best/second/third with dims (time,pil)
        try:
            for vn, lab in [("swed_best", "best"), ("swed_second", "second"), ("swed_third", "third")]:
                if vn in pred_ds:
                    s = pred_ds[vn].sel(pil=pillow).to_series()
                    s.index = pd.to_datetime(s.index).normalize()
                    ax1.plot(idx, s.reindex(idx).values, linewidth=1.2, label=lab)
            ax1.set_ylabel("SnowModel SWE (mm)")
            ax1.grid(True, alpha=0.2)
            ax1.legend(loc="upper left", ncol=4, fontsize=9)
        except Exception as e:
            ax1.text(0.01, 0.85, f"pred_ds plot skipped: {e}", transform=ax1.transAxes)

    else:
        ax1.set_axis_off()

    # --- panel 2: flag strip + reasons ---
    axf.set_ylim(0, 1)
    axf.set_yticks([])
    axf.grid(False)

    # manual removed
    axf.fill_between(idx, 0.00, 0.33, where=manual_removed.astype(bool).values,
                     step="post", alpha=0.25, label="Manual removed")

    # method flag
    axf.fill_between(idx, 0.33, 0.66, where=final_flag.astype(bool).values,
                     step="post", alpha=0.35, label=f"{qa_method} flag")

    # majority (optional)
    if maj is not None:
        axf.fill_between(idx, 0.66, 1.00, where=maj.astype(bool).values,
                         step="post", alpha=0.25, label=f"Majority (≥{majority_k}/{len(methods_for_majority)})")

    axf.legend(loc="upper left", ncol=3)
    axf.set_xlabel("Date")

    plt.tight_layout()
    if saveFIG:
        if figDIR is None:
            figDIR = "./qa_method_diagnostics_2026/"
        if not os.path.exists(figDIR):
            os.makedirs(figDIR)
        plt.savefig(f"{figDIR}{pillow}_{qa_method}_diagnostics.png", dpi=150)
        plt.close()
    else:
        plt.show()

    return {
        "raw": raw,
        "qa": qa,
        "detail": d,
        "final_flag": final_flag,
        "reason_code": reason_code,
        "reason_detail": reason_detail,
        "peer_median": peer_med,
        "yhat": yhat,
        "thr": thr,
        "diff": diff,
        "dq_lo": dq_lo,
        "dq_hi": dq_hi,
        "voters": voters,
        "voter_series": voter_series,
        "majority": maj,
    }
