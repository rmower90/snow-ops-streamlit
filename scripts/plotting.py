# plotting.py
import os 
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Dict
import numpy as np
import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle


def timeseries_pillow_selection(mlr_pred_lst: List,
                                mlr_pred_dict: Dict,
                                uaswe_m_df: pd.DataFrame,
                                snodas_m_df: pd.DataFrame,
                                obs_raw: xr.Dataset,
                                obs_clean: xr.Dataset,
                                start_date: str = None,
                                end_date: str = None,
                                model_type: str = 'accum',
                                model_units: str = 'mm'
                                ):
    """
    Docstring for timeseries_pillow_selection
    
    :param mlr_pred_lst: Description
    :type mlr_pred_lst: List
    :param mlr_pred_dict: Description
    :type mlr_pred_dict: Dict
    :param uaswe_m_df: Description
    :type uaswe_m_df: pd.DataFrame
    :param snodas_m_df: Description
    :type snodas_m_df: pd.DataFrame
    :param obs_raw: Description
    :type obs_raw: xr.Dataset
    :param obs_clean: Description
    :type obs_clean: xr.Dataset
    :param start_date: Description
    :type start_date: str
    :param end_date: Description
    :type end_date: str
    :param model_type: Description
    :type model_type: str
    :param model_units: Description
    :type model_units: str
    """
    # get datatables.
    for k,v in mlr_pred_dict.items():
        print(k,v)
        if v[1] == model_type and v[2] == model_units:
            idx_mm = k

    for k,v in mlr_pred_dict.items():
        if v[1] == model_type and v[2] == 'pillows':
            idx_pillows = k
    print(idx_mm,idx_pillows)
    
    if start_date is None:
        start_date = str(mlr_pred_lst[0]['Date'].min().values)[:10]
    if end_date is None:
        end_date = str(mlr_pred_lst[0]['Date'].max().values)[:10]
    
    # slice predictions and pillows.
    df_pred_slice = mlr_pred_lst[idx_mm][(mlr_pred_lst[idx_mm]['Training Infer NaNs'] == 'predict NaNs') &  \
              (mlr_pred_lst[idx_mm]['Date'] >= np.datetime64(start_date)) & \
              (mlr_pred_lst[idx_mm]['Date'] <= np.datetime64(end_date))]

    df_pil_slice = mlr_pred_lst[idx_pillows][(mlr_pred_lst[idx_pillows]['Training Infer NaNs'] == 'predict NaNs') &  \
              (mlr_pred_lst[idx_pillows]['Date'] >= np.datetime64(start_date)) & \
              (mlr_pred_lst[idx_pillows]['Date'] <= np.datetime64(end_date)) \
              ][['Date','Basin']]
    
    # create unique pillows and matrix.
    df_pil_explode, df_pil_mat, unique_pils = pillow_explode(df_pil_slice)
    # slice raw observations.
    raw_df = obs_raw.where(obs_raw.time >= np.datetime64(start_date),drop = True) \
                    .where(obs_raw.time <= np.datetime64(end_date),drop = True) \
                    .where(obs_raw.time.isin(df_pil_slice.Date.values),drop = True) \
                   [unique_pils].to_dataframe()
    # slice clean observations.
    test_df = obs_clean.where(obs_clean.time >= np.datetime64(start_date),drop = True) \
                    .where(obs_clean.time <= np.datetime64(end_date),drop = True) \
                    .where(obs_clean.time.isin(df_pil_slice.Date.values),drop = True) \
                   [unique_pils].to_dataframe()
    # diff mask.
    diff_mask = ~(
        (raw_df == test_df) |
        (raw_df.isna() & test_df.isna())
    )
    # diff array.
    diff_arr = mask_to_matrix(diff_mask)
    # nan mask.
    nan_mask = ~raw_df.isna()
    # nan array.
    nan_arr = mask_to_matrix(nan_mask)

    # plotting.
    fig,ax = plt.subplots(5,1,figsize=(10,10),sharex=False)
    mlr_timeseries_plot(mlr_pred_lst,mlr_pred_dict,uaswe_m_df,snodas_m_df,start_date,end_date,model_type,idx_mm,ax[0])
    
    for pil in unique_pils:
        raw_df[pil].plot(ax=ax[1],label = pil)
    ax[1].legend()
    ax[1].set_title('Pillow Observations - raw',fontweight = 'bold')

    for pil in unique_pils:
        test_df[pil].plot(ax=ax[2],label = pil)
    ax[2].legend()
    ax[2].set_title('Pillow Observations - clean',fontweight = 'bold')

    im = ax[4].imshow(
        df_pil_mat.values,
        aspect="auto",
        interpolation="nearest",
        cmap = 'binary',
    )
    ax[4].set_title('Pillow Selection (black = selected)',fontweight = 'bold')
    # im = ax[4].imshow(
    #     # nan_arr[::-1,:],
    #     nan_arr,
    #     aspect="auto",
    #     interpolation="nearest",
    #     cmap = 'binary'
    # )
    # ax[4].set_title('Non-NaN Observations (black = observed)',fontweight = 'bold')
    im = ax[3].imshow(
        # diff_arr[::-1,:],
        diff_arr,
        aspect="auto",
        interpolation="nearest",
        cmap = 'binary'
    )
    ax[3].set_title('QA (black = Flagged)',fontweight = 'bold')
    for i in range(0,5):
        if i == 4:
            ax[i].set_yticks(range(len(df_pil_mat.index)))
            ax[i].set_yticklabels(df_pil_mat.index)
            ax[i].set_xticks(range(len(df_pil_mat.columns)))
            ax[i].set_xticklabels(
                [d.strftime("%Y-%m-%d") for d in df_pil_mat.columns],
                rotation=45,
                ha="right"
            )
            ax[i].set_xlabel('Date',fontweight = 'bold')
        elif i <=2:
            ax[i].set_ylabel('SWE [mm]',fontweight = 'bold')
            ax[i].set_xticks([])
            ax[i].set_xlabel('')
        else:
            ax[i].set_xticks([])
            ax[i].set_xlabel('')
            ax[i].set_yticks(range(len(df_pil_mat.index)))
            ax[i].set_yticklabels(df_pil_mat.index)
    plt.tight_layout()
    plt.show()

    return df_pred_slice,df_pil_slice,df_pil_explode,df_pil_mat,diff_arr,nan_arr,diff_mask

def mask_to_matrix(df_mask):
    df = df_mask.copy()

    # Ensure datetime + sorted
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()

    # Convert to numeric for plotting:
    #   True -> 1, False -> 0, NaN -> np.nan (so it can be a separate color if you want)
    arr = df.astype("float").to_numpy().T  
    return arr

def pillow_explode(df_pil_slice):
    df = df_pil_slice.copy()
    df["Date"] = pd.to_datetime(df["Date"])

    # Split basin strings into lists
    df["Pillow"] = df["Basin"].str.split(r"\s*,\s*")

    # Explode to long format
    df_long = df.explode("Pillow")[["Date", "Pillow"]]
    # Create binary presence matrix
    mat = (
        df_long
        .assign(value=1)
        .pivot(index="Pillow", columns="Date", values="value")
        .fillna(0)
        .sort_index()
    )
    unique_pils = df_long["Pillow"].unique().tolist()
    return df_long,mat,unique_pils

def slice_timeseries_df(start_date: str,
                     end_date: str,
                     df: pd.DataFrame):
    """
    """
    if start_date is not None:
        df = df[df['Date'] >= np.datetime64(start_date)]
    if end_date is not None:
        df = df[df['Date'] <= np.datetime64(end_date)]
    return df

def slice_timeseries_xr(start_date: str,
                     end_date: str,
                     ds: xr.Dataset):
    """
    """
    if start_date is not None:
        ds = ds.sel(Date=slice(np.datetime64(start_date), None))
    if end_date is not None:
        ds = ds.sel(Date=slice(None, np.datetime64(end_date)))
    return ds
def mlr_timeseries_plot(
        mlr_tables: List,
        mlr_identifiers: Dict,
        uaswe_m_df: pd.DataFrame,
        snodas_m_df: pd.DataFrame,
        start_date: str,
        end_date: str,
        model_type: str,
        swe_index: int,
        ax):
    """
    """
    ax.plot(uaswe_m_df['Date'],uaswe_m_df['total']*1000,label='UASWE',color='black',linewidth=2,linestyle = '--')
    ax.plot(snodas_m_df['Date'],snodas_m_df['total']*1000,label='SNODAS',color='gray',linewidth=2,linestyle = '--')
    if model_type == 'season':
        mlr_tables[swe_index][mlr_tables[swe_index]['Training Infer NaNs'] == 'predict NaNs'].plot(ax=ax,x='Date',y='Basin',color = 'C0',linewidth=2,label = 'MLR Prediction')
    elif model_type == 'accum':
        mlr_tables[swe_index][mlr_tables[swe_index]['Training Infer NaNs'] == 'predict NaNs'].plot(ax=ax,x='Date',y='Basin',color = 'C1',linewidth=2,label = 'MLR Prediction')
    else:
        mlr_tables[swe_index][mlr_tables[swe_index]['Training Infer NaNs'] == 'predict NaNs'].plot(ax=ax,x='Date',y='Basin',color = 'C2',linewidth=2,label = 'MLR Prediction')

    ax.legend()
    ax.set_title(f'Model Predictions - {model_type}',fontweight = 'bold')
    ax.set_xlim(np.datetime64(start_date),np.datetime64(end_date))
    return
    

def visualize_pillow_selection_heatmap(
    mlr_tables: list,
    mlr_identifiers: dict,
    start_date: str = None,
    end_date: str = None,
    train_infer: str = 'predict NaNs',
    figsize: tuple = (16, 8),
    ax=None,
):
    """
    Create a unified heatmap showing which pillows were selected by which models across time.
    
    Parameters:
    -----------
    mlr_tables : list
        List of MLR prediction dataframes
    mlr_identifiers : dict
        Dictionary mapping indices to [aso_stack_type, seasonal_dir, data_type]
    start_date : str, optional
        Start date for visualization (format: 'YYYY-MM-DD')
    end_date : str, optional
        End date for visualization (format: 'YYYY-MM-DD')
    train_infer : str
        Filter for 'Training Infer NaNs' column
    figsize : tuple
        Figure size (width, height)
    ax : matplotlib.axes.Axes, optional
        Axes object to draw the plot onto, otherwise a new figure and axes are created.
    
    Returns:
    --------
    fig, ax : matplotlib figure and axes objects
    """

    
    # Find indices for each model type (pillows only)
    model_indices = {}
    for idx, (stack_type, seasonal_dir, data_type) in mlr_identifiers.items():
        if seasonal_dir in ['season', 'accum', 'melt'] and data_type == 'pillows':
            model_indices[seasonal_dir] = idx
    
    model_order = ['season', 'accum', 'melt']
    
    # Collect all pillow selections for each model
    all_selections = {}
    all_dates = set()
    all_pillows = set()
    
    for model_type in model_order:
        if model_type not in model_indices:
            continue
        
        pil_idx = model_indices[model_type]
        pil_df = mlr_tables[pil_idx][mlr_tables[pil_idx]['Training Infer NaNs'] == train_infer].copy()
        
        if start_date:
            pil_df = pil_df[pil_df['Date'] >= np.datetime64(start_date)]
        if end_date:
            pil_df = pil_df[pil_df['Date'] <= np.datetime64(end_date)]
        
        # Parse pillow selections
        df_pil = pil_df[['Date', 'Basin']].copy()
        df_pil["Date"] = pd.to_datetime(df_pil["Date"])
        df_pil["Pillow"] = df_pil["Basin"].str.split(r"\s*,\s*")
        
        # Explode to long format
        df_long = df_pil.explode("Pillow")[["Date", "Pillow"]]
        
        if len(df_long) > 0:
            all_selections[model_type] = df_long
            all_dates.update(df_long['Date'].unique())
            all_pillows.update(df_long['Pillow'].unique())
    
    # Create sorted lists
    sorted_dates = sorted(list(all_dates))
    sorted_pillows = sorted(list(all_pillows))
    
    # Create a 2D matrix: pillows x dates
    # Values: 0=none, 1=season, 2=accum, 3=melt, 4=season+accum, 5=season+melt, 6=accum+melt, 7=all three
    n_pillows = len(sorted_pillows)
    n_dates = len(sorted_dates)
    
    # Model encoding
    model_encoding = {'season': 1, 'accum': 2, 'melt': 4}
    
    # Initialize matrix
    selection_matrix = np.zeros((n_pillows, n_dates), dtype=int)
    
    # Fill matrix with bitwise OR for overlapping selections
    for model_type, df_long in all_selections.items():
        model_val = model_encoding[model_type]
        for _, row in df_long.iterrows():
            pil_idx = sorted_pillows.index(row['Pillow'])
            date_idx = sorted_dates.index(row['Date'])
            selection_matrix[pil_idx, date_idx] |= model_val
    
    # Define colors for each model
    color_season = 'C0'  # blue
    color_accum = 'C1'   # orange
    color_melt = 'C2'    # green
    color_white = 'white'
    
    # Create figure
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
    
    # Draw rectangles - divide each cell into thirds (vertically stacked)
    # Top third = season, middle third = accum, bottom third = melt
    
    for i in range(n_pillows):
        for j in range(n_dates):
            val = selection_matrix[i, j]
            if val == 0:
                continue  # Skip empty cells
            
            # Determine which models selected this pillow on this date
            # Season=1, Accum=2, Melt=4
            has_season = (val & 1) > 0
            has_accum = (val & 2) > 0
            has_melt = (val & 4) > 0
            
            # Draw top third (season)
            rect_season = Rectangle((j - 0.5, i - 0.5 + 2/3), 1, 1/3, 
                           facecolor=color_season if has_season else color_white, 
                           edgecolor='gray',
                           linewidth=0.5,
                           linestyle='--')
            ax.add_patch(rect_season)
            
            # Draw middle third (accum)
            rect_accum = Rectangle((j - 0.5, i - 0.5 + 1/3), 1, 1/3, 
                           facecolor=color_accum if has_accum else color_white, 
                           edgecolor='gray',
                           linewidth=0.5,
                           linestyle='--')
            ax.add_patch(rect_accum)
            
            # Draw bottom third (melt)
            rect_melt = Rectangle((j - 0.5, i - 0.5), 1, 1/3, 
                           facecolor=color_melt if has_melt else color_white, 
                           edgecolor='gray',
                           linewidth=0.5,
                           linestyle='--')
            ax.add_patch(rect_melt)
    
    # Add gridlines
    # Horizontal gridlines (black) separating pillows
    for i in range(n_pillows + 1):
        ax.axhline(y=i - 0.5, color='black', linewidth=1.5, zorder=10)
    
    # Vertical gridlines (gray) separating days
    for j in range(n_dates + 1):
        ax.axvline(x=j - 0.5, color='gray', linewidth=0.5, zorder=10)
    
    # Set axis limits and appearance
    ax.set_xlim(-0.5, n_dates - 0.5)
    ax.set_ylim(-0.5, n_pillows - 0.5)
    ax.invert_yaxis()  # Invert y-axis so first pillow is at top
    
    # Set y-axis (pillows)
    ax.set_yticks(range(n_pillows))
    ax.set_yticklabels(sorted_pillows, fontsize=10)
    ax.set_ylabel('Pillow', fontweight='bold', fontsize=12)
    
    # Set x-axis (dates)
    step = max(1, n_dates // 20)  # Show ~20 tick labels
    xticks = list(range(0, n_dates, step))
    ax.set_xticks(xticks)
    date_labels = [pd.Timestamp(sorted_dates[i]).strftime("%m-%d") for i in xticks]
    ax.set_xticklabels(date_labels, rotation=45, ha='right', fontsize=9)
    ax.set_xlabel('Date', fontweight='bold', fontsize=12)
    
    # Create legend with color patches
    legend_elements = [
        mpatches.Patch(facecolor=color_season, edgecolor='gray', linestyle='--', label='season'),
        mpatches.Patch(facecolor=color_accum, edgecolor='gray', linestyle='--', label='accum'),
        mpatches.Patch(facecolor=color_melt, edgecolor='gray', linestyle='--', label='melt'),
    ]
    ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.02, 1), 
              fontsize=10, frameon=True)
    
    ax.set_title('Pillow Selection', fontweight='bold', fontsize=14, pad=10)
    
    # plt.tight_layout()
    
    return fig, ax

def html_timeseries_plot(
    mlr_lst: List,
    mlr_ids: Dict,
    uaswe_df: pd.DataFrame,
    uaswe_provisional_df: pd.DataFrame,
    snodas_df: pd.DataFrame,
    sm_df: pd.DataFrame,
    basin: str,
    current_dates: np.ndarray,
    current_swe: np.ndarray,
    uaswe_snowtrax_df: pd.DataFrame = None,
    plotDir: str = None,
    model_name: str = None,
    train_infer: str = 'predict NaNs',
    saveFIG: bool = True,
    ax=None,
    start_date: str = '2026-01-01', 
    end_date: str = None,
):
    if ax is None:
        fig, ax = plt.subplots(dpi=200, sharex=True, figsize=(10, 6))
        own_fig = True
    else:
        fig = ax.get_figure()
        own_fig = False

    # --- BASE MODEL PREDICTIONS ---
    l1, = ax.plot(
        uaswe_df['Date'], uaswe_df['total'] / 1000,
        label='UASWE Early', color='black', linestyle='--'
    )
    l2, = ax.plot(
        uaswe_provisional_df['Date'], uaswe_provisional_df['total'] / 1000,
        label='UASWE Provisional', color='black', linestyle=':'
    )
    if uaswe_snowtrax_df is not None:
        l2_, = ax.plot(
            uaswe_snowtrax_df['DATE'], uaswe_snowtrax_df['SWANN_UA_SWE_AF'] / 1000,
            label='UASWE SnowTrax', color='black', linestyle='-.'
        )
    l3, = ax.plot(
        snodas_df['Date'], snodas_df['total'] / 1000,
        label='SNODAS', color='gray', linestyle='--'
    )
    l4, = ax.plot(
        sm_df['Date'], sm_df['total'] / 1000,
        label='SnowModel', color='brown', linestyle='--'
    )
    
    # Current SWE scatter plot
    l5 = ax.scatter(
        current_dates, current_swe / 1000,
        label='ASO', color='purple', s=200, marker = u'$\u2744$'
    )

    if uaswe_snowtrax_df is not None:
        model_handles = [l1, l2, l2_, l3, l4, l5]
    else:
        model_handles = [l1, l2, l3, l4, l5]

    # --- MLR PREDICTIONS ---
    mlr_handles = []
    mlr_labels = []

    for i in range(len(mlr_lst)):
        mlr_df = mlr_lst[i]
        mlr_id = mlr_ids[i]

        if mlr_id[2] == 'acreFt':
            line = mlr_df[
                mlr_df['Training Infer NaNs'] == train_infer
            ].plot(
                x='Date', y='Basin', ax=ax,
                linestyle='-', legend=False
            )

            h = ax.lines[-1]  # grab the line just added
            mlr_handles.append(h)
            mlr_labels.append(f"MLR {mlr_id[1]}")

    # --- FIRST LEGEND (MLR models — TOP) ---
    # Adjust legend position based on whether we own the figure
    if own_fig:
        # Place legends outside the axes (original behavior)
        leg1 = ax.legend(
            handles=mlr_handles,
            labels=mlr_labels,
            title='MLR Predictions',
            loc='upper left',
            bbox_to_anchor=(1.02, 1.0),
            title_fontproperties={'weight': 'bold'}
        )
        ax.add_artist(leg1)

        # --- SECOND LEGEND (Base models — BOTTOM) ---
        ax.legend(
            handles=model_handles,
            title='Model Predictions',
            loc='lower left',
            bbox_to_anchor=(1.02, 0.0),
            title_fontproperties={'weight': 'bold'}
        )
    else:
        # Place legends inside the axes to avoid overlap with other subplots
        leg1 = ax.legend(
            handles=mlr_handles,
            labels=mlr_labels,
            title='MLR Predictions',
            loc='upper left',
            fontsize=8,
            title_fontproperties={'weight': 'bold', 'size': 9}
        )
        ax.add_artist(leg1)

        # --- SECOND LEGEND (Base models — BOTTOM) ---
        ax.legend(
            handles=model_handles,
            title='Model Predictions',
            loc='lower left',
            fontsize=8,
            title_fontproperties={'weight': 'bold', 'size': 9}
        )

    ax.set_ylabel('Mean SWE [thousand acre-feet]', fontweight='bold')
    ax.set_xlabel('Date', fontweight='bold')
    ax.set_title(f'Basin Mean SWE Prediction', fontweight='bold', fontsize=14, pad=10)
    ax.set_xlim(np.datetime64(start_date) if start_date else None, np.datetime64(end_date) if end_date else None)

    # Make room for legends on the right (only if we created the figure)
    if own_fig:
        fig.subplots_adjust(right=0.75)

    # if saveFIG:
    #     print(plotDir)
    #     plt.savefig(f'{plotDir}/{basin}_{model_name}_timeseries_plot.png', dpi=200, bbox_inches='tight')
    # else:
    #     if own_fig:
    #         plt.show()

    # if own_fig:
    #     plt.show()
    #     plt.close(fig)
    return



# def sheet_for_heatmap(sheet,end_date,area_dict):
#     plot_cols = sheet.columns.to_list()
#     plot_cols.remove('Model Type')
#     plot_cols.remove('Training Infer NaNs')
#     plot_cols.remove('Prediction QA')
#     # clean table.
#     new_sheet = sheet[plot_cols]
#     new_sheet = new_sheet.set_index('Date')
#     new_sheet = new_sheet[['Total','>12k','11k-12k','10k-11k','9k-10k','8k-9k','7k-8k','<7k']]
#     new_sheet = new_sheet[new_sheet.index >= np.datetime64(end_date + timedelta(days=-14))]
#     for col in new_sheet.columns:
#         new_sheet[col] = new_sheet[col] / area_dict[col] * 12
#     return new_sheet


# def prediction_by_elevation_heatmap2(sheet1,sheet2,sheet3,sheet4,sheet5,sheet6,end_date,area_dict,aso_site_name,
#                                      savePlot = True,showPlot = False):
#     """
#         Plot heatmap of last two week predictions by elevation.
#         Input:
#             sheet - geopandas dataframe for predictions.  
#         Output:
#             plot of basins.
#     """
#     # list of elevations.
#     new_sheet1 = sheet_for_heatmap(sheet1,end_date,area_dict)
#     new_sheet2 = sheet_for_heatmap(sheet2,end_date,area_dict)
#     new_sheet3 = sheet_for_heatmap(sheet3,end_date,area_dict)
#     new_sheet4 = sheet_for_heatmap(sheet4,end_date,area_dict)
#     new_sheet5 = sheet_for_heatmap(sheet5,end_date,area_dict)
#     new_sheet6 = sheet_for_heatmap(sheet6,end_date,area_dict)
#     all_sheets = [new_sheet1,new_sheet2,new_sheet3,new_sheet4,new_sheet5,new_sheet6]

#     # plotting.
#     all_max = 0.0
#     all_min = 1000.0
#     for i in range(0,len(all_sheets)):
#         max_ = all_sheets[i].max().max()
#         min_ = all_sheets[i].min().min()
#         if max_ > all_max:
#             all_max = max_
#         if min_ < all_min:
#             all_min = min_

#     fig,ax = plt.subplots(3,2,dpi = 250,sharex = True,sharey = True,constrained_layout = True)
#     for i in range(0,len(all_sheets)):
#         im = ax[i//2,i%2].imshow(all_sheets[i].T,cmap = 'inferno',vmax = all_max,vmin = all_min)
#         if i%2 == 0:
#             ax[i//2,i%2].set_yticks(ticks = np.arange(0,len(all_sheets[i].columns)), labels = all_sheets[i].columns)
#         if i//2 == 2:
#             ax[i//2,i%2].set_xticks(ticks = np.arange(0,len(all_sheets[i]))[::2], labels = all_sheets[i].index.strftime('%m-%d')[::2],rotation = 45)
#     ax[0,0].set_title('Drop NaNs QA = 1',fontweight = 'bold')
#     ax[0,1].set_title('Predict NaNs QA = 1',fontweight = 'bold')
#     ax[1,0].set_title('Drop NaNs QA = 2',fontweight = 'bold')
#     ax[1,1].set_title('Predict NaNs QA = 2',fontweight = 'bold')
#     ax[2,0].set_title('Drop NaNs QA = 3',fontweight = 'bold')
#     ax[2,1].set_title('Predict NaNs QA = 3',fontweight = 'bold')
#     plt.suptitle('Model Predictions by Elevation',fontweight = 'bold',fontsize = 16)
#     fig.subplots_adjust(right=0.90)
#     cbar_ax = fig.add_axes([0.48, 0.10, 0.02, 0.8])
#     fig.colorbar(im, cax=cbar_ax,label = 'SWE [in]')
#     #plt.tight_layout()
#     if savePlot:
#         plt.savefig(f'./data/predictions/{aso_site_name}/historic/elevation_comparison.png',dpi = 250)
#     if showPlot:
#         plt.show()
#     return



# def create_prediction_plot(sheet1,sheet2,sheet3,sheet4,sheet5,sheet6,aso_site_name,
#                            end_date,area_dict,basinID = 'SBF',showPlot = True,savePlot = True):
#     """
#         Plot heatmap of last two week predictions by elevation.
#         Input:
#             sheet - geopandas dataframe for predictions.  
#         Output:
#             plot of basins.
#     """
#     # instantiate figure.
#     fig,ax = plt.subplots(1,2,figsize = (10,5),sharex = False,gridspec_kw = {'width_ratios': [4,1]})
#     # ax[0] = plot_predicted_comparisons2(sheet1,sheet2,sheet3,sheet4,sheet5,sheet6,aso_site_name,end_date,ax[0],
#     #                            basinID = 'SBF',showPlot = True,savePlot = True)
#     ax[0] = plot_predicted_comparisons2(sheet1,sheet2,sheet3,sheet4,sheet5,sheet6,aso_site_name,end_date,ax[0],
#                                basinID = 'SBF',showPlot = True,savePlot = True)
#     # ax[0] = prediction_by_elevation_heatmap2(sheet1,end_date,area_dict,ax[0])
#     ax[1] = prediction_by_elevation_heatmap2(sheet1,end_date,area_dict,ax[1])
#     plt.tight_layout()
#     plt.show()


# def plot_predicted_comparisons3(sheet1,sheet2,sheet3,sheet4,sheet5,sheet6,aso_site_name,end_date,ax_,
#                                basinID = 'SBF',showPlot = True,savePlot = True):
    

# def plot_predicted_comparisons2(sheet1,sheet2,sheet3,sheet4,sheet5,sheet6,aso_site_name,end_date,ax_,
#                                basinID = 'SBF',showPlot = True,savePlot = True):
#     """
#         Plot basin location with state lines.
#         Input:
#             shape_geog - geopandas shape of basin in geographic coordinates.
#             states_fpath - python string for relative filepath to US States 
#                            shapefile.
#             proj_crs - python string for projected crs epgs.
#             showPlot - boolean to produce plot.  
#         Output:
#             plot of basins.
#     """
#     # read in other model predictions.
#     data_other = pd.read_csv('https://snow.water.ca.gov/service/plotly/data/download?dash=fcast_resources&file=csv/wsfr_snow.csv')
#     data_other['DATE'] = pd.to_datetime(data_other['DATE'])
#     # slice to basin.
#     sj_other = data_other[data_other['STA_ID'] == basinID]
#     sj_all = copy.deepcopy(sj_other)
#     # slice dates.
#     sj_other = sj_other[(sj_other['DATE'] >= np.datetime64(end_date + timedelta(days=-14))) & (sj_other['DATE'] <= np.datetime64(sheet2.Date.iloc[-1]))]
#     # sj_other = sj_other[(sj_other['DATE'] >= np.datetime64('2024-11-15')) & (sj_other['DATE'] <= np.datetime64(sheet2.Date.iloc[-1]))]

#     # drop missing columns.
#     sj_other = sj_other.dropna(axis = 1,how = 'all')

#     # divide values by 1000.
#     col_names = sj_other.columns.to_list()
#     # remove columns names.
#     col_names.remove('DATE')
#     col_names.remove('STA_ID')
#     col_names.remove('STATION_NAME')
#     for col in ['ISNOBAL_DWR_SWE_AF','SNODAS_SWE_AF','SNOW17_SWE_AF','SWANN_UA_SWE_AF']:
#         sj_other[col] = sj_other[col]/1000
#     # create day of year index.
#     sj_other['day_of_year'] = sj_other['DATE'].dt.dayofyear
#     sj_other = sj_other.set_index('day_of_year')

#     # create median dataframe.
#     snodas_df = sj_all[['DATE','SNODAS_SWE_AF']]
#     snodas_df = snodas_df[snodas_df['DATE'] < np.datetime64(end_date + timedelta(days=-14))]
#     snodas_df['day_of_year'] = snodas_df['DATE'].dt.dayofyear
#     median_df = snodas_df.groupby('day_of_year')[['SNODAS_SWE_AF']].median() / 1000
#     median_df = median_df.rename(columns = {'SNODAS_SWE_AF':'SNODAS_MEDIAN'})
#     # merge median
#     sj_other = pd.merge(sj_other,median_df,how = 'left',left_on = 'day_of_year',right_on = 'day_of_year')
#     # clean up sheet data.
#     sheet1['Date'] = pd.to_datetime(sheet1['Date'])
#     sheet2['Date'] = pd.to_datetime(sheet2['Date'])
#     sheet3['Date'] = pd.to_datetime(sheet3['Date'])
#     sheet4['Date'] = pd.to_datetime(sheet4['Date'])
#     sheet5['Date'] = pd.to_datetime(sheet5['Date'])
#     sheet6['Date'] = pd.to_datetime(sheet6['Date'])

#     nan_df_1 = sheet1[['Date','Total','Prediction QA','Training Infer NaNs']]
#     nan_df_2 = sheet3[['Date','Total','Prediction QA','Training Infer NaNs']]
#     nan_df_3 = sheet5[['Date','Total','Prediction QA','Training Infer NaNs']]

#     pred_df_1 = sheet2[['Date','Total','Prediction QA','Training Infer NaNs']]
#     pred_df_2 = sheet4[['Date','Total','Prediction QA','Training Infer NaNs']]
#     pred_df_3 = sheet6[['Date','Total','Prediction QA','Training Infer NaNs']]

#     df_nan__ = pd.concat([nan_df_1,nan_df_2,nan_df_3])
#     df_pred__ = pd.concat([pred_df_1,pred_df_2,pred_df_3])
#     df_nan__['Total'] = df_nan__['Total']/1000
#     df_pred__['Total'] = df_pred__['Total']/1000
#     df_sns = pd.concat([df_nan__,df_pred__])
#     df_sns = df_sns.rename(columns = {'Training Infer NaNs':'MLR Prediction'})

#     df_sns = df_sns[(df_sns['Date'] >= np.datetime64(end_date + timedelta(days=-14)))]
#     # plotting.
#     # fig,ax = plt.subplots(dpi = 150)
#     g = sns.lineplot(ax=ax_,
#                  x = 'Date',
#                  y = 'Total',
#                  hue = 'MLR Prediction',
#                  style='MLR Prediction', 
#                  markers=['D', 'X'],
#                  data=df_sns,
#                  dashes = False,
#                  legend = True)
#     first_legend = ax_.legend(title = 'MLR Prediction',loc = 'upper left',title_fontproperties={'weight': 'bold'},)



#     l1 = ax_.plot(sj_other['DATE'],sj_other['ISNOBAL_DWR_SWE_AF'],color = 'C2',linestyle = '--',linewidth = 1)
#     l2 = ax_.plot(sj_other['DATE'],sj_other['SNODAS_SWE_AF'],color = 'C3',linestyle = '--',linewidth = 1)
#     l3 = ax_.plot(sj_other['DATE'],sj_other['SNOW17_SWE_AF'],color = 'C4',linestyle = '--',linewidth = 1)
#     l4 = ax_.plot(sj_other['DATE'],sj_other['SWANN_UA_SWE_AF'],color = 'C5',linestyle = '--',linewidth = 1)
#     l5 = ax_.plot(sj_other['DATE'],sj_other['SNODAS_MEDIAN'],color = 'black',linestyle = '-',linewidth = 2)

#     # Make sure to pass single Line2D objects rather than 1-element lists:
#     line1 = l1[0]
#     line2 = l2[0]
#     line3 = l3[0]
#     line4 = l4[0]
#     line5 = l5[0]

#     # Second legend
#     second_legend = ax_.legend(
#     handles=[line1, line2, line3, line4,line5],
#     labels=['ISNOBAL', 'SNODAS', 'SNOW17', 'SWANN','SNODAS_MEDIAN'],
#     loc='lower right',
#     title = 'Other Products',
#     title_fontproperties={'weight': 'bold'},  # makes the legend title bold
#     )

#     # Add *both* legends
#     ax_.add_artist(first_legend)
#     ax_.add_artist(second_legend)
#     ax_.set_ylabel('Basin Mean SWE [thousand acre-feet]',fontweight = 'bold')
#     # plt.xlabel('Date',fontweight = 'bold')
#     ax_.set_title('San Joaquin Total Basin Mean SWE Prediction',fontweight = 'bold')
#     ax_.set_xticks(rotation = 45)
#     # plt.ylim([0,1000])
#     plt.tight_layout()
#     if savePlot:
#         plt.savefig(f'./prediction_comparison.png',dpi = 150)
#     if showPlot:
#         plt.show()


#     return ax_

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
    manual_windows: list = None,
    saveFIG: bool = False,
    figDIR: str = None
):
    """
    manual_windows: optional list of (start, end) tuples for this pillow's
    YAML-specified exclusion ranges. Drawn as full-height translucent bars
    labelled 'Manual removed'. Independent of the automated QA flags.
    """

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


    # --- plotting ---
    fig, ax = plt.subplots(figsize=(13, 5))
    ax.plot(raw.index, raw.values, label="Raw SWE", linewidth=1.8)
    if qa is not None:
        ax.plot(qa.index, qa.values, label="QA SWE", linewidth=2.2, linestyle="--")

    # manual exclusion windows from YAML config (full-height vertical band per window).
    if manual_windows:
        for i, (ws, we) in enumerate(manual_windows):
            ax.axvspan(
                pd.Timestamp(ws), pd.Timestamp(we),
                color="gray", alpha=0.22, zorder=0,
                label="Manual removed" if i == 0 else "_nolegend_",
            )

    # y-scale helpers for the bands
    y0, y1 = ax.get_ylim()
    yr = y1 - y0
    band_h = 0.06 * yr
    gap = 0.01 * yr
    base = y0 + 0.02 * yr
    cur_base = base

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
        return {"raw": raw, "qa": qa, "flags": flags_plot, "majority": maj_true}
    else:
        return {"raw": raw, "flags": flags_plot, "majority": maj_true}


def plot_all_pillows_qa_timeline(
    ds_raw: "xr.Dataset",
    df_simple: pd.DataFrame,
    methods=("static", "voting", "snowmodel"),
    majority_k: int = 2,
    start=None,
    end=None,
    manual_windows_per_pillow: dict = None,
    pillows: list = None,
    ncols: int = 5,
    saveFIG: bool = False,
    figDIR: str = None,
    figFNAME: str = "all_pils.png",
    dpi: int = 300,
):
    """
    All-pillows grid version of plot_pillow_qa_timeline.

    One PNG with a subplot per pillow (sharex=True, sharey=True). Each subplot shows
    the raw SWE line, the three method flag bands, the majority band, and the
    YAML-specified manual-removal windows.

    A single shared legend is placed to the right of the grid.
    """
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D

    if pillows is None:
        pillows = list(ds_raw.data_vars)
    if manual_windows_per_pillow is None:
        manual_windows_per_pillow = {}

    # ---- prep df_simple once ----
    d = df_simple.copy()
    d["time"] = pd.to_datetime(d["time"]).dt.normalize()
    d["pillow"] = d["pillow"].astype(str).str.strip()
    d["method"] = d["method"].astype(str).str.strip()
    d["flag"] = d["flag"].fillna(0).astype(int)

    # ---- common time axis ----
    raw_times = pd.to_datetime(ds_raw.time.values)
    tmin = raw_times.min().normalize()
    tmax = raw_times.max().normalize()
    if start is not None:
        tmin = max(tmin, pd.to_datetime(start).normalize())
    if end is not None:
        tmax = min(tmax, pd.to_datetime(end).normalize())
    idx = pd.date_range(tmin, tmax, freq="D")

    # ---- precompute global y-range so the flag/majority bands sit consistently ----
    raw_series_by_pil = {}
    g_min, g_max = np.inf, -np.inf
    for pil in pillows:
        s = ds_raw[pil].to_series()
        s.index = pd.to_datetime(s.index).normalize()
        s = s.reindex(idx)
        raw_series_by_pil[pil] = s
        if s.notna().any():
            g_min = min(g_min, float(s.min()))
            g_max = max(g_max, float(s.max()))
    if not np.isfinite(g_min):
        g_min, g_max = 0.0, 1.0
    yr = max(g_max - g_min, 1.0)
    band_h = 0.04 * yr
    gap = 0.005 * yr
    base = g_min + 0.02 * yr  # stack of bands starts just above the y-min

    # ---- figure layout ----
    npils = len(pillows)
    nrows = int(np.ceil(npils / ncols))
    fig, axes = plt.subplots(
        nrows=nrows, ncols=ncols,
        figsize=(3.6 * ncols, 2.4 * nrows),
        sharex=True, sharey=True,
        squeeze=False,
    )

    method_colors = {"static": "C1", "voting": "C2", "snowmodel": "C3"}

    for i, pil in enumerate(pillows):
        ax = axes.flat[i]
        raw = raw_series_by_pil[pil]

        # raw line
        ax.plot(raw.index, raw.values, color="C0", linewidth=1.3, label="_nolegend_")

        # manual removal windows (full-height vertical bands)
        for ws, we in manual_windows_per_pillow.get(pil, []) or []:
            ax.axvspan(
                pd.Timestamp(ws), pd.Timestamp(we),
                color="gray", alpha=0.22, zorder=0, label="_nolegend_",
            )

        # per-pillow flags
        d_pil = d[(d["pillow"] == pil) & (d["method"].isin(methods))]
        d_pil = d_pil[(d_pil["time"] >= tmin) & (d_pil["time"] <= tmax)]
        flags = d_pil.pivot_table(index="time", columns="method", values="flag", aggfunc="max")
        flags = flags.reindex(columns=list(methods)).fillna(0).astype(int)
        flags = flags.reindex(idx).fillna(0).astype(int)

        # widen single-day flags so they read at this small subplot scale
        flags_plot = flags.copy()
        for c in flags_plot.columns:
            s = flags_plot[c]
            single = (s == 1) & (s.shift(1, fill_value=0) == 0) & (s.shift(-1, fill_value=0) == 0)
            flags_plot.loc[single.shift(1, fill_value=False), c] = 1
            flags_plot.loc[single.shift(-1, fill_value=False), c] = 1

        maj_plot = (flags_plot.sum(axis=1) >= majority_k).astype(int)

        # stack: method bands, then majority on top
        cur_base = base
        for m in methods:
            ax.fill_between(
                idx, cur_base, cur_base + band_h,
                where=flags_plot[m].astype(bool).values,
                alpha=0.30, color=method_colors[m],
                step="post", label="_nolegend_",
            )
            cur_base += band_h + gap

        ax.fill_between(
            idx, cur_base, cur_base + band_h * 1.2,
            where=maj_plot.astype(bool).values,
            alpha=0.40, color="C4",
            step="post", label="_nolegend_",
        )

        ax.set_title(pil, fontweight="bold", fontsize=10)
        ax.grid(True, alpha=0.2)
        ax.tick_params(axis="x", labelrotation=30, labelsize=8)
        ax.tick_params(axis="y", labelsize=8)

    # hide unused axes
    for j in range(npils, nrows * ncols):
        axes.flat[j].set_visible(False)

    fig.supxlabel("Date", fontweight="bold")
    fig.supylabel("SWE (mm)", fontweight="bold", x=0.005)

    # shared legend to the right of all subplots
    legend_elements = [
        Line2D([0], [0], color="C0", lw=1.5, label="Raw SWE"),
        Patch(facecolor=method_colors["static"], alpha=0.30, label="static flag"),
        Patch(facecolor=method_colors["voting"], alpha=0.30, label="voting flag"),
        Patch(facecolor=method_colors["snowmodel"], alpha=0.30, label="snowmodel flag"),
        Patch(facecolor="C4", alpha=0.40, label=f"Majority (≥{majority_k}/{len(methods)})"),
        Patch(facecolor="gray", alpha=0.22, label="Manual removed"),
    ]
    fig.legend(
        handles=legend_elements,
        loc="center left",
        bbox_to_anchor=(0.92, 0.5),
        frameon=True,
        title="QA Flags",
        title_fontproperties={"weight": "bold"},
    )

    plt.tight_layout(rect=[0.03, 0.02, 0.91, 1])

    if saveFIG:
        if figDIR is None:
            figDIR = "./qa_viz_2026/"
        if not os.path.exists(figDIR):
            os.makedirs(figDIR)
        plt.savefig(os.path.join(figDIR, figFNAME), dpi=dpi, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def plot_swe_volume_by_elevation(
    aso_site_name: str,
    comp_date,
    aso_mean_swe_df: pd.DataFrame,
    uaswe_df: pd.DataFrame,
    snodas_df: pd.DataFrame,
    sm_df: "pd.DataFrame|None",
    prediction_acreFt_df: pd.DataFrame,
    aso_stack_type: str,
    seasonal_dir: str,
    area_ref_dict: dict,
    figDIR: str = None,
    saveFIG: bool = True,
    dpi: int = 150,
):
    """
    Single 3-panel SWE-volume-by-elevation comparison for one (model, seasonality, ASO date).

    Panels:
      - line plot: SWE Volume vs. elevation bin (ASO black; MLR / UASWE / SNODAS [/ SnowModel] dashed)
      - bar chart: basin total SWE volume per source
      - table: Basin SWE Error [%] and RMSE across elevation bins [TAF] for each model vs. ASO

    Saves to {figDIR}/{basin}_{model}_{seasonality}_{YYYY-MM-DD}.png.
    Returns None if there's no ASO data for the given date.
    """
    from matplotlib.gridspec import GridSpec
    from matplotlib.patches import Patch

    elev_bins = ['<7k', '7k-8k', '8k-9k', '9k-10k', '10k-11k', '11k-12k', '>12k']

    def _to_canonical(df):
        rename = {c: area_ref_dict[c] for c in df.columns if c in area_ref_dict}
        if 'Basin' in df.columns:
            rename['Basin'] = 'Total'
        return df.rename(columns=rename)

    # normalize date for comparisons across heterogeneous source DFs
    comp_ts = pd.Timestamp(comp_date).normalize()

    # ASO is long-form; rebuild aso_mean_swe_df with datetime index for matching.
    aso_local = aso_mean_swe_df.copy()
    aso_local['Date'] = pd.to_datetime(aso_local['Date']).dt.normalize()
    aso_row = aso_local[aso_local['Date'] == comp_ts]
    if aso_row.empty:
        print(f'  [plot_swe_volume_by_elevation] no ASO row for {comp_ts.date()}; skipping')
        return None

    aso_wide = aso_row[['elev', 'mean_swe']].set_index('elev').T
    aso_wide = _to_canonical(aso_wide) / 1000.0  # acre-Ft -> TAF
    try:
        aso_bin_vals = aso_wide[elev_bins].values.flatten()
        aso_total = float(aso_wide['Total'].values[0])
    except KeyError as e:
        print(f'  [plot_swe_volume_by_elevation] ASO missing expected columns ({e}); skipping')
        return None

    # slice a wide-format source DF to a one-row DataFrame for comp_ts, canonicalize columns.
    # `scale` divides numeric columns (e.g., 1000.0 for acre-Ft -> TAF).
    # `infer_nans_filter` collapses the MLR case where there are multiple rows per date.
    def _slice_wide(df, scale=1.0, infer_nans_filter=None):
        if df is None or len(df) == 0:
            return None
        d = df.copy()
        d['Date'] = pd.to_datetime(d['Date']).dt.normalize()
        sub = d[d['Date'] == comp_ts]
        if sub.empty:
            return None
        if infer_nans_filter is not None and 'Training Infer NaNs' in sub.columns:
            sub = sub[sub['Training Infer NaNs'] == infer_nans_filter]
            if sub.empty:
                return None
        sub = sub.set_index('Date')
        sub = sub.select_dtypes(include=[np.number]) / scale
        return _to_canonical(sub)

    # UASWE / SNODAS / SnowModel CSVs are in acre-Ft (despite their filename suffix); /1000 -> TAF.
    uaswe_d = _slice_wide(uaswe_df,  scale=1000.0)
    snodas_d = _slice_wide(snodas_df, scale=1000.0)
    sm_d = _slice_wide(sm_df, scale=1000.0) if sm_df is not None else None
    # MLR prediction CSVs are stored in TAF directly (filename suffix is misleading); no scaling.
    mlr_d = _slice_wide(prediction_acreFt_df, scale=1.0, infer_nans_filter='predict NaNs')

    def _bins(df):
        return df[elev_bins].values.flatten()
    def _total(df):
        return float(df['Total'].values[0])

    # assemble model list (skip any that have no data for this date).
    # NOTE: SnowModel currently excluded — its basin total runs much higher than ASO and the
    # bias correction isn't reliable enough to show alongside the other models. Re-add by
    # uncommenting the SnowModel line below.
    candidates = [
        ('MLR',       'tab:blue',   mlr_d),
        ('SNODAS',    'tab:purple', snodas_d),
        ('UASWE',     'tab:brown',  uaswe_d),
        # ('SnowModel', 'tab:green',  sm_d),
    ]
    models = []
    for name, color, df_d in candidates:
        if df_d is None or len(df_d) == 0:
            continue
        try:
            bv = _bins(df_d)
            tot = _total(df_d)
        except KeyError:
            continue
        models.append((name, color, bv, tot))

    if not models:
        print(f'  [plot_swe_volume_by_elevation] no model data for {comp_ts.date()}; skipping')
        return None

    # KPIs (basin error %, elevation-bin RMSE)
    kpi_rows = []
    for name, color, bin_vals, total in models:
        acc_pct = abs(total - aso_total) / aso_total * 100.0 if aso_total else float('nan')
        rmse = float(np.sqrt(np.mean((bin_vals - aso_bin_vals) ** 2)))
        kpi_rows.append((name, color, acc_pct, rmse))

    # title pieces
    basin_label = {
        'USCASJ': 'San Joaquin',
        'USCATM': 'Tuolumne',
        'USCOBR': 'Blue River',
        'USCOGE': 'Gunnison',
    }.get(aso_site_name, aso_site_name)
    day = comp_ts.day
    suffix = 'th' if 10 <= day % 100 <= 20 else {1: 'st', 2: 'nd', 3: 'rd'}.get(day % 10, 'th')
    title_date = comp_ts.strftime(f'%B {day}{suffix}, %Y')

    fig = plt.figure(figsize=(14, 8), dpi=dpi)
    gs = GridSpec(2, 3, figure=fig, height_ratios=[1.0, 0.4], hspace=0.18, wspace=0.32)
    ax_line = fig.add_subplot(gs[0, :2])
    ax_bar = fig.add_subplot(gs[0, 2])
    ax_tbl = fig.add_subplot(gs[1, :])
    fig.suptitle(f'{basin_label} Model Comparison\n{title_date}', fontsize=18, fontweight='bold', y=0.98)

    # SWE vs. Elevation
    aso_line, = ax_line.plot(elev_bins, aso_bin_vals, color='black', lw=2.5, marker='o', label='ASO')
    model_lines = []
    for name, color, bin_vals, _ in models:
        ln, = ax_line.plot(elev_bins, bin_vals, color=color, linestyle='--', label=name)
        model_lines.append(ln)
    ax_line.set_title('SWE vs. Elevation', fontweight='bold')
    ax_line.set_xlabel('Elevation Bins [ft]', fontweight='bold')
    ax_line.set_ylabel('SWE Volume [TAF]', fontweight='bold')
    leg1 = ax_line.legend(handles=[aso_line], title='Reference', loc='upper left',
                          title_fontproperties={'weight': 'bold'})
    ax_line.add_artist(leg1)
    ax_line.legend(handles=model_lines, title='Model Output', loc='upper right',
                   title_fontproperties={'weight': 'bold'})

    # Basin SWE Volume bars
    bar_labels = [m[0] for m in models] + ['ASO']
    bar_colors = [m[1] for m in models] + ['black']
    bar_values = [m[3] for m in models] + [aso_total]
    ax_bar.bar(range(len(bar_labels)), bar_values, color=bar_colors)
    ax_bar.set_xticks([])
    # add ~35% headroom on the y-axis so the top-left & bottom-right legends don't
    # overlap the tallest bar.
    y_top = max(bar_values + [1.0])
    ax_bar.set_ylim(0, y_top * 1.35)
    ax_bar.set_title('Basin SWE Volume', fontweight='bold')
    ax_bar.set_ylabel('SWE Volume [TAF]', fontweight='bold')
    model_handles = [Patch(facecolor=m[1], label=m[0]) for m in models]
    leg_models = ax_bar.legend(handles=model_handles, title='Model Output', loc='upper left',
                               title_fontproperties={'weight': 'bold'})
    ax_bar.add_artist(leg_models)
    ax_bar.legend(handles=[Patch(facecolor='black', label='ASO')],
                  title='Reference', loc='lower right',
                  title_fontproperties={'weight': 'bold'})

    # KPI table
    ax_tbl.set_axis_off()
    table_data = [[name, f'{acc:.1f}', f'{rmse:.0f}'] for name, _, acc, rmse in kpi_rows]
    tbl = ax_tbl.table(
        cellText=table_data,
        colLabels=['Model', 'Basin SWE Error [%]', 'RMSE Across Elevation Bins [TAF]'],
        colWidths=[0.20, 0.35, 0.45],
        cellLoc='center',
        loc='upper center',
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(11)
    tbl.scale(1, 1.6)
    for col_idx in range(3):
        header = tbl[(0, col_idx)]
        header.set_facecolor('#888888')
        header.set_text_props(color='white', weight='bold', fontsize=10)
    for row_idx, (_, color, _, _) in enumerate(kpi_rows, start=1):
        cell = tbl[(row_idx, 0)]
        cell.set_facecolor(color)
        cell.set_text_props(color='white', weight='bold')

    if saveFIG:
        if figDIR is None:
            figDIR = './swe_volume/'
        if not os.path.exists(figDIR):
            os.makedirs(figDIR)
        date_str = comp_ts.strftime('%Y-%m-%d')
        fname = f'{aso_site_name}_{aso_stack_type}_{seasonal_dir}_{date_str}.png'
        plt.savefig(os.path.join(figDIR, fname), dpi=dpi, bbox_inches='tight')
        plt.close()
        return os.path.join(figDIR, fname)
    else:
        plt.show()
        return None


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
    manual_windows: list = None,
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

    # method-specific signal isolated from shared data-integrity checks
    # (MISSING_TODAY, PHYS_*, NAN_*, DSWE_* are computed identically by all 3 methods —
    # plotting `final_flag` here would conflate them with the actual method-specific signal).
    METHOD_SPECIFIC_REASONS = {
        "static":    "PEER_RESID_TOO_LARGE",
        "voting":    "VOTING_INCONSISTENT",
        "snowmodel": "PRED_RESID_TOO_LARGE",
    }
    specific_reason = METHOD_SPECIFIC_REASONS.get(qa_method)
    method_flag = (
        (reason_code == specific_reason).astype(int)
        if specific_reason is not None else final_flag
    )

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

    # manual exclusion windows from YAML config: full-height bars on each panel for visual continuity.
    if manual_windows:
        for ax_i in (ax0, ax1, axf):
            for j, (ws, we) in enumerate(manual_windows):
                ax_i.axvspan(
                    pd.Timestamp(ws), pd.Timestamp(we),
                    color="gray", alpha=0.22, zorder=0,
                    label="Manual removed" if (ax_i is ax0 and j == 0) else "_nolegend_",
                )

    # --- panel 0: target series ---
    ax0.plot(idx, raw.values, label="Raw SWE", linewidth=1.8)
    if qa is not None:
        ax0.plot(idx, qa.values, label="QA SWE", linewidth=2.2, linestyle="--")

    # static: peer band
    if qa_method == "static" and peer_med is not None and thr is not None:
        ax0.plot(idx, peer_med.values, label="peer median", linewidth=1.5)
        ax0.fill_between(idx, (peer_med - thr).values, (peer_med + thr).values, alpha=0.2, label="peer ± thr")

    # snowmodel: yhat band
    if qa_method == "snowmodel" and yhat is not None and thr is not None:
        ax0.plot(idx, yhat.values, label="snowmodel yhat", linewidth=1.5)
        ax0.fill_between(idx, (yhat - thr).values, (yhat + thr).values, alpha=0.2, label="yhat ± thr")

    # mark method-specific flags only (shared data-integrity reasons are filtered out;
    # they're identical across all three methods and would mis-attribute the cause).
    ax0.scatter(idx[method_flag.astype(bool)], raw[method_flag.astype(bool)],
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

    # method-specific flag (lower half) + majority (upper half)
    axf.fill_between(idx, 0.00, 0.50, where=method_flag.astype(bool).values,
                     step="post", alpha=0.35, label=f"{qa_method} flag")

    if maj is not None:
        axf.fill_between(idx, 0.50, 1.00, where=maj.astype(bool).values,
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
