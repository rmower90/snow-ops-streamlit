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
            loc='upper right',
            fontsize=8,
            title_fontproperties={'weight': 'bold', 'size': 9}
        )
        ax.add_artist(leg1)

        # --- SECOND LEGEND (Base models — BOTTOM) ---
        ax.legend(
            handles=model_handles,
            title='Model Predictions',
            loc='lower right',
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
