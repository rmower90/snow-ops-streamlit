# postprocessing.py
import pandas as pd 
import numpy as np
from typing import List
import datetime 
import os
import sys
import pathlib
import yaml
import xarray as xr

def fill_NaNs_prediction_tables(prediction_mm_lst: List,
                              prediction_acreFt_lst: List,
                              prediction_pillows_lst: List,
                              elev_bin_labels: List,
                              current_date: np.ndarray,
                              prediction_mm_df: pd.DataFrame = None,
                              prediction_acreFt_df: pd.DataFrame = None,
                              prediction_pillows_df: pd.DataFrame = None,
                             ):
    
    # combine current dates predictions for prediction tables.
    prediction_mm_today_df = pd.concat([prediction_mm_lst[0],prediction_mm_lst[1]],ignore_index = True)
    prediction_acreFt_today_df = pd.concat([prediction_acreFt_lst[0],prediction_acreFt_lst[1]],ignore_index = True)
    prediction_pillows_today_df = pd.concat([prediction_pillows_lst[0],prediction_pillows_lst[1]],ignore_index = True)
    # rearrange prediction columns.
    prediction_mm_today_df = format_prediction_tables(prediction_mm_today_df,elev_bin_labels,isAcreFt = False)
    prediction_acreFt_today_df = format_prediction_tables(prediction_acreFt_today_df,elev_bin_labels,isAcreFt = True)
    prediction_pillows_today_df = format_prediction_tables(prediction_pillows_today_df,elev_bin_labels,isAcreFt = False)
    # fill with nans and switch dates.
    prediction_mm_today_df = fill_df_na(prediction_mm_today_df,elev_bin_labels,current_date,isAcreFt = False)
    prediction_acreFt_today_df = fill_df_na(prediction_acreFt_today_df,elev_bin_labels,current_date,isAcreFt = True)
    prediction_pillows_today_df = fill_df_na(prediction_pillows_today_df,elev_bin_labels,current_date,isAcreFt = False)
    # combine current dates predictions with previous prediction tables.
    prediction_mm_df = pd.concat([prediction_mm_df,prediction_mm_today_df],ignore_index = True)
    prediction_acreFt_df = pd.concat([prediction_acreFt_df,prediction_acreFt_today_df],ignore_index = True)
    prediction_pillows_df = pd.concat([prediction_pillows_df,prediction_pillows_today_df],ignore_index = True)
    
    return prediction_mm_df,prediction_acreFt_df,prediction_pillows_df


def arrange_prediction_tables(prediction_mm_lst: List,
                              prediction_acreFt_lst: List,
                              prediction_pillows_lst: List,
                              elev_bin_labels: List,
                              prediction_mm_df: pd.DataFrame = None,
                              prediction_acreFt_df: pd.DataFrame = None,
                              prediction_pillows_df: pd.DataFrame = None,
                             ):
    
    # combine current dates predictions for prediction tables.
    prediction_mm_today_df = pd.concat([prediction_mm_lst[0],prediction_mm_lst[1]],ignore_index = True)
    prediction_acreFt_today_df = pd.concat([prediction_acreFt_lst[0],prediction_acreFt_lst[1]],ignore_index = True)
    prediction_pillows_today_df = pd.concat([prediction_pillows_lst[0],prediction_pillows_lst[1]],ignore_index = True)
    # rearrange prediction columns.
    prediction_mm_today_df = format_prediction_tables(prediction_mm_today_df,elev_bin_labels,isAcreFt = False)
    prediction_acreFt_today_df = format_prediction_tables(prediction_acreFt_today_df,elev_bin_labels,isAcreFt = True)
    prediction_pillows_today_df = format_prediction_tables(prediction_pillows_today_df,elev_bin_labels,isAcreFt = False)
    # combine current dates predictions with previous prediction tables.
    prediction_mm_df = pd.concat([prediction_mm_df,prediction_mm_today_df],ignore_index = True)
    prediction_acreFt_df = pd.concat([prediction_acreFt_df,prediction_acreFt_today_df],ignore_index = True)
    prediction_pillows_df = pd.concat([prediction_pillows_df,prediction_pillows_today_df],ignore_index = True)
    
    return prediction_mm_df,prediction_acreFt_df,prediction_pillows_df

def format_prediction_tables(today_df: pd.DataFrame,
                             elev_bin_labels: List,
                             isAcreFt: bool = True,
                            ):
    # rename total basin predicted SWE.
    today_df = today_df.rename(columns = {'Total':'Basin'})
    # change to date.
    today_df['Date'] = pd.to_datetime(today_df['Date'])
    if isAcreFt:
    # final basin ordering
        column_order = ['Date','Model Type','Training Infer NaNs','Prediction QA'] + elev_bin_labels + ['Band Sum','Basin']
        # create summed total SWE.
        today_df['Band Sum'] = 0
        for bin in elev_bin_labels:
            today_df['Band Sum'] += today_df[bin]
        # reorder.
        today_df = today_df[column_order]
        # convert to thousands of acre feet for table output.
        conversion_columns = elev_bin_labels + ['Band Sum','Basin']
        today_df[conversion_columns] = today_df[conversion_columns] / 1000
        today_df[conversion_columns] = today_df[conversion_columns].astype(int)
    else:
        column_order = ['Date','Model Type','Training Infer NaNs','Prediction QA'] + elev_bin_labels + ['Basin']
        today_df = today_df[column_order]
    return today_df

def fill_df_na(today_df: pd.DataFrame,
               elev_bin_labels: List,
               current_date: np.ndarray,
               isAcreFt: bool = True,
                ):
    # switch to current date.
    today_df['Date'] = current_date
    today_df['Date'] = pd.to_datetime(today_df['Date'])
    if isAcreFt:
    # final basin ordering
        columns = elev_bin_labels + ['Band Sum','Basin']
    else:
        columns = elev_bin_labels + ['Basin']
    for bin in columns:
        today_df[bin] = np.nan
    return today_df