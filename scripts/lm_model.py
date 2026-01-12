# lm_model.py
import xarray as xr
import os 
import rioxarray as rxr
from pyproj import CRS, Transformer
import sys
import geopandas as gpd
import numpy as np
import warnings
import rasterio
from sklearn import linear_model
import matplotlib.pyplot as plt
import copy
import sklearn.metrics as metrics
import pandas as pd
import matplotlib.cm as cm
import matplotlib.colors as mcolors
warnings.filterwarnings("ignore")
from itertools import combinations

# lm_model.py
import xarray as xr
import os 
import rioxarray as rxr
from pyproj import CRS, Transformer
import sys
import geopandas as gpd
import numpy as np
import warnings
import rasterio
from sklearn import linear_model
import matplotlib.pyplot as plt
import copy
import sklearn.metrics as metrics
import pandas as pd
import matplotlib.cm as cm
import matplotlib.colors as mcolors
warnings.filterwarnings("ignore")
from itertools import combinations


def run_all_mlr_models(aso_tseries_1,obs_data_hist,current_vals_df,aso_site_name,all_pils,all_pils_QA,df_sum_total,baseline_pils,start_wy,
                       end_wy,isSplit,isAccum,prediction_dir,prediction_date,dem_bin,labels_from_yaml,QA_flag = 'NA',
                       modelNUM = None,isMean = False,saveModels = True,showOutput = False,
                       saveValidation = False,pickledir = None,add_zeroASO = True,isCombination_ = False):
    """
    Slice data based on accumulation / melt threshold for each year.
    Input:
      aso_tseries_1 - relative filepath of threshold csv. 
      obs_data_hist - list of xarray time series of pillow data.
      current_vals_df - pandas dataframe of current days pillow data.
      aso_site_name - python string of aso site name.
      all_pils - python list of all possible pillows.
      all_pils_QA - python list of all possible pillows based on QA category.
      df_sum_total - pandas summary datatable of a features and labels.
      baseline_pils - python list of selected pillows to be considered.
      start_wy - integer of start water year.
      end_wy - integer of end water year.
      aso_site_name - python string of aso site name.
      isSplit - boolean for indicating whether accumulation and melt seasons will be split.
      isAccum - boolean for indicating accumulation or melt season. Only relevant if isSplit = True. 
      prediction_dir - python string of relative filepath for model output.
      prediction_date - datetime for prediction.
      dem_bin - xarray for DEM based on elevation.
      labels_from_yaml - python list of bin labels.
      QA_flag - current date data quality flag. (1 - most conservative; 3 - most liberal)
      modelNUM - model number.
      isMean - boolean for indicating whether imputation is with mean or predicted value.
      saveModels - boolean for indicating whether model summary and pickle files will be outputted to disc.
      pickledir - python string of relative filepath for model pickle files.
      add_zeroASO - Boolean indicating whether final model will be trained with an observation of
                    zero ASO and feature values.
      isCombination - Boolean indicating whether selection process uses combination.
    
    Output:
      df_accum - accumulation dataframe.
      df_melt - melt dataframe.
    """
    # make sure output directories exist.
    pred_site_dir = f'{prediction_dir}{aso_site_name}/'
    pred_summary_dir = f'{pred_site_dir}summary/'
    pred_model_dir = f'{pred_site_dir}model/'
    pred_table_dir = f'{pred_site_dir}historic/'

    if saveModels:
        create_directory(prediction_dir)
        create_directory(pred_site_dir)
        create_directory(pred_summary_dir)
        create_directory(pred_model_dir)
        create_directory(pred_table_dir)

    # load save output files.
    summary_dict = f'{pred_summary_dir}{aso_site_name}_test_summary.json'
    if os.path.exists(summary_dict):
        # load json.
        pass 
    else:
        if modelNUM is None:
            modelID = 0
        else:
            modelID = modelNUM
        summary_dict_all = {}

    # run models
    ## iterate over imputation or not.
    df_sheet_mm_lst = []
    df_sheet_acreFt_lst = []
    df_sheet_lst_rmse = []
    df_sheet_pillow_lst = []
    for isImpute in [False,True]:
        if showOutput: print('Impute',isImpute)
    
    ## iterate over elevation band.
        for elev_band in range(0,len(aso_tseries_1.elev)):
            if showOutput: print('Elev', elev_band)
            # run cross validation.
            df_split,summary_dict_model,aso_tseries_2,obs_data_6,rmse = run_mlr_train_predict(aso_tseries_1,obs_data_hist,elev_band,all_pils,all_pils_QA,df_sum_total,
                                                                baseline_pils,start_wy,end_wy,aso_site_name,isSplit,isAccum,isImpute,
                                                                isMean,prediction_date,modelID, QA_flag,model_type = 'MLR',
                                                                showOutput = showOutput,isCombination = isCombination_,
                                                                saveValidation = saveValidation)
            
            # pull selected pillows.
            selected_pils = summary_dict_model[modelID]['model_features']['features']
            if showOutput: print('elev_band',elev_band,'selected_pils',selected_pils)

    # return df_split,summary_dict_model
            #run daily prediction.
            # try:
            elev_vals = dem_bin[elev_band].values.flatten()
            # count of valid grids.
            count = len(elev_vals[~np.isnan(elev_vals)])
            # count to m2.
            area_m2 = count * 50 * 50
            # predictions in mm
            yhat_mm,df_train,yhat_test = run_daily_prediction(df_split,current_vals_df,selected_pils,modelID,conversion = 0,
                                        area_m2 = area_m2,outdir = pickledir,add_zeroASO = add_zeroASO)
            # predictions in acre ft
            yhat_acreft,df_train,yhat_test = run_daily_prediction(df_split,current_vals_df,selected_pils,modelID,conversion = 2,
                                        area_m2 = area_m2,outdir = pickledir,add_zeroASO = add_zeroASO)
            # add prediction to summary dictionary.
            ## predictions mm
            summary_dict_model[modelID]['prediction']['mm'].append(float(yhat_mm))
            ## prediction acre-ft
            summary_dict_model[modelID]['prediction']['acre_ft'].append(float(yhat_acreft))

            summary_dict_all = {**summary_dict_all, **summary_dict_model}
            modelID += 1
        #     break 
        # break

        # total basin swe volume sheet.
        df_sheet_mm = create_prediction_sheet(summary_dict_all,modelID-1,labels_from_yaml,'mm')
        # total basin swe volume [acre-ft] sheet.
        df_sheet_acreFt = create_prediction_sheet(summary_dict_all,modelID-1,labels_from_yaml,'acre_ft')
        # pillow sheet.
        df_pillow_sheet = create_pillow_sheet(summary_dict_all,modelID-1,labels_from_yaml)
        # rmse sheet.
        df_sheet_rmse = create_rmse_sheet(summary_dict_all,modelID-1)

        # append sheets.
        df_sheet_mm_lst.append(df_sheet_mm)
        df_sheet_acreFt_lst.append(df_sheet_acreFt)
        df_sheet_lst_rmse.append(df_sheet_rmse)
        df_sheet_pillow_lst.append(df_pillow_sheet)
        

    # return summary_dict_all,df_split,yhat_mm,df_sheet_mm_lst,df_sheet_acreFt_lst,df_sheet_lst_rmse,df_train,yhat_test,aso_tseries_2,obs_data_6
    return summary_dict_all,df_sheet_mm_lst,df_sheet_acreFt_lst,df_sheet_pillow_lst

def process_melt_accum_thresh(thresh_fpath,df_sum_total,elev_bin,isAccum):
    """
    Slice data based on accumulation / melt threshold for each year.
    Input:
      thresh_fpath - relative filepath of threshold csv. 
      df_sum_total - pandas dataframe of summary data. 
      elev_bin - integer of elevation bin.
      isAccum - boolean for indicating accumulation or melt analysis.
    
    Output:
      df_accum - accumulation dataframe.
      df_melt - melt dataframe.
    """
    ## load melt threshold dataframe ##
    melt_thresh_df = pd.read_csv(thresh_fpath)
    melt_thresh_df['threshold_1'] = pd.to_datetime(melt_thresh_df['threshold_1'])
    melt_thresh_df['threshold_2'] = pd.to_datetime(melt_thresh_df['threshold_2'])
    melt_thresh_df['threshold_best'] = pd.to_datetime(melt_thresh_df['threshold_best'])

    ## slice data ##
    if elev_bin < 3:
        ## BIG ASSUMPTION - that thresholds below elev bin 3 are equal to 
        df_thresh_elev = melt_thresh_df[melt_thresh_df['elev_bin'] == elev_bin][['water_year','threshold_best']].reset_index().drop(columns = ['index'])
    else:
        df_thresh_elev = melt_thresh_df[melt_thresh_df['elev_bin'] == elev_bin][['water_year','threshold_best']].reset_index().drop(columns = ['index'])

    ## add wateryear to df_sum_total for joining
    df_sum_total['water_year'] = df_sum_total.time.dt.year

    ## merge and threshold data 
    df_merge = df_sum_total.merge(df_thresh_elev, on='water_year', how='left')
    if isAccum:
        df_accum = df_merge[df_merge['time'] < df_merge['threshold_best']].reset_index().drop(columns = ['water_year','threshold_best','index'])
        return df_accum
    else:
        df_melt = df_merge[df_merge['time'] >= df_merge['threshold_best']].reset_index().drop(columns = ['water_year','threshold_best','index'])
        return df_melt

def run_mlr_train_predict(aso_tseries_1,obs_data_hist,elev_band,all_pils,all_pils_QA,df_sum_total,baseline_pils,start_wy,end_wy,
                         aso_site_name,isSplit,isAccum,isImpute,isMean,prediction_date,
                         modelID,QA_flag,model_type = 'MLR',showOutput = False,isCombination = False,
                         saveValidation = False):
    """
    Run single multiple linear regression cross validation and output results in dictionary.
    Input:
      aso_tseries_1 - relative filepath of threshold csv. 
      obs_data_hist - list of xarray time series of pillow data. 
      elev_band - integer of elevation bin.
      all_pils - python list of all possible pillows.
      all_pils_QA - all_pils_QA - python list of all possible pillows based on QA category.
      df_sum_total - pandas summary datatable of a features and labels.
      baseline_pils - python list of selected pillows to be considered.
      start_wy - integer of start water year.
      end_wy - integer of end water year.
      aso_site_name - python string of aso site name.
      isSplit - boolean for indicating whether accumulation and melt seasons will be split.
      isAccum - boolean for indicating accumulation or melt season. Only relevant if isSplit = True.
      isImpute - boolean for indicating whether NaNs will be filled.
      isMean - boolean for indicating whether imputation will use mean or predicted value.
      prediction_date - prediction date.
      modelID - python integer for unique model indentifier.
      QA_flag - current date data quality flag. (1 - most conservative; 3 - most liberal)
      model_type - python string for type of ML model.
      showOutput - boolean for printing and visualizing plots.
      isCombination - boolean for indicating station selection 
      (either by looking at combination of all selected pillows in cross folds or most frequently selected pillows).

    
    Output:
      df_split - training dataframe.
      summary_dict - summary dictionary of validation parameters and results.
    """

    ## code
    title_str = str(aso_tseries_1.elev[elev_band].values)

    if isSplit:
        if isImpute:
            ## update missing observations with average from flight date.
            if isMean:
                obs_data_5_,pils_removed,df_summary_impute = impute_pillow_mean(df_sum_total,all_pils_QA,obs_data_hist,obs_threshold = 0.50)
            else:
                obs_data_5_,pils_removed,df_summary_impute = impute_pillow_prediction(df_sum_total,all_pils_QA,obs_data_hist,aso_site_name,prediction_date,obs_threshold = 0.50)

            ## split up data into accumlation and melt.
            # df_split = process_melt_accum_thresh(f'./data/summary_table/{aso_site_name}/1000_ft/melt_threshold.csv',
            df_split = process_melt_accum_thresh(f'/home/rossamower/work/aso/data/mlr_prediction/{aso_site_name}/melt_threshold/melt_threshold.csv',
                                                           df_summary_impute,
                                                           elev_band,
                                                           isAccum)
            predictions_bestfit, predictions_validation, stations2, aso_tseries_2, obs_data_6 = run_cross_val_selection(obs_data_5_,df_split,aso_tseries_1,pils_removed,start_wy,end_wy,
                                                                                                                         elev_band,isCombination = isCombination,showOutput = showOutput,isMelt = isSplit)
        else:
            ## split up data into accumlation and melt.
            # df_split = process_melt_accum_thresh(f'./data/summary_table/{aso_site_name}/1000_ft/melt_threshold.csv',
            df_split = process_melt_accum_thresh(f'/home/rossamower/work/aso/data/mlr_prediction/{aso_site_name}/melt_threshold/melt_threshold.csv',
                                                           df_sum_total,
                                                           elev_band,
                                                           isAccum)
            # return df_split,obs_data_hist,aso_tseries_1,baseline_pils
            predictions_bestfit, predictions_validation, stations2, aso_tseries_2, obs_data_6 = run_cross_val_selection(obs_data_hist,df_split,aso_tseries_1,baseline_pils,start_wy,end_wy,
                                                                                                                         elev_band,isCombination = isCombination,showOutput = showOutput,isMelt = isSplit)
    else:
        if isImpute:
            ## update missing observations with average from flight date.
            if isMean:
                obs_data_5_,pils_removed,df_summary_impute = impute_pillow_mean(df_sum_total,all_pils_QA,obs_data_hist,obs_threshold = 0.50)
            else:
                obs_data_5_,pils_removed,df_summary_impute = impute_pillow_prediction(df_sum_total,all_pils_QA,obs_data_hist,aso_site_name,prediction_date,obs_threshold = 0.50)

            predictions_bestfit, predictions_validation, stations2, aso_tseries_2, obs_data_6 = run_cross_val_selection(obs_data_5_,df_summary_impute,aso_tseries_1,pils_removed,start_wy,end_wy,
                                                                                                                         elev_band,isCombination = isCombination,showOutput = showOutput,isMelt = isSplit)
        else:
            predictions_bestfit, predictions_validation, stations2, aso_tseries_2, obs_data_6 = run_cross_val_selection(obs_data_hist,df_sum_total,aso_tseries_1,baseline_pils,start_wy,end_wy,
                                                                                                                         elev_band,isCombination = isCombination,showOutput = showOutput,isMelt = isSplit)

    if saveValidation:
        # fig = plt.figure(figsize=(10, 13))

        # ax1 = plt.subplot2grid((4, 2), (0, 0), colspan=2,rowspan=2)  # Spans all 3 columns in row 0
        # ax2 = plt.subplot2grid((4, 2), (2, 0), colspan=2)  # Spans 2 columns in row 1
        # ax4 = plt.subplot2grid((4, 2), (3, 0))            # Single cell in row 2, column 0
        # ax5 = plt.subplot2grid((4, 2), (3, 1))            # Single cell in row 2, column 1
        # if isImpute:
        #     if elev_band == 7:
        #         plt.suptitle(f"{aso_site_name} Validation Total Mean SWE\nn={len(predictions_validation)}, predict NaNs, QA_Flag = {QA_flag}",fontweight = 'bold',fontsize = 20)
        #     else:
        #         plt.suptitle(f"{aso_site_name} Validation {title_str}' Mean SWE\nn = {len(predictions_validation)},predict NaNs, QA_Flag = {QA_flag}",fontweight = 'bold',fontsize = 20)
        # else:
        #     if elev_band == 7:
        #         plt.suptitle(f"{aso_site_name} Validation Total Mean SWE\nn={len(predictions_validation)}, drop NaNs, QA_Flag = {QA_flag}",fontweight = 'bold',fontsize = 20)
        #     else:
        #         plt.suptitle(f"{aso_site_name} Validation {title_str}' Mean SWE\nn = {len(predictions_validation)},drop NaNs, QA_Flag = {QA_flag}",fontweight = 'bold',fontsize = 20)
    
        # plt.subplots_adjust(left=None, bottom=None, right=None, top=0.90, wspace=0.3, hspace=0.5)

        # predictions_v_observations(aso_tseries_2[:,elev_band],predictions_bestfit,predictions_validation,ax1,elev_band,
        #                        features = None,showPlot = True)

        # predictions_v_observations_tseries(aso_tseries_2[:,elev_band],predictions_validation,obs_data_6,
        #                                stations2,start_wy,end_wy,ax2,max_swe = 1500)
        predictions_v_observations(aso_tseries_2[:,elev_band],
                                   predictions_bestfit,
                                   predictions_validation,
                                   aso_site_name,
                                   features = None,
                                   showPlot = True)

        predictions_v_observations_tseries(aso_tseries_2[:,elev_band],
                                           predictions_validation,
                                           obs_data_6,
                                           stations2,
                                           start_wy,
                                           end_wy,
                                           aso_site_name,
                                           max_swe = 1500)

    ## plot residuals.
    if saveValidation:
        max_resid_absolute,max_resid_percent = residual_comparison(aso_tseries_2[:,elev_band],predictions_validation,None,None,None,
                                                               mask_val = 50,showPlot = False)
        # max_resid_absolute,max_resid_percent = residual_comparison(aso_tseries_2[:,elev_band],predictions_validation,ax4,ax5,plt,
        #                                                        mask_val = 50,showPlot = True)
    else:
        max_resid_absolute,max_resid_percent = residual_comparison(aso_tseries_2[:,elev_band],predictions_validation,None,None,None,
                                                               mask_val = 50,showPlot = False)
    

    if showOutput:
        plt.show()
    if saveValidation:
        pass
        # validation_dir = f'./data/predictions/{aso_site_name}/validation/'
        # mlr_dir = f'{validation_dir}MLR/'
        # if isImpute:
        #     impute_dir = f'{mlr_dir}Predict_NaNs/'
        # else:
        #     impute_dir = f'{mlr_dir}Drop_NaNs/'
        
        # if elev_band == 0:
        #     elev_dir = f'{impute_dir}7k/' 
        # elif elev_band == 1:
        #     elev_dir = f'{impute_dir}7-8k/' 
        # elif elev_band == 2:
        #     elev_dir = f'{impute_dir}8-9k/' 
        # elif elev_band == 3:
        #     elev_dir = f'{impute_dir}9-10k/' 
        # elif elev_band == 4:
        #     elev_dir = f'{impute_dir}10-11k/' 
        # elif elev_band == 5:
        #     elev_dir = f'{impute_dir}11-12k/' 
        # elif elev_band == 6:
        #     elev_dir = f'{impute_dir}12k/' 
        # elif elev_band == 7:
        #     elev_dir = f'{impute_dir}Total/' 

        # qa_dir = f'{elev_dir}QA_{QA_flag}/'

        # if not os.path.exists(validation_dir): os.makedirs(validation_dir)
        # if not os.path.exists(mlr_dir): os.makedirs(mlr_dir)
        # if not os.path.exists(impute_dir): os.makedirs(impute_dir)
        # if not os.path.exists(elev_dir): os.makedirs(elev_dir)
        # if not os.path.exists(qa_dir): os.makedirs(qa_dir)

    selected_pils = [obs_data_6[i].name for i in range(0,len(obs_data_6)) if i in stations2]

    rmse_mm = metrics.root_mean_squared_error(aso_tseries_2[:,elev_band].values ,predictions_validation)

    summary_dict = regression_results(aso_tseries_2[:,elev_band].values * 0.0393701, predictions_validation * 0.0393701,
                                      max_resid_absolute * 0.0393701,max_resid_percent,rmse_mm,selected_pils,prediction_date,
                                      modelID,isAccum,isImpute,isMean,title_str,model_type,QA_flag)

    
    df_final = aso_tseries_2[:,elev_band].to_dataframe()
    for i in stations2:
        df_inter = obs_data_6[i][obs_data_6[i].time.isin(df_final.index.values)].to_dataframe()
        df_inter.index.names = ['date']
        df_final = pd.merge(left = df_final,right = df_inter,left_index = True,right_index=True)

    return df_final,summary_dict,aso_tseries_2,obs_data_6,rmse_mm

def predictions_v_observations(aso_mean_swe,
                               bestFit,
                               validation,
                               aso_site_name,
                               saveFig = False,
                               max_swe = 1500,
                               features = None,
                               showPlot = True):
    """
        Produces one to one plot of basin mean swe vs. predicted swe 
        Input:
            aso_mean_swe - xarray dataarray of basin mean swe for aso flight dates.
            bestFit - np array of best fit predictions. 
            validation - np array of cross-validation approach.
            aso_site_name - python string for aso basin name.
            saveFig - python boolean for saving figure
            max_swe - max swe used for limits on figure.
            features - list of features.
        Output:
            plot of best fit and cross validation.
    """
    ## calculate statistics
    rms_validation = np.sqrt(((aso_mean_swe - validation)**2).mean().values)
    r2_validation = np.corrcoef(aso_mean_swe.values,validation)[0,1]**2
    rms_best = np.sqrt(((aso_mean_swe - bestFit)**2).mean().values)
    r2_best = np.corrcoef(aso_mean_swe.values,bestFit)[0,1]**2
    ## plotting
    if showPlot:
        plt.figure(dpi=300)
        plt.plot(validation, aso_mean_swe,'o', color = 'C1', label="Validation: r$^2$" + f'= {r2_validation:.3f}\nRMS error     = {rms_validation:.2f}')
        plt.plot(bestFit, aso_mean_swe,'x', color = 'C0', label = 'Best Fit: r$^2$' + f'    = {r2_best:.3f}\nRMS error     = {rms_best:.2f}')
        plt.ylabel("Basin Mean SWE [mm]")
        plt.xlabel("Predicted Basin Mean SWE [mm]")
        if features is not None:
            plt.title(f'Predictions \nn = {len(validation)}; k = {len(features)}',fontweight = 'bold')
        plt.legend()
        plt.ylim([0,max_swe])
        plt.xlim([0,max_swe])
        if saveFig == True:
            if not os.path.exists(f'./data/figures/{aso_site_name}'):
                os.makedirs(f'./data/figures/{aso_site_name}')
            plt.savefig(f'./data/figures/{aso_site_name}/03_predicted_mean_swe.png',dpi=300)
        plt.show()
        return 
    else:
        return r2_validation,rms_validation,r2_best,rms_best

def predictions_v_observations_tseries(aso_mean_swe,
                                       validation,
                                       pillow_data,
                                       stations,
                                       start_wy,
                                       end_wy,
                                       aso_site_name,
                                       saveFig = False,
                                       max_swe = 1500):
                              
    """
        Plot time series of pillows that are best predictors for ASO.
        Input:
            aso_mean_swe - xarray dataarray of basin mean swe for aso flight dates.
            validation - np array of cross-validation approach.
            pillow_data - xarray dataset with pillow data.
            stations - list of station integers identifying best predictors.
            start_wy - integer for starting water year.
            end_wy - integer for ending water year.
            max_swe - max swe used for limits on figure.
        Output:
            time series plot.
    """
    
    plt.figure(dpi=300, figsize=(8,3))

    ax2 = plt.gca()
    ax1 = plt.twinx(ax2)

    ax2.plot(aso_mean_swe.date, aso_mean_swe.data, marker="o", linestyle="", color="C0", label="ASO", zorder=3)
    ax2.plot(aso_mean_swe.date, validation, 'x', color="C1", label="Predicted", zorder=3)
    plt.ylabel("Basin Mean SWE [mm]")

    colors = [f'C{str(i+2)}' for i in range(0,len(stations))]

    for i,c in zip(stations, colors):
        plot_data = np.array(pillow_data[i].data, dtype="f")
        
        ax1.plot(pillow_data[i].time, plot_data, label=pillow_data[i].name, color=c)


    ax1.set_ylabel("Pillow Observed SWE [mm]")
    ax2.set_ylabel("Basin Mean SWE [mm]")
    # ax2.set_ylim(0,max_swe)
    ax2.set_xlim(np.datetime64(f'{start_wy-1}-10-01'),np.datetime64(f'{end_wy}-08-01'))
    ax1.set_xlim(np.datetime64(f'{start_wy-1}-10-01'),np.datetime64(f'{end_wy}-08-01'))
    ax2.set_ylim(0,max_swe)
    ax1.set_ylim(0,3600)
    # plt.xlim(np.datetime64("2012-10-01"),np.datetime64("2019-09-01"))
    ax1.legend(loc="upper left")
    ax2.legend(loc="upper right")
    if saveFig == True:
        if not os.path.exists(f'./data/figures/{aso_site_name}'):
            os.makedirs(f'./data/figures/{aso_site_name}')
        plt.savefig(f'./data/figures/{aso_site_name}/02_aso_insitu_tseries.png',dpi=300)
    plt.show()
    return

def run_cross_val_selection(obs_data,df_sum_total,aso_tseries,all_pillows,start_wy,end_wy,
                            elev_band = -1,isCombination = True,showOutput = True,isMelt = False):
    """
        Runs cross-validation approach of linear regression based on year used 
        to run testing.
        Input:
            obs_data - list of xarray pillow observations.
            df_sum_total - pandas dataframe for observations, features, and labels WITH nans.
            aso_tseries_1 - xarray object for aso mean swe based on elevation.
            all_pillows - list of pillows to use.
            start_wy - integer for start water year.
            end_wy - integer for end water year.
            elev_band - elevation band to run model (note: -1 is entire domain)
            isCombination - boolean to determine which approach to use.
            isMelt - boolean to indicate additional threshold of dates for melt and accum.
            showOutput - boolean to indicate show output.
        Output:
            preds - list of predictions for each cross-validated year.
    """
    ## slice observation data ##
    obs_data_6 = [obs_data[i] for i in range(0,len(obs_data)) if obs_data[i].name in all_pillows]

    ## slice for melt or accum ##
    if isMelt:
        aso_tseries = aso_tseries[aso_tseries.date.isin(df_sum_total.time.values)]

    ## identify missing times ##
    missing_times = df_sum_total[df_sum_total[all_pillows].isnull().sum(axis=1) > 0].time.values

    ## slice aso mean swe ##
    aso_tseries_2 = aso_tseries[~aso_tseries.date.isin(missing_times)]

    ## run summary ##
    summary_data_total = summarize_data(obs_data_6, aso_tseries_2[:,elev_band])

    predictions_bestfit = summarize(summary_data_total,plotFig = showOutput,saveFig = False, axis_max = 1600)
    if showOutput:
        print(f'All features correlation: {np.corrcoef(predictions_bestfit, summary_data_total[-1])[0,1]**2:.3f}')
    ## run station selection based on cross-validation
    nstations = summary_data_total.shape[0]-1
    predictions_validation,best_stations = cross_val_loyo_pred_select(aso_tseries_2[:,elev_band], summary_data_total[:-1,:],
                                                                      start_wy,end_wy,showOutput = showOutput)
    predictions_validation = np.array(predictions_validation)
    predictions_validation[predictions_validation<0]=0
    if showOutput:
        print('')
        print(f'Validation station correlation: {np.corrcoef(predictions_validation,summary_data_total[-1])[0,1]**2:.3f}')
        print('Select best stations')
    stations2 = identify_best_stations(best_stations,aso_tseries_2,elev_band,summary_data_total,isCombination,showOutput = showOutput)
    predictions_bestfit = cross_val_loo(aso_tseries_2[:,elev_band], summary_data_total[stations2,:])
    predictions_bestfit = np.array(predictions_bestfit)
    predictions_bestfit[predictions_bestfit<0]=0

    # print(f'Best fit station correlation: {np.corrcoef(predictions_validation,summary_data_total[-1])[0,1]**3:.3f}')
    # return predictions_bestfit, predictions_validation, stations2, aso_tseries_2, obs_data_6
    return predictions_bestfit, predictions_validation, stations2, aso_tseries_2, obs_data_6

def summarize_data(pillow_data, aso_tseries):
    """
        Processes pillow data to match ASO time series using nearest date. 
        Input:
            pillow_data - list object of xarray dataarray objects representing
                          timeseries of each snow pillow.m
            time_series - dataarray object representing ASO time series of 
                          basin averaged SWE (mm)
        Output:
            full_data - numpy 2D array with number of pillows represented by number of
                        rows and aso dates represented by columns. Values within the
                        matrix represent the snow pillow values closest to each ASO 
                        flight. The last row data[-1,:] represents the actual ASO 
                        averaged values.
    """
    full_data = np.zeros((len(pillow_data)+1,len(aso_tseries)))

    for i,t in enumerate(aso_tseries):
        for j,p in enumerate(pillow_data):
            full_data[j,i] = p.sel(time=t.date, method="nearest")
            
## set the aso observations in the last row ##
        full_data[len(pillow_data),i] = t.values

    return full_data


def summarize(data,plotFig = True,saveFig=False,axis_max = 1600):
    """
        Runs linear regression on snow pillows to predict ASO basin mean SWE. 
        Utilizes Sklearn's linear_model.LinearRegression() to make the predictions. 
        Input:
            data - numpy 2D array of snow pillow observations and ASO basin mean SWE
                   in milimeters. The last row data[-1,:] represents the actual ASO 
                   averaged values.
        Output:
            res - numpy 1D array represented the predictions of ASO basin mean SWE for 
                  each data on linear regression model. 
    """
    ## create the linear regression model ##
    lm = linear_model.LinearRegression()
    
    ## fit model with data ##
    lm.fit(data[:-1,:].T,data[-1,:])
    
    ## predict aso values ##
    res = lm.predict(data[:-1,:].T)
    
    ## calculate statistics
    rms = np.sqrt(((res-data[-1])**2).mean())
    r2 = np.corrcoef(res,data[-1])[0,1]**2

    ## plot ASO vs. predicted SWE [mm] ##
    if plotFig:
        plt.clf()
        plt.plot(res,data[-1],'x',label = 'best fit: r$^2$' +f'={r2:.3f},RMS error={rms:.3f}')
        plt.ylabel("ASO SWE [mm]")
        plt.xlabel("predicted SWE [mm]")
        plt.legend()
        plt.xlim([0,axis_max])
        plt.ylim([0,axis_max])


        # plt.title("r$^2$  "+str(r2)[:5]+"  RMS error = "+str(rms)[:5])
        plt.title(f"Best Fit")
        plt.tight_layout()
        plt.show()
        if saveFig:
            plt.savefig("summary.png")
    # plt.xlim([0,625])
    # plt.ylim([0,575])
    return res

def cross_val_loyo_pred_select(aso, index, start_year, end_year,
                               add_points=None, max_params=None,showOutput = True):
    """
        Runs cross-validation approach of linear regression based on year used 
        to run testing.
        Input:
            aso - datarray object representing ASO basin SWE.
            index - 2D numpy array representing the snow pillow features.
            start_year - integer representing starting year of ASO flights. 
            end_year - integer representing ending year of ASO flights. 
        Output:
            preds - list of predictions for each cross-validated year.
    """
    preds = []
    param_lst = []
    ## loop through and test each year of data in cross validation approach ##
    for y in range(start_year,end_year+1):
        ## check to see if any flights are observed in year ##
        if len([True for i in aso.date if y == int(i.date.dt.year.values)]) > 0:
            test = []
            testy = []
            train = []
            trainy = []

            for i in range(len(aso)):
                if int(aso[i].date.dt.year.values) == y:
                    ## pull out pillow values for date ##
                    values = index[:,i]
                    ## set nonfinite values to 0 ##
                    values[~np.isfinite(values)] = 0
                    ## append features ##
                    test.append(values)
                    ## append labels ##
                    testy.append(aso.values[i])
                else:
                    values = index[:,i]
                    values[~np.isfinite(values)] = 0
                    train.append(values)
                    trainy.append(aso.values[i])

            if add_points is not None:
                for point in add_points:
                    train.append(point[0])
                    trainy.append(point[1])
               
            testy, params,error = compute_linear(np.array(train), np.array(trainy), np.array(test), np.array(testy), max_params=max_params)

            if showOutput: print(f"{y}: {params}, {error:.2f}")
            param_lst.append(params)

            if testy is not None:
                preds.extend(testy)
            else:
                preds.extend(np.zeros(len(test)))
        else:
            # pass
            if showOutput: print(f'No flights for: {y}')
            
    return preds,param_lst

def compute_linear(x, y, x_hat=None, y_hat=None, max_params=5):
    """
        Runs linear regression to produce predictions for each tested year.
        Input:
            x - numpy array for training features
            y - numpy array for training labels
            x_hat - numpy array for testing features.
            y_hat - numpy array for testing labels.
            max_params - integer ??????
        Output:
            res - numpy array of predictions
            p - list of parameters or most important snow observations
    """
    ## if no data exist for testing year, run linear regression on training ##
    if (y_hat is None) or (x_hat is None): 
        lm = linear_model.LinearRegression()
        lm.fit(x,y)
        p = np.arange(x.shape[1])
    else:
    ## find best model ##
        lm, p, error = fit_best_model(x,y, x_hat,y_hat, max_params=max_params)
        if lm is None:
            return None, None


    if x_hat is None:
        res = lm.predict(x[:,p])
    else:
        res = lm.predict(x_hat[:,p])

    if (y_hat is None) or (x_hat is None):
        return res
    else:
        return res, p, error

def fit_best_model(x,y, x_hat,y_hat, max_params=None):
    """
        Runs linear regression to produce predictions for each tested year.
        Input:
            x - numpy array for training features
            y - numpy array for training labels
            x_hat - numpy array for testing features.
            y_hat - numpy array for testing labels.
            max_params - integer ??????
        Output:
            res - numpy array of predictions
            p - numpy array of number of observations.
    """
    new_parameters = []
    parameters = []

    error = 10000000
    best_error = error

    if max_params is None:
        max_params = x.shape[0]-1

    ## select optimimum model based on maximum r2 values provided by
    ## subsequently adding more observations ##
    while (error <= best_error) and (len(parameters) < max_params):
        best_error = error
        parameters = new_parameters
        ## add new station that improves r2 ##
        new_parameters = add_parameter(parameters, x,y)
        ## calculate improvement as a result of new station ##
        lm = get_model(new_parameters, x, y)
        error = np.sqrt(np.mean((lm.predict(x_hat[:,new_parameters]) - y_hat)**2))
        # print(f'{new_parameters},{error:.2f}')

    if len(parameters)==0:
        print('HERE')
        return None, None, None
        # raise ValueError("0 parameters not allowed")
    
    ## get linear regression model for optimum observations ##
    lm = get_model(parameters, x,y)

    return lm, parameters, best_error

def add_parameter(param,x,y):
    corr = np.zeros(x.shape[1])
    for i in range(x.shape[1]):
        if i in param:
            pass
        else:
            test_param = param.copy()
            test_param.append(i)
            lm = get_model(test_param, x, y)
            newy = lm.predict(x[:,test_param])
            corr[i] = np.corrcoef(newy,y)[0,1]**2
            if not np.isfinite(corr[i]):
                corr[i] = 0

    best_param = np.argmax(corr)
    test_param = param.copy()
    test_param.append(best_param)
    return test_param


def get_model(param, x, y):
    lm = linear_model.LinearRegression()
    lm.fit(x[:, param],y)
    return lm

def identify_best_stations(best_stations,aso_tseries_2,elev_band,summary_data_total,
                           isCombination = True,showOutput = True):
    """
        Identifies best stations from cross-validation folds using two approaches.
        First, run combination of all selected pillows and select best stations
        based on adjusted R2. Second, start with pillow that has the most occurences in
        across all folds and add on the subsequent pillows of most occurence based
        on R2 value.
        Input:
            best_stations - list lists of selected pillows from cross-validation.
            summary_data_total - numpy array containing features and labels.
            aso_tseries_2 - xarray object for aso mean swe based on elevation.
            all_pillows - list of pillows to use.
            elev_band - elevation band to run model (note: -1 is entire domain).
            isCombination - boolean to determine which approach to use.
            showOutput - boolean to print output.
        Output:
            best_pillows - list of predictions for each cross-validated year.

    """
    ## instantiate max adjusted r2 variable.
    r2adj_max = -1000
    ## use combination approach
    if isCombination:
        station_lst = list(set(sum(best_stations, [])))
        # for i in range(1,len(station_lst)+1):
        for i in range(1,6):
            if showOutput: print(i)
            comb = list(combinations(station_lst, i))
            for val in comb:
                predictions_validation = cross_val_loo(aso_tseries_2[:,elev_band], summary_data_total[list(val),:])
                predictions_validation = np.array(predictions_validation)
                predictions_validation[predictions_validation<0]=0
                r2 = np.corrcoef(predictions_validation,summary_data_total[-1])[0,1]**2
                n_ = len(predictions_validation)
                k_ = len(list(val))
                adj_r2 = 1 - ((1-r2) * (n_-1)/(n_-k_-1))
                if showOutput: print(list(val),adj_r2)
                if adj_r2 > r2adj_max:
                    best_pils = list(val)
                    r2adj_max = adj_r2
        if showOutput: print(best_pils,r2adj_max)
    else:
    ## use most repeat pillows 
        dic_ = {}
        for num in sum(best_stations, []):
            if num not in dic_:
                dic_[num] = 1
            else:
                dic_[num] += 1
        val_based = {k: v for k, v in sorted(dic_.items(), key=lambda item: item[1], reverse=True)}
        counter = 0
        val_lst = []
        for k,v in val_based.items():
            val_lst.append(k)
            predictions_validation = cross_val_loo(aso_tseries_2[:,elev_band], summary_data_total[val_lst,:])
            predictions_validation = np.array(predictions_validation)
            predictions_validation[predictions_validation<0]=0
            r2 = np.corrcoef(predictions_validation,summary_data_total[-1])[0,1]**2
            n_ = len(predictions_validation)
            k_ = len(val_lst)
            adj_r2 = 1 - ((1-r2) * (n_-1)/(n_-k_-1))
            if showOutput: print(val_lst,adj_r2)
            if adj_r2 > r2adj_max:
                best_pils = copy.deepcopy(val_lst)
                r2adj_max = adj_r2
            else:
                val_lst.remove(k)
        if showOutput:
            try: 
                print(best_pils,r2adj_max)
            except:
                pass


    return best_pils


def cross_val_loo(aso, index, add_points=None):
    """
        Runs linear regression given the stations selected from  
        cross-validation analysis.
        Input:
            aso - datarray object representing ASO basin SWE.
            index - 2D numpy array representing the snow pillow features. 
        Output:
            output - array of predictions for each cross-validated year.
    """
    output = []
    for y in range(len(aso)):
        test = []
        train = []
        trainy = []

        for i in range(len(aso)):
            if i == y:
                test.append(index[:,i])
            else:
                train.append(index[:,i])
                trainy.append(aso.values[i])

        if add_points is not None:
            for point in add_points:
                train.append(point[0])
                trainy.append(point[1])

        testy = compute_linear(np.array(train), np.array(trainy), np.array(test))

        output.extend(testy)
    return output

def impute_pillow_prediction(df_sum_total,
                             all_pils,
                             obs_data_5,
                             aso_site_name,
                             prediction_date,
                             obs_threshold = 0.50,
                             saveImputeCSV = True):
    """
        Fit a linear regression model
        Input:
            df_sum_total - summary pandas dataframe with features and labels.
            all_pils - list of all pillow ids.
            obs_data_5 - list of xarray datarrays for pillow observations.
            obs_threshold - float of fraction of missing observations for pillow removal.
            aso_site_name - python string of aso_site_name.
        Output:
            obs_data_5_ - new list of xarray datarrays for pillow observations with filled values.
            pils_removed - updated list of features.
            df_summary_impute - updated summary datatable.
    """

    ## find pillows with observations more than threshold.
    drop_bool = ((df_sum_total[all_pils].isnull().sum(axis=0) / len(df_sum_total)) < obs_threshold).values
    ## array of new pillows with observations more than threshold
    pils_removed = np.array(all_pils)[drop_bool]
    ## subset summary table to remove pillows with minimal observations.
    df_dropped_pils = df_sum_total[pils_removed]
    ## create output fpath name.
    if not os.path.exists(f'/home/rossamower/work/aso/data/mlr_prediction/{aso_site_name}/imputation/'):
        os.makedirs(f'/home/rossamower/work/aso/data/mlr_prediction/{aso_site_name}/imputation/')
        
    if prediction_date.month >= 10:
        impute_df_fpath = f'/home/rossamower/work/aso/data/mlr_prediction/{aso_site_name}/imputation/pillow_impute_threePils_wy{prediction_date.year+1}.csv'
    else:
        impute_df_fpath = f'/home/rossamower/work/aso/data/mlr_prediction/{aso_site_name}/imputation/pillow_impute_threePils_wy{prediction_date.year}.csv'
    ## if table does not exist.
    if not os.path.exists(impute_df_fpath) or (saveImputeCSV == False):

        for pil in df_dropped_pils.columns:
            df_new = pd.DataFrame(df_dropped_pils[pil])
# iterate over observations.
            for row,col in df_new.iterrows():
                val = col[pil]
                # find nans
                if np.isnan(val):
                    valid_pillows = df_dropped_pils.iloc[row][~df_dropped_pils.iloc[row].isna()].index.tolist()
                    target_id = int([i for i in range(0,len(obs_data_5)) if obs_data_5[i].name == pil][0])
                    target_da = obs_data_5[target_id]

                    first_corr = {}
                    feature_list = []
                    for feat in valid_pillows:
                        feat_id = int([i for i in range(0,len(obs_data_5)) if obs_data_5[i].name == feat][0])
                        feat_da = obs_data_5[feat_id]
                        mask_target = ~target_da.isnull()
                        mask_feat = ~feat_da.isnull()
                        target_vals = target_da.where(mask_target).where(mask_feat).values
                        feat_vals = feat_da.where(mask_target).where(mask_feat).values

                        target_vals = target_vals[~np.isnan(target_vals)]
                        feat_vals = feat_vals[~np.isnan(feat_vals)]
                        r2 = np.corrcoef(target_vals, feat_vals)[0,1]**2
                        k = 1
                        n = len(feat_vals)
                        adjr2= 1 - ((1-r2) * (n-1)/(n-k-1))

                        first_corr[feat] = adjr2

                    # find max correlated pillow.
                    best_corr_pillow = max(first_corr, key=first_corr.get)
                    best_corr_adjr2 = first_corr[best_corr_pillow]
                    feature_list.append(best_corr_pillow)
            
            
                    bestfeat_id = int([i for i in range(0,len(obs_data_5)) if obs_data_5[i].name == best_corr_pillow][0])
                    # remove pillow from list.
                    valid_pillows.remove(best_corr_pillow)
                    blah = True
                    train_df = pd.merge(target_da.to_dataframe(),obs_data_5[bestfeat_id].to_dataframe(),on = 'time')
                    second_corr = {}
                    for feat in valid_pillows:
                        feat_id = int([i for i in range(0,len(obs_data_5)) if obs_data_5[i].name == feat][0])
                        train_df_ = pd.merge(train_df,obs_data_5[feat_id].to_dataframe(),on = 'time')
                        train_df_ = train_df_[train_df_.index.year >= 2013]
                        train_df_ = train_df_.dropna(axis = 0)
                        feat_cols = train_df_.columns.to_list()
                        feat_cols.remove(pil)
                        target_vals = train_df_[pil].values
                        feat_vals = train_df_[feat_cols].values
                        r2 = np.corrcoef(target_vals, feat_vals.T)[0,1]**2
                        k = len(feat_cols)
                        n = len(feat_vals)
                        adjr2= 1 - ((1-r2) * (n-1)/(n-k-1))

                        second_corr[feat] = adjr2

                    # find max correlated pillow.
                    second_corr_pillow = max(second_corr, key=second_corr.get)
                    second_corr_adjr2 = second_corr[second_corr_pillow]
                    if second_corr_adjr2 > best_corr_adjr2:
                        feature_list.append(second_corr_pillow)

                        second_id = int([i for i in range(0,len(obs_data_5)) if obs_data_5[i].name == second_corr_pillow][0])
                        # remove pillow from list.
                        valid_pillows.remove(second_corr_pillow)
                        blah = True
                        train_df = pd.merge(train_df,obs_data_5[second_id].to_dataframe(),on = 'time')
                        third_corr = {}
                        for feat in valid_pillows:
                            feat_id = int([i for i in range(0,len(obs_data_5)) if obs_data_5[i].name == feat][0])
                            train_df_ = pd.merge(train_df,obs_data_5[feat_id].to_dataframe(),on = 'time')
                            train_df_ = train_df_[train_df_.index.year >= 2013]
                            train_df_ = train_df_.dropna(axis = 0)
                            feat_cols = train_df_.columns.to_list()
                            feat_cols.remove(pil)
                            target_vals = train_df_[pil].values
                            feat_vals = train_df_[feat_cols].values
                            r2 = np.corrcoef(target_vals, feat_vals.T)[0,1]**2
                            k = len(feat_cols)
                            n = len(feat_vals)
                            adjr2= 1 - ((1-r2) * (n-1)/(n-k-1))
                    
                            third_corr[feat] = adjr2

                        # find max correlated pillow.
                        try:
                            third_corr_pillow = max(third_corr, key=third_corr.get)
                            third_corr_adjr2 = third_corr[third_corr_pillow]
                            if third_corr_adjr2 > second_corr_adjr2:
                                feature_list.append(third_corr_pillow)
                        except:
                            pass

                    # train regression
                    final_train_df_ = obs_data_5[target_id].to_dataframe()
                    for feat in feature_list:
                        feat_id = int([i for i in range(0,len(obs_data_5)) if obs_data_5[i].name == feat][0])
                        final_train_df_ = pd.merge(final_train_df_,obs_data_5[feat_id].to_dataframe(),on = 'time')

                    final_train_df = final_train_df_[final_train_df_.index.year >= 2013]
                    final_train_df = final_train_df.dropna(axis = 0)

                    lm = linear_model.LinearRegression()
                    lm.fit(final_train_df.values[:,1:],final_train_df.values[:,0])

                    res = lm.predict(df_dropped_pils.iloc[row][feature_list].values.reshape(1,-1))
                    if res < 0:
                        res = 0
                    df_dropped_pils.loc[row, pil] = res




        df_summary_impute = copy.deepcopy(df_dropped_pils)
        pillows_ = df_dropped_pils.columns.to_list()


        df_summary_impute['time'] = df_sum_total.time.values
        df_summary_impute['aso_mean_bins_mm'] = df_sum_total.aso_mean_bins_mm.values
        df_summary_impute = df_summary_impute[['time','aso_mean_bins_mm'] + pillows_]

        # ouput table.
        if saveImputeCSV:
            df_summary_impute.to_csv(impute_df_fpath,index = False)

    # load impute table.
    if os.path.exists(impute_df_fpath):
        df_summary_impute = pd.read_csv(impute_df_fpath)
    df_summary_impute['time'] = pd.to_datetime(df_summary_impute['time'])

    ## create copy of observations.
    obs_data_5_ = copy.deepcopy(obs_data_5)
    ## fill observation data arrays with imputed values.
    for pil_id in pils_removed:
        ## create dataframe with time and pillow.
        df_impute = df_summary_impute[['time',pil_id]]
        ## identify pillow index.
        pil_idx = [i for i in range(0,len(obs_data_5_)) if obs_data_5_[i].name == pil_id][0]
        for row,col in df_impute.iterrows():
            time = col['time']
            val = col[pil_id]
            ## if values are same pass, else update value
            if (obs_data_5_[pil_idx].sel({'time':time}) == val).values == True:
                pass
            else:
                obs_data_5_[pil_idx].loc[dict(time=time)] = val
    return obs_data_5_,list(pils_removed),df_summary_impute

def regression_results(y_true, y_pred,max_resid_absolute,max_resid_percent,rmse_mm,features,date,
                      modelID,isAccum,isImpute,isMean,elev_band,model_type,QA_flag):
    """
        Get regression statistics for validation.
        Input:
            y_true - numpy array of labels.
            y_pred - numpy array of predictions.
            max_resid_absolute - max residual.
            max_resid_percent - max residual percent.
            rmse_mm - root mean squared error mm.
            features - list of pillows used in prediction.
            date - datetime date.
            modelID - python integer for uniqueID.
            isAccum - boolean for accumulation vs melt.
            isImpute - boolean for imputed predictions vs dropped nans.
            isMean - boolean for type of imputation.
            elev_band - elevation band.
            model_type - python string for model type.
        Output:
            summary_dict - python dictionary of summary.
    """



    # Regression metrics
    explained_variance=metrics.explained_variance_score(y_true, y_pred)
    mean_absolute_error=metrics.mean_absolute_error(y_true, y_pred) 
    mse=metrics.mean_squared_error(y_true, y_pred) 
    mean_squared_log_error=metrics.mean_squared_log_error(y_true, y_pred)
    median_absolute_error=metrics.median_absolute_error(y_true, y_pred)
    # r2=metrics.r2_score(y_true, y_pred)
    r2=np.corrcoef(y_true,y_pred)[0,1]**2

    summary_dict = {modelID: {'model_features':{'date':str(date),
                                                'model_type': model_type,
                                                'elevation_band':elev_band,
                                                'isAccum':isAccum,
                                                'isImpute':isImpute,
                                                'isMean':isMean,
                                                'features':features,
                                                'feat_QA_flag':QA_flag,
                                                'num_obs':len(y_true),
                                                },
                              'validation_stats':{'explained_variance': round(explained_variance,3),
                                                  'mean_squared_log_error':round(mean_squared_log_error,3),
                                                  'r2':round(r2,3),
                                                  'MAE':round(mean_absolute_error,3),
                                                  'MSE':round(mse,3),
                                                  'RMSE':round(np.sqrt(mse),3),
                                                  'RMSE_MM':round(rmse_mm,3),
                                                  'max_resid_absolute':round(max_resid_absolute,3),
                                                  'max_resid_percent':round(max_resid_percent,3),
                                                  },
                                'prediction':{'mm':[],
                                             'acre_ft':[]},
                              }
    }
    return summary_dict


def residual_comparison(y,yhat,ax1,ax2,plt,mask_val = 50,showPlot = True):
    """
        Compares manually downloaded and API data for snow pillows.
        Input:
            y - xarray of mean swe with date dimension.
            yhat - numpy array of predicted mean swe.
            mask_val - integer for mask.
        Output:
            plots of residuals.
    """
    
    percent_resid = 100*((y - yhat ) / y)
    swe_msk = y >= mask_val
    resid = y-yhat

    if showPlot:
        # fig,ax = plt.subplots(1,2,figsize = (12,6))
        im = ax1.scatter(percent_resid.date,resid,c = y,cmap = 'inferno', label = 'mean swe > 50mm')
        ax1.set_title(f'Residuals',fontweight = 'bold')
        ax1.set_ylabel('Residual [mm]\n mean swe - predicted',fontweight = 'bold')
        ax1.set_xlabel('Year',fontweight = 'bold')
        limit_vals_resid = resid.values
        limit_vals_resid = limit_vals_resid[~np.isnan(limit_vals_resid)]
        ax1.set_ylim(-1*(np.abs(limit_vals_resid).max()+10),
                   np.abs(limit_vals_resid).max()+10)
        plt.colorbar(im,ax=ax1,label = 'ASO Mean SWE [mm]')
        for tick in ax1.get_xticklabels():
            tick.set_rotation(45)

        im = ax2.scatter(percent_resid.date.where(swe_msk),percent_resid.where(swe_msk),c = y, cmap = 'inferno',label = 'mean swe > 50mm')
        ax2.set_title(f'Residual %\nfor SWE > {mask_val}mm',fontweight = 'bold')
        ax2.set_ylabel('Residual [%]',fontweight = 'bold')
        ax2.set_xlabel('Year',fontweight = 'bold')
        limit_vals_percent = percent_resid.where(swe_msk).values
        limit_vals_percent = limit_vals_percent[~np.isnan(limit_vals_percent)]
        ax2.set_ylim(-1*(np.abs(limit_vals_percent).max()+10),
                   (np.abs(limit_vals_percent).max()+10))
        plt.colorbar(im,ax=ax2,label = 'ASO Mean SWE [mm]')
        for tick in ax2.get_xticklabels():
            tick.set_rotation(45)

        # plt.tight_layout()
        # plt.show()
        return np.abs(limit_vals_resid).max(),np.abs(limit_vals_percent).max()
    else:
        try:
            limit_vals_resid = resid.where(swe_msk).values
            limit_vals_resid = limit_vals_resid[~np.isnan(limit_vals_resid)]
            limit_vals_percent = percent_resid.where(swe_msk).values
            limit_vals_percent = limit_vals_percent[~np.isnan(limit_vals_percent)]
            return np.abs(limit_vals_resid).max(),np.abs(limit_vals_percent).max()
        except:
            limit_vals_resid = resid.values
            limit_vals_resid = limit_vals_resid[~np.isnan(limit_vals_resid)]
            return np.abs(limit_vals_resid).max(),np.nan

def create_directory(fpath):
    """
    Create directory if doesn't exist.
    """
    if not os.path.exists(fpath):
        os.mkdir(fpath)


def run_daily_prediction(df_train,df_test,selected_pils,modelID,conversion = 0,
                         area_m2 = None,outdir = None,add_zeroASO = True):
    """
    Run daily prediction with selected features.
    Input:
      df_train - pandas training dataframe. 
      df_test - pandas testing dataframe. 
      selected_pils - python list of selected pillows during cross-validation.
      conversion - python integer indicating conversion values.
                    0 - no conversion (ie milimeters).
                    1 - inches.
                    2 - acre-feet
      area_m2 - basin area (m2).
      modelID - python integer for unique model ID.
    
    Output:
      yhat - daily prediction.
    """
    try:
        df_train = df_train.rename(columns = {'aso_swe':'aso_mean_bins_mm'})
    except:
        pass
    
    # milimeters
    if conversion == 0:
        conv_val = 1 
    # milimeters to inches
    elif conversion == 1:
        conv_val =  0.0393701
    # acre-ft
    elif conversion == 2:
        mm_m = 0.001 # mm to meters
        m3_acreft = 0.000810714 # m3 to acre feet
        conv_val = area_m2 * mm_m * m3_acreft

    all_cols = copy.deepcopy(selected_pils)
    all_cols.append('aso_mean_bins_mm')

    if add_zeroASO:
        new_row_index = np.datetime64('1900-01-01')
        new_row_values = {'aso_mean_bins_mm':0.0}
        for i in selected_pils:
            new_row_values[i] = 0.0
        # Add the new row using loc
        df_train.loc[new_row_index] = new_row_values

    # df_nan = df_train[all_cols]
    # df_nan = df_nan.dropna(axis = 0)

    # if add_zeroASO:
    #     df_nan.loc[len(df_nan)] = 0.0

    df_nan = copy.deepcopy(df_train)
    df_nan = df_nan.reset_index()
    df_nan = df_nan[all_cols]
    df_nan = df_nan.dropna(axis = 0)


    lm = linear_model.LinearRegression()
    lm1 = linear_model.LinearRegression()
    # create prediction table.
    lm1.fit(df_nan[selected_pils].values,df_nan['aso_mean_bins_mm'].values)
    yhat_train = lm1.predict(df_nan[selected_pils].values)
    
    # values converted from mm to inches.
    lm.fit(df_nan[selected_pils].values * conv_val,df_nan['aso_mean_bins_mm'].values * conv_val)
    if outdir is not None:
        pickle.dump(lm, open(f'{outdir}model_{modelID}.pkl', 'wb'))
    yhat = lm.predict(df_test[selected_pils].values * conv_val)
    if yhat < 0:
        yhat = 0

    df_nan['yhat'] = yhat_train
    df_nan['resid'] = df_nan['aso_mean_bins_mm'] - df_nan['yhat']
    df_nan['date'] = df_train.index.values
    df_nan['model'] = 'train'
    df_nan = df_nan.rename(columns = {'aso_mean_bins_mm':'aso'})
    df_nan = df_nan[['model','date','aso'] + selected_pils + ['yhat','resid']]
    yhat_test = lm1.predict(df_test[selected_pils].values)

    # add test to dataframe.
    new_row_index = df_nan.index[-1] + 1
    new_row_values = {'model':'test',
                     'date':str(df_test['time'].values[0])[0:10],
                     'aso':np.nan}
    for i in selected_pils:
        new_row_values[i] = float(df_test[i].values[0])
    new_row_values['yhat'] = float(yhat_test[0])
    new_row_values['resid'] = np.nan
    # Add the new row using loc
    df_nan.loc[new_row_index] = new_row_values
    df_nan['date'] = pd.to_datetime(df_nan['date'])

    
    return yhat,df_nan,yhat_test

def create_metadata_sheet():
    """
    Create metadata sheet.
    Input:
    
    Output:
      df - df_sheet.
    """
    index = ["Model Type","Training Infer NaNs","Date","Prediction QA","<7k","7k-8k","8k-9k","9k-10k","10k-11k","11k-12k",">12k","Total"]
    text = ["Type of machine learning model used in predictions. MLR = Multiple Linear Regression.",
        "Filling of missing information. drop NaNs = missing values are dropped. predict NaNs = missing values are filled using predicted values based on MLR of three most correlated pillows from the historic cleaned QA dataset.",
        "Date of prediction.",
        "Automated pillow QA flag for today's downloaded pillow data based on three checks: missing data (NaN), change is SWE from yesterday (dswe/dz) is greater/less than historic max/min dswe/dz for that pillow, and today's SWE value is outside of 95% confidence of predicted value.\nQA=1 (most conservative): Pillows will be dropped if any of the flags are raised.\nQA=2 (mid conservative): Pillows will be dropped if nan or historic dwe/dz flag is raised.\nQA=3 (most liberal): Pillows will be dropped only if NaN.",
        "Prediction of mean SWE (acre-feet) for area (~420,366 acres) with elevation less than 7000 ft in the basin.",
        "Prediction of mean SWE (acre-feet) for area (~134,296 acres) with elevation between 7000-8000 ft in the basin.",
        "Prediction of mean SWE (acre-feet) for area (~125,655 acres) with elevation between 8000-9000 ft in the basin.",
        "Prediction of mean SWE (acre-feet) for area (~127,414 acres) with elevation between 9000-10,000 ft in the basin.",
        "Prediction of mean SWE (acre-feet) for area (~101,731 acres) with elevation between 10,000-11,000 ft in the basin.",
        "Prediction of mean SWE (acre-feet) for area (~74,344 acres) with elevation between 11,000-12,000 ft in the basin.",
        "Prediction of mean SWE (acre-feet) for area (~19,051 acres) with elevation greater than 12,000 ft in the basin.",
        "Prediction of mean SWE (acre-feet) for total area (~1,002,860 acres) in the basin.",
       ]
    return pd.DataFrame(data = text,index = index,columns = ['Description'])



def create_prediction_sheet(summary_dict_all, ModelID, labels_from_yaml,unit):
    """
    Build a one-row DataFrame using elevation bin labels from config.
    
    Parameters
    ----------
    summary_dict_all : dict
        Your summary dictionary keyed by ModelID and offsets.
    ModelID : int
        The "total" record id; bins are expected at ModelID - N .. ModelID - 1.
    labels_from_yaml : list[str]
        Elevation bin labels from the region YAML. May include 'Total' at the end.
    unit: str
        Units of daily prediction [mm or acre_ft]

    Returns
    -------
    pd.DataFrame
        One-row sheet with metadata + columns for each elevation bin + 'Total'.
    """
    # Handle YAMLs that include 'Total' as a label—remove it for bin mapping
    labels = list(labels_from_yaml)
    has_total_in_yaml = False
    if labels and labels[-1].strip().lower() == "total":
        has_total_in_yaml = True
        labels = labels[:-1]

    # Pull common metadata once
    mf = summary_dict_all[ModelID]['model_features']
    imputation = 'predict NaNs' if mf.get('isImpute') else 'drop NaNs'

    # Map labels to offsets: oldest label -> ModelID - len(labels), ..., last -> ModelID - 1
    n = len(labels)
    row = {
        'Date': mf['date'],
        'Model Type': mf['model_type'],
        'Training Infer NaNs': imputation,
        'Prediction QA': mf['feat_QA_flag'],
    }

    # Fill each bin column
    for i, lab in enumerate(labels):
#        offset = (i + 1) - n  # yields -n, -n+1, ..., -1
        offset = i  - n  # yields -n, -n+1, ..., -1
        key = ModelID + offset
        row[lab] = int(summary_dict_all[key]['prediction'][unit][0])

    # Total: prefer the provided total at ModelID; else compute as sum
    try:
        total_val = int(summary_dict_all[ModelID]['prediction'][unit][0])
    except Exception:
        total_val = int(sum(row[lab] for lab in labels))

    row['Total'] = total_val

    # If YAML explicitly listed 'Total' (for column ordering expectations),
    # keep the final order as: metadata -> labels -> Total
    ordered_cols = ['Date', 'Model Type', 'Training Infer NaNs', 'Prediction QA', *labels, 'Total']
    return pd.DataFrame([row], columns=ordered_cols)

def create_pillow_sheet(summary_dict_all, ModelID, labels_from_yaml):
    """
    Build a one-row DataFrame using elevation bin labels from config.
    
    Parameters
    ----------
    summary_dict_all : dict
        Your summary dictionary keyed by ModelID and offsets.
    ModelID : int
        The "total" record id; bins are expected at ModelID - N .. ModelID - 1.
    labels_from_yaml : list[str]
        Elevation bin labels from the region YAML. May include 'Total' at the end.
    unit: str
        Units of daily prediction [mm or acre_ft]

    Returns
    -------
    pd.DataFrame
        One-row sheet with metadata + columns for each elevation bin + 'Total'.
    """
    # Handle YAMLs that include 'Total' as a label—remove it for bin mapping
    labels = list(labels_from_yaml)
    has_total_in_yaml = False
    if labels and labels[-1].strip().lower() == "total":
        has_total_in_yaml = True
        labels = labels[:-1]

    # Pull common metadata once
    mf = summary_dict_all[ModelID]['model_features']
    imputation = 'predict NaNs' if mf.get('isImpute') else 'drop NaNs'

    # Map labels to offsets: oldest label -> ModelID - len(labels), ..., last -> ModelID - 1
    n = len(labels)
    row = {
        'Date': mf['date'],
        'Model Type': mf['model_type'],
        'Training Infer NaNs': imputation,
        'Prediction QA': mf['feat_QA_flag'],
    }

    # Fill each bin column
    for i, lab in enumerate(labels):
        # offset = (i + 1) - n  # yields -n, -n+1, ..., -1
        offset = i - n  # yields -n, -n+1, ..., -1
        key = ModelID + offset
        s = ", ".join(summary_dict_all[key]['model_features']['features'])
        row[lab] = s
 
    # Total: prefer the provided total at ModelID; else compute as sum
    # try:
    total_val = s = ", ".join(summary_dict_all[key+1]['model_features']['features'])

    row['Total'] = total_val

    # If YAML explicitly listed 'Total' (for column ordering expectations),
    # keep the final order as: metadata -> labels -> Total
    ordered_cols = ['Date', 'Model Type', 'Training Infer NaNs', 'Prediction QA', *labels, 'Total']
    return pd.DataFrame([row], columns=ordered_cols)




def create_cmap_hex_color(val,cmap_,vmin_,vmax_):
    """
    Create sheet.
    Input:
      val - float representing RMSE. 
      cmap - string for colormap.
      vmin - minimum of colormap.
      vmax - maximum of colormap. 
    
    Output:
      hex string.
    """
    cmap = cm.get_cmap(cmap_)
    norm = mcolors.Normalize(vmin=vmin_, vmax=vmax_)


    # Convert to hex colors
    hex_color = mcolors.to_hex(cmap(norm(val)))

    return hex_color

def create_rmse_sheet(summary_dict_all,ModelID,cmap_='RdYlGn_r',vmin_=50,vmax_=200,isCA = False):

    """
    Create sheet.
    Input:
      summary_dict_all - summary dictionary. 
      ModelID - python diction. 
    
    Output:
      df - df_sheet.
    """
    
    if summary_dict_all[ModelID]['model_features']['isImpute']:
        imputation = 'predict NaNs'
    else:
        imputation = 'drop NaNs'

    if isCA:
        df = pd.DataFrame({
                  'Date':[summary_dict_all[ModelID]['model_features']['date']],
                  'Model Type':[summary_dict_all[ModelID]['model_features']['model_type']],
                  'Training Infer NaNs':[imputation],
                  'Prediction QA':[summary_dict_all[ModelID]['model_features']['feat_QA_flag']],
                  '<7k':[create_cmap_hex_color(summary_dict_all[ModelID-7]['validation_stats']['RMSE_MM'],cmap_,vmin_,vmax_)],
                  '7k-8k':[create_cmap_hex_color(summary_dict_all[ModelID-6]['validation_stats']['RMSE_MM'],cmap_,vmin_,vmax_)],
                  '8k-9k':[create_cmap_hex_color(summary_dict_all[ModelID-5]['validation_stats']['RMSE_MM'],cmap_,vmin_,vmax_)],
                  '9k-10k':[create_cmap_hex_color(summary_dict_all[ModelID-4]['validation_stats']['RMSE_MM'],cmap_,vmin_,vmax_)],
                  '10k-11k':[create_cmap_hex_color(summary_dict_all[ModelID-3]['validation_stats']['RMSE_MM'],cmap_,vmin_,vmax_)],
                  '11k-12k':[create_cmap_hex_color(summary_dict_all[ModelID-2]['validation_stats']['RMSE_MM'],cmap_,vmin_,vmax_)],
                  '>12k':[create_cmap_hex_color(summary_dict_all[ModelID-1]['validation_stats']['RMSE_MM'],cmap_,vmin_,vmax_)],
                  'Total':[create_cmap_hex_color(summary_dict_all[ModelID]['validation_stats']['RMSE_MM'],cmap_,vmin_,vmax_)]
                  }
                  )
    else:
        df = pd.DataFrame({
                  'Date':[summary_dict_all[ModelID]['model_features']['date']],
                  'Model Type':[summary_dict_all[ModelID]['model_features']['model_type']],
                  'Training Infer NaNs':[imputation],
                  'Prediction QA':[summary_dict_all[ModelID]['model_features']['feat_QA_flag']],
                  '<9k':[create_cmap_hex_color(summary_dict_all[ModelID-5]['validation_stats']['RMSE_MM'],cmap_,vmin_,vmax_)],
                  '9k-10k':[create_cmap_hex_color(summary_dict_all[ModelID-4]['validation_stats']['RMSE_MM'],cmap_,vmin_,vmax_)],
                  '10k-11k':[create_cmap_hex_color(summary_dict_all[ModelID-3]['validation_stats']['RMSE_MM'],cmap_,vmin_,vmax_)],
                  '11k-12k':[create_cmap_hex_color(summary_dict_all[ModelID-2]['validation_stats']['RMSE_MM'],cmap_,vmin_,vmax_)],
                  '>12k':[create_cmap_hex_color(summary_dict_all[ModelID-1]['validation_stats']['RMSE_MM'],cmap_,vmin_,vmax_)],
                  'Total':[create_cmap_hex_color(summary_dict_all[ModelID]['validation_stats']['RMSE_MM'],cmap_,vmin_,vmax_)]
                  }
                  )
    return df



