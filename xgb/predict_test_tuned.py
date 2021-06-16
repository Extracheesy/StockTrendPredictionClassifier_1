import os
import numpy as np
import pandas as pd

from error import get_error_metrics_one_pred
from error import get_error_metrics_GS
from error import get_accuracy_trends
from data_df import set_data_df_param
from data_df import set_bet_proba
from data_df import raw_summary_ticker
import config

def run_test_set_with_tuned_param(df, tic, OUT_DIR, df_summary):

    if(config.GENERIC_PARAM_FOR_TEST == True):
        filename = config.RESULTS_DIR + "GENERIC_final_param.csv"
    else:
        filename = config.RESULTS_DIR + tic + "_final_param.csv"
    df_param = pd.read_csv(filename, index_col=0)

    OUT_DIR = OUT_DIR + "test_set_tuning/"
    if (os.path.isdir(OUT_DIR) == False):
        print("new results directory: ", OUT_DIR)
        os.mkdir(OUT_DIR)

    df_results = pd.DataFrame(columns=config.LIST_COLUMNS_RESULTS)

    for i in range(0,3,1):
        if i == 0:
            df_param = df_param.sort_values(by=['rmse'], ascending=True)
            df_param = df_param.reset_index(drop=True)
            type = 'best_rmse'
        if i == 1:
            df_param = df_param.sort_values(by=['mape'], ascending=True)
            df_param = df_param.reset_index(drop=True)
            type = 'best_mape'
        if i == 2:
            df_param = df_param.sort_values(by=['accuracy'], ascending=False)
            df_param = df_param.reset_index(drop=True)
            type = 'best_accuracy'

        n_estimators_opt = df_param['n_estimators'][0]
        max_depth_opt = df_param['max_depth'][0]
        learning_rate_opt = df_param['learning_rate'][0]
        min_child_weight_opt = df_param['min_child_weight'][0]
        subsample_opt = df_param['subsample'][0]
        colsample_bytree_opt = df_param['colsample_bytree'][0]
        colsample_bylevel_opt = df_param['colsample_bylevel'][0]
        gamma_opt = df_param['gamma'][0]

        for pred_day in config.PRED_DAY_LIST:
            print("Do prediction on test set")
            test_rmse_aft_tuning, test_mape_aft_tuning, test_mae_aft_tuning, test_accuracy_bef_tuning, est, feature_importances, features = get_error_metrics_one_pred(df[pred_day - config.TRAIN_VAL_SIZE:pred_day + config.H],
                                                                                                                                                                       config.TRAIN_SIZE + config.VAL_SIZE,
                                                                                                                                                                       config.N,
                                                                                                                                                                       config.H,
                                                                                                                                                                       seed=config.MODEL_SEED,
                                                                                                                                                                       n_estimators=n_estimators_opt,
                                                                                                                                                                       max_depth=max_depth_opt,
                                                                                                                                                                       learning_rate=learning_rate_opt,
                                                                                                                                                                       min_child_weight=min_child_weight_opt,
                                                                                                                                                                       subsample=subsample_opt,
                                                                                                                                                                       colsample_bytree=colsample_bytree_opt,
                                                                                                                                                                       colsample_bylevel=colsample_bylevel_opt,
                                                                                                                                                                       gamma=gamma_opt)

            val = df[config.ADD_LAGS].values[pred_day - 1]
            df_prediction = pd.DataFrame({config.ADD_LAGS: [val]})
            df_prediction_tmp = pd.DataFrame(est, columns=[config.ADD_LAGS])
            df_prediction = df_prediction.append(df_prediction_tmp)
            df_prediction = df_prediction.reset_index(drop=True)

            serie_y_test = df[pred_day - 1: pred_day + config.H][config.ADD_LAGS]

            df_y_test = pd.DataFrame(serie_y_test)
            df_y_test = df_y_test.reset_index(drop=True)

            trend_day_first, trend_day_end, trend_all_percent, first_trend_test, first_trend_pred, end_trend_test, end_trend_pred = get_accuracy_trends(df_y_test, df_prediction)

            if(config.MODE_DEBUG == True):
                print("day: ", pred_day," prediction on test set One pred => RMSE = %0.3f" % test_rmse_aft_tuning)
                print("day: ", pred_day," prediction on test set One pred => MAPE = %0.3f%%" % test_mape_aft_tuning)
                print("day: ", pred_day," prediction on test set One pred => MAE = %0.3f" % test_mae_aft_tuning)
                print("day: ", pred_day," prediction on test set One pred => ACCURACY = %0.3f" % test_accuracy_bef_tuning)

            df_results = set_data_df_param(df_results,
                                           tic,
                                           pred_day,
                                           type,
                                           test_rmse_aft_tuning,
                                           test_mape_aft_tuning,
                                           test_mae_aft_tuning,
                                           test_accuracy_bef_tuning,
                                           trend_day_first,
                                           trend_day_end,
                                           trend_all_percent,
                                           first_trend_test,
                                           first_trend_pred,
                                           end_trend_test,
                                           end_trend_pred,
                                           n_estimators_opt,
                                           max_depth_opt,
                                           learning_rate_opt,
                                           min_child_weight_opt,
                                           subsample_opt,
                                           colsample_bytree_opt,
                                           colsample_bylevel_opt,
                                           gamma_opt)

            if(config.PREDICT_GRID_SEARCH == True):
                print("Do prediction on test set with GS:")
                test_rmse_aft_tuning, test_mape_aft_tuning, test_mae_aft_tuning, test_accuracy_bef_tuning, est = get_error_metrics_GS(df[pred_day - config.TRAIN_VAL_SIZE:pred_day + config.H],
                                                                                                                                      config.TRAIN_SIZE + config.VAL_SIZE,
                                                                                                                                      config.N,
                                                                                                                                      config.H,
                                                                                                                                      seed=config.MODEL_SEED,
                                                                                                                                      n_estimators=n_estimators_opt,
                                                                                                                                      max_depth=max_depth_opt,
                                                                                                                                      learning_rate=learning_rate_opt,
                                                                                                                                      min_child_weight=min_child_weight_opt,
                                                                                                                                      subsample=subsample_opt,
                                                                                                                                      colsample_bytree=colsample_bytree_opt,
                                                                                                                                      colsample_bylevel=colsample_bylevel_opt,
                                                                                                                                      gamma=gamma_opt)

                print("day: ", pred_day," prediction on test set with GS => RMSE = %0.3f" % test_rmse_bef_tuning)
                print("day: ", pred_day," prediction on test set with GS => MAPE = %0.3f%%" % test_mape_bef_tuning)
                print("day: ", pred_day," prediction on test set with GS => MAE = %0.3f" % test_mae_bef_tuning)
                print("day: ", pred_day," prediction on test set with GS => ACCURACY = %0.3f" % test_accuracy_bef_tuning)

    df_results = set_bet_proba(df_results)

    df_summary = raw_summary_ticker(df_results, df_summary, tic)

    filename = config.RESULTS_DIR + str(tic) + "_results_tuned.csv"
    df_results.to_csv(filename, index=False)

    filename = config.RESULTS_DIR + str(tic) + "_tmp_summary.csv"
    df_summary.to_csv(filename, index=False)

    return df_summary