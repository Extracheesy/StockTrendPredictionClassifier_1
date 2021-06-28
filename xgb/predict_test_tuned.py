import os
import numpy as np
import pandas as pd

from error import get_error_metrics_one_pred
from error import get_error_metrics_GS
from error import get_accuracy_trends
from data_df import set_data_df_param
from data_df import set_bet_proba
from data_df import raw_summary_ticker

from tools import addRow

import config

def run_test_set_with_tuned_param(df, tic, OUT_DIR, df_summary):

    if(config.GENERIC_PARAM_FOR_TEST == True):
        filename = config.RESULTS_DIR + "GENERIC_final_param.csv"
    else:
        filename = OUT_DIR + "after_tuning/" + tic + "_final_param.csv"
    df_param = pd.read_csv(filename, index_col=0)

    OUT_DIR = OUT_DIR + "test_set_tuning/"
    if (os.path.isdir(OUT_DIR) == False):
        print("new results directory: ", OUT_DIR)
        os.mkdir(OUT_DIR)

    df_results = pd.DataFrame(columns=config.LIST_COLUMNS_RESULTS)

    df_param = df_param.sort_values(by=['precision'], ascending=False)
    df_param = df_param.reset_index(drop=True)

    lst_column = ['date','iter','precision','recall','f1','accuracy']
    for i in range(0, config.H, 1):
        lst_column.append("day_" + str(i))
    df_pred = pd.DataFrame(columns=lst_column)

    for pred_day in config.PRED_DAY_LIST:
        print("Do prediction on test set - pred day: ",pred_day)
        for i in range(0, 4, 1):
            n_estimators_opt = df_param['n_estimators'][i]
            max_depth_opt = df_param['max_depth'][i]
            learning_rate_opt = df_param['learning_rate'][i]
            min_child_weight_opt = df_param['min_child_weight'][i]
            subsample_opt = df_param['subsample'][i]
            colsample_bytree_opt = df_param['colsample_bytree'][i]
            colsample_bylevel_opt = df_param['colsample_bylevel'][i]
            gamma_opt = df_param['gamma'][i]

            precision, recall, f1, accuracy, est, _ = get_error_metrics_one_pred(df[pred_day - config.TRAIN_VAL_SIZE:pred_day + config.H],
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

            lst_est = []
            lst_est.append(str(pred_day))
            lst_est.append(str(i))
            lst_est.append(round(precision,2))
            lst_est.append(round(recall,2))
            lst_est.append(round(f1,2))
            lst_est.append(round(accuracy,2))
            lst_est.extend(est)
            df_pred = addRow(df_pred, lst_est)

        lst_pred = []
        lst_pred.append(str(pred_day))
        lst_pred.append("pred")
        lst_pred.append(0)
        lst_pred.append(0)
        lst_pred.append(0)
        lst_pred.append(0)

        test = df[pred_day:pred_day + config.H].copy()
        y_pred = test['target'].to_numpy()
        lst_pred.extend(y_pred)
        df_pred = addRow(df_pred, lst_pred)

    df_pred.to_csv('test_predict_tmp.csv')
    print("##############################")

    """
            val = df[config.ADD_LAGS].values[pred_day - 1]
            df_prediction = pd.DataFrame({config.ADD_LAGS: [val]})
            df_prediction_tmp = pd.DataFrame(est, columns=[config.ADD_LAGS])
            df_prediction = df_prediction.append(df_prediction_tmp)
            df_prediction = df_prediction.reset_index(drop=True)

            serie_y_test = df[pred_day - 1: pred_day + config.H][config.ADD_LAGS]

            df_y_test = pd.DataFrame(serie_y_test)
            df_y_test = df_y_test.reset_index(drop=True)

            trend_day_first, trend_day_end, trend_all_percent, first_trend_test, first_trend_pred, end_trend_test, end_trend_pred = get_accuracy_trends(df_y_test, df_prediction)

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
    """

    df_results = set_bet_proba(df_results)

    df_summary = raw_summary_ticker(df_results, df_summary, tic)

    filename = config.RESULTS_DIR + str(tic) + "_results_tuned.csv"
    df_results.to_csv(filename, index=False)

    filename = config.RESULTS_DIR + str(tic) + "_tmp_summary.csv"
    df_summary.to_csv(filename, index=False)

    return df_summary