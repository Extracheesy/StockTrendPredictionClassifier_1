import os
import numpy as np
import pandas as pd

from error import get_error_metrics_one_pred
from error import get_error_metrics_GS
import config

def run_validation_set_with_tuned_param(df, tic, OUT_DIR):

    filename = OUT_DIR + tic + "_final_opt_param.csv"
    df_param = pd.read_csv(filename)

    OUT_DIR = OUT_DIR + "test_set_tuning/"
    if (os.path.isdir(OUT_DIR) == False):
        print("new results directory: ", OUT_DIR)
        os.mkdir(OUT_DIR)


    n_estimators_opt = df_param['n_estimators']
    max_depth_opt = df_param['max_depth']
    learning_rate_opt = df_param['learning_rate']
    min_child_weight_opt = df_param['min_child_weight']
    subsample_opt = df_param['subsample']
    colsample_bytree_opt = df_param['colsample_bytree']
    colsample_bylevel_opt = df_param['colsample_bylevel']
    gamma_opt = df_param['gamma']

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

        print("prediction on test set => RMSE = %0.3f" % test_rmse_aft_tuning)
        print("prediction on test set => MAPE = %0.3f%%" % test_mape_aft_tuning)
        print("prediction on test set => MAE = %0.3f" % test_mae_aft_tuning)
        print("prediction on test set => ACCURACY = %0.3f" % test_accuracy_bef_tuning)


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

        print("prediction on test set with GS => RMSE = %0.3f" % test_rmse_bef_tuning)
        print("prediction on test set with GS => MAPE = %0.3f%%" % test_mape_bef_tuning)
        print("prediction on test set with GS => MAE = %0.3f" % test_mae_bef_tuning)
        print("prediction on test set with GS => ACCURACY = %0.3f" % test_accuracy_bef_tuning)
