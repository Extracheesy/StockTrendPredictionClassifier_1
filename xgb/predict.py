import os
import numpy as np
import pandas as pd

import config

from plot import plot_preds_before_tuning
from plot import plot_preds_before_tuning_one_pred
from plot import plot_importance_features_no_tuning
from tools import add_lags
from error import get_mov_avg_std
from tools import do_scaling
from error import get_rmse
from error import get_mape
from error import get_mae
from error import get_accuracy_trend
from error import get_error_metrics_one_pred
from predict_tuned import run_validation_set_with_tuned_param
from predict_test_tuned import run_test_set_with_tuned_param
from tuning import tuning_param

def predict_df_before_tuning(df, tic, OUT_DIR):

    for pred_day in config.PRED_DAY_LIST:
        print("Predicting on day %d, date %s, with forecast horizon H = %d" % (pred_day, df.iloc[pred_day]['date'], config.H))

        train = df[pred_day - config.TRAIN_VAL_SIZE:pred_day - config.VAL_SIZE].copy()
        val = df[pred_day - config.VAL_SIZE:pred_day].copy()
        train_val = df[pred_day - config.TRAIN_VAL_SIZE:pred_day].copy()
        test = df[pred_day:pred_day + config.H].copy()

        if (config.PRINT_SHAPE == True):
            print("train.shape     = " + str(train.shape))
            print("val.shape       = " + str(val.shape))
            print("train_val.shape = " + str(train_val.shape))
            print("test.shape      = " + str(test.shape))

        print("Get error metrics on validation set before hyperparameter tuning:")
        rmse_bef_tuning, mape_bef_tuning, mae_bef_tuning, accuracy_bef_tuning, preds_dict = get_error_metrics(train_val,
                                                                                                              config.TRAIN_SIZE,
                                                                                                              config.N,
                                                                                                              config.H,
                                                                                                              seed=config.MODEL_SEED,
                                                                                                              n_estimators=config.N_ESTIMATORS,
                                                                                                              max_depth=config.MAX_DEPTH,
                                                                                                              learning_rate=config.LEARNING_RATE,
                                                                                                              min_child_weight=config.MIN_CHILD_WEIGHT,
                                                                                                              subsample=config.SUBSAMPLE,
                                                                                                              colsample_bytree=config.COLSAMPLE_BYTREE,
                                                                                                              colsample_bylevel=config.COLSAMPLE_BYLEVEL,
                                                                                                              gamma=config.GAMMA)
        print("before tuning RMSE     = %0.3f" % rmse_bef_tuning)
        print("before tuning MAPE     = %0.3f%%" % mape_bef_tuning)
        print("before tuning MAE      = %0.3f%%" % mae_bef_tuning)
        print("before tuning ACCURACY = %0.3f%%" % accuracy_bef_tuning)

        plot_preds_before_tuning(train, val, test, train_val, config.H, preds_dict, tic, OUT_DIR)


def predict_df_before_tuning_one_pred(df, tic, OUT_DIR):

    for pred_day in config.PRED_DAY_LIST:
        print("Predicting on day %d, date %s, with forecast horizon H = %d" % (pred_day, df.iloc[pred_day]['date'], config.H))

        train = df[pred_day - config.TRAIN_VAL_SIZE:pred_day - config.VAL_SIZE].copy()
        val = df[pred_day - config.VAL_SIZE:pred_day].copy()
        train_val = df[pred_day - config.TRAIN_VAL_SIZE:pred_day].copy()
        test = df[pred_day:pred_day + config.H].copy()

        if (config.PRINT_SHAPE == True):
            print("train.shape     = " + str(train.shape))
            print("val.shape       = " + str(val.shape))
            print("train_val.shape = " + str(train_val.shape))
            print("test.shape      = " + str(test.shape))

        print("Do prediction on test set:")
        test_rmse_bef_tuning, test_mape_bef_tuning, test_mae_bef_tuning, test_accuracy_bef_tuning, est, feature_importances, features = get_error_metrics_one_pred(df[pred_day - config.TRAIN_VAL_SIZE:pred_day + config.H],
                                                                                                                                                                   config.TRAIN_VAL_SIZE,
                                                                                                                                                                   config.N,
                                                                                                                                                                   config.H,
                                                                                                                                                                   seed=config.MODEL_SEED,
                                                                                                                                                                   n_estimators=config.N_ESTIMATORS,
                                                                                                                                                                   max_depth=config.MAX_DEPTH,
                                                                                                                                                                   learning_rate=config.LEARNING_RATE,
                                                                                                                                                                   min_child_weight=config.MIN_CHILD_WEIGHT,
                                                                                                                                                                   subsample=config.SUBSAMPLE,
                                                                                                                                                                   colsample_bytree=config.COLSAMPLE_BYTREE,
                                                                                                                                                                   colsample_bylevel=config.COLSAMPLE_BYLEVEL,
                                                                                                                                                                   gamma=config.GAMMA)

        print("before tuning one pred RMSE = %0.3f" % test_rmse_bef_tuning)
        print("before tuning one pred MAPE = %0.3f%%" % test_mape_bef_tuning)
        print("before tuning one pred MAE = %0.3f" % test_mae_bef_tuning)
        print("before tuning one pred ACCURACY = %0.3f" % test_accuracy_bef_tuning)

        plot_preds_before_tuning_one_pred(train, val, test, est, config.H, tic + str(pred_day), OUT_DIR)

        # View a list of the features and their importance scores
        imp = list(zip(features, feature_importances))
        imp.sort(key=lambda tup: tup[1], reverse=False)

        plot_importance_features_no_tuning(imp, tic + str(pred_day), OUT_DIR)


def predict_df_val_tuned_param(df, tic, OUT_DIR):

    run_validation_set_with_tuned_param(df, tic, OUT_DIR)

def predict_tuning_param(df, tic, OUT_DIR):

    tuning_param(df, tic, OUT_DIR)

def predict_test_set_with_tuned_param(df, tic, OUT_DIR):

    run_test_set_with_tuned_param(df, tic, OUT_DIR)