import os
import numpy as np
import pandas as pd

import config

from collections import defaultdict
from error import get_error_metrics
from init import init_pred_day_list
from init import init_opt_param
from plot import plot_error_rate
from dataset import get_pivot_df_best_param

def tuning_param(df, tic, OUT_DIR):
    init_opt_param()
    init_pred_day_list(df)

    n_estimators_opt_param, max_depth_opt_param = get_opt_param_n_estimators_max_depth(df, tic, OUT_DIR)

    learning_rate_opt_param, min_child_weight_opt_param = get_opt_param_learning_rate_min_child_weight(df, tic, OUT_DIR)

    subsample_opt_param, gamma_opt_param = get_opt_param_subsample_gamma(df, tic, OUT_DIR)

    colsample_bytree_opt_param, colsample_bylevel_opt_param = get_opt_param_colsample_bytree_colsample_bylevel(df, tic, OUT_DIR)

    dict_param = {'n_estimators' : n_estimators_opt_param,
                  'max_depth' : max_depth_opt_param,
                  'learning_rate' : learning_rate_opt_param,
                  'min_child_weight' : min_child_weight_opt_param,
                  'subsample' : subsample_opt_param,
                  'colsample_bytree' : colsample_bytree_opt_param,
                  'colsample_bylevel' : colsample_bylevel_opt_param,
                  'gamma': gamma_opt_param}

    df_param = pd.DataFrame(dict_param)

    filename = OUT_DIR + tic + "_opt_param.csv"
    df_param.to_csv(filename)


def get_opt_param_n_estimators_max_depth(df, tic, OUT_DIR):

    print("Tuning n_estimators and max_depth:")
    for pred_day in config.PRED_DAY_LIST:
        print("Predicting on day %d, with forecast horizon H = %d" % (pred_day, config.H))

        train = df[pred_day - config.TRAIN_VAL_SIZE:pred_day - config.VAL_SIZE].copy()
        val = df[pred_day - config.VAL_SIZE:pred_day].copy()
        train_val = df[pred_day - config.TRAIN_VAL_SIZE:pred_day].copy()
        test = df[pred_day:pred_day + config.H].copy()

        if(config.MODE_DEBUG == True):
            param_label = 'n_estimators'
            param_list = [100]
            param2_label = 'max_depth'
            param2_list = [3]
        else:
            param_label = 'n_estimators'
            param_list = [50, 80, 100, 200, 500, 800, 1000]
            param2_label = 'max_depth'
            param2_list = [2, 3, 4, 5, 6, 7, 8, 9]

        error_rate = defaultdict(list)

        for param in param_list:
            for param2 in param2_list:
                prediction_mean, recall_mean, f1_mean, accuracy_mean, _ = get_error_metrics(train_val,
                                                                                            config.TRAIN_SIZE,
                                                                                            config.N,
                                                                                            config.H,
                                                                                            seed=config.MODEL_SEED,
                                                                                            n_estimators=param,
                                                                                            max_depth=param2,
                                                                                            learning_rate=config.OPT_LEARNING_RATE,
                                                                                            min_child_weight=config.OPT_MIN_CHILD_WEIGHT,
                                                                                            subsample=config.OPT_SUBSAMPLE,
                                                                                            colsample_bytree=config.OPT_COLSAMPLE_BYTREE,
                                                                                            colsample_bylevel=config.OPT_COLSAMPLE_BYLEVEL,
                                                                                            gamma=config.OPT_GAMMA)

                # Collect results
                error_rate[param_label].append(param)
                error_rate[param2_label].append(param2)
                error_rate['prediction'].append(prediction_mean)
                error_rate['recall'].append(recall_mean)
                error_rate['f1'].append(f1_mean)
                error_rate['accuracy'].append(accuracy_mean)
                if(config.PRINT_PARAM_TUNING):
                    print("n_estimator: ", param, "max_depth: ", param2, "prediction: ",prediction_mean, "recall: ",recall_mean, "f1: ",f1_mean ,"accuracy: ",accuracy_mean)

    error_rate = pd.DataFrame(error_rate)

    if(config.ERROR_RATE_DISPLAY == True):
        if (os.path.isdir(OUT_DIR + "error_rate/") == False):
            print("new directory: ", OUT_DIR + "error_rate/")
            os.mkdir(OUT_DIR + "error_rate/")

        filename = OUT_DIR + "error_rate/" + tic + "_error_rate_n_esti_max_depth.csv"
        error_rate.to_csv(filename)
        #plot_error_rate(error_rate, OUT_DIR + "error_rate/", tic + "_n_estimator_max_depth")

    n_estimators_opt_param, max_depth_opt_param = get_pivot_df_best_param(error_rate, "n_estimators", "max_depth")

    # Get optimum value for param and param2, using prediction
    temp = error_rate[error_rate['prediction'] == error_rate['prediction'].max()]
    config.OPT_N_ESTIMATORS = temp['n_estimators'].values[0]
    config.OPT_MAX_DEPTH = temp['max_depth'].values[0]

    return n_estimators_opt_param, max_depth_opt_param


def get_opt_param_learning_rate_min_child_weight(df, tic, OUT_DIR):
    print("Tuning learning_rate and min_child_weight:")
    for pred_day in config.PRED_DAY_LIST:
        print("Predicting on day %d, with forecast horizon H = %d" % (pred_day, config.H))

        train = df[pred_day - config.TRAIN_VAL_SIZE:pred_day - config.VAL_SIZE].copy()
        val = df[pred_day - config.VAL_SIZE:pred_day].copy()
        train_val = df[pred_day - config.TRAIN_VAL_SIZE:pred_day].copy()
        test = df[pred_day:pred_day + config.H].copy()

        if(config.MODE_DEBUG == True):
            param_label = 'learning_rate'
            param_list = [0.2]

            param2_label = 'min_child_weight'
            param2_list = [5]
        else:
            param_label = 'learning_rate'
            # param_list = [0.0001, 0.001, 0.005, 0.01, 0.05, 0.1, 0.2, 0.3]
            param_list = [0.001, 0.01, 0.1, 0.2, 0.4, 0.5]
    
            param2_label = 'min_child_weight'
            param2_list = range(5, 25, 5)

        error_rate = defaultdict(list)

        for param in param_list:
            for param2 in param2_list:
                prediction_mean, recall_mean, f1_mean, accuracy_mean, _ = get_error_metrics(train_val,
                                                                                            config.TRAIN_SIZE,
                                                                                            config.N,
                                                                                            config.H,
                                                                                            seed=config.MODEL_SEED,
                                                                                            n_estimators=config.OPT_N_ESTIMATORS,
                                                                                            max_depth=config.OPT_MAX_DEPTH,
                                                                                            learning_rate=param,
                                                                                            min_child_weight=param2,
                                                                                            subsample=config.OPT_SUBSAMPLE,
                                                                                            colsample_bytree=config.OPT_COLSAMPLE_BYTREE,
                                                                                            colsample_bylevel=config.OPT_COLSAMPLE_BYLEVEL,
                                                                                            gamma=config.OPT_GAMMA)

                # Collect results
                error_rate[param_label].append(param)
                error_rate[param2_label].append(param2)
                error_rate['prediction'].append(prediction_mean)
                error_rate['recall'].append(recall_mean)
                error_rate['f1'].append(f1_mean)
                error_rate['accuracy'].append(accuracy_mean)
                if (config.PRINT_PARAM_TUNING):
                    print("learning rate: ", param, "min child weight: ", param2, "prediction: ", prediction_mean, "recall: ", recall_mean, "f1: ", f1_mean, "accuracy: ", accuracy_mean)

    error_rate = pd.DataFrame(error_rate)

    if(config.ERROR_RATE_DISPLAY == True):
        if (os.path.isdir(OUT_DIR + "error_rate/") == False):
            print("new directory: ", OUT_DIR + "error_rate/")
            os.mkdir(OUT_DIR + "error_rate/")

        filename = OUT_DIR + "error_rate/" + tic + "_error_rate_learn_rate_min_child_w.csv"
        error_rate.to_csv(filename)
        #plot_error_rate(error_rate, OUT_DIR + "error_rate/", tic + "_learn_rate_min_child_w")

    learning_rate_opt_param, min_child_weight_opt_param = get_pivot_df_best_param(error_rate, 'learning_rate', 'min_child_weight')

    # Get optimum value for param and param2, using prediction
    temp = error_rate[error_rate['prediction'] == error_rate['prediction'].max()]
    config.OPT_LEARNING_RATE = temp['learning_rate'].values[0]
    config.OPT_MIN_CHILD_WEIGHT = temp['min_child_weight'].values[0]

    return learning_rate_opt_param, min_child_weight_opt_param


def get_opt_param_subsample_gamma(df, tic, OUT_DIR):
    print("Tuning subsample and gamma:")
    for pred_day in config.PRED_DAY_LIST:
        print("Predicting on day %d, with forecast horizon H = %d" % (pred_day, config.H))

        train = df[pred_day - config.TRAIN_VAL_SIZE:pred_day - config.VAL_SIZE].copy()
        val = df[pred_day - config.VAL_SIZE:pred_day].copy()
        train_val = df[pred_day - config.TRAIN_VAL_SIZE:pred_day].copy()
        test = df[pred_day:pred_day + config.H].copy()

        if (config.MODE_DEBUG == True):
            param_label = 'subsample'
            param_list = [0.4]

            param2_label = 'gamma'
            param2_list = [0.5]
        else:   
            param_label = 'subsample'
            param_list = [0, 0.2, 0.4, 0.6, 0.8, 1]
            param2_label = 'gamma'
            param2_list = [0, 0.2, 0.4, 0.6, 0.8, 1]

        error_rate = defaultdict(list)

        for param in param_list:
            for param2 in param2_list:
                prediction_mean, recall_mean, f1_mean, accuracy_mean, _ = get_error_metrics(train_val,
                                                                                            config.TRAIN_SIZE,
                                                                                            config.N,
                                                                                            config.H,
                                                                                            seed=config.MODEL_SEED,
                                                                                            n_estimators=config.OPT_N_ESTIMATORS,
                                                                                            max_depth=config.OPT_MAX_DEPTH,
                                                                                            learning_rate=config.OPT_LEARNING_RATE,
                                                                                            min_child_weight=config.OPT_MIN_CHILD_WEIGHT,
                                                                                            subsample=param,
                                                                                            colsample_bytree=config.OPT_COLSAMPLE_BYTREE,
                                                                                            colsample_bylevel=config.OPT_COLSAMPLE_BYLEVEL,
                                                                                            gamma=param2)

                # Collect results
                error_rate[param_label].append(param)
                error_rate[param2_label].append(param2)
                error_rate['prediction'].append(prediction_mean)
                error_rate['recall'].append(recall_mean)
                error_rate['f1'].append(f1_mean)
                error_rate['accuracy'].append(accuracy_mean)
                if (config.PRINT_PARAM_TUNING):
                    print("subsample: ", param, "gamma: ", param2, "prediction: ", prediction_mean, "recall: ", recall_mean, "f1: ", f1_mean, "accuracy: ", accuracy_mean)

    error_rate = pd.DataFrame(error_rate)

    if(config.ERROR_RATE_DISPLAY == True):
        if (os.path.isdir(OUT_DIR + "error_rate/") == False):
            print("new directory: ", OUT_DIR + "error_rate/")
            os.mkdir(OUT_DIR + "error_rate/")

        filename = OUT_DIR + "error_rate/" + tic + "_error_rate_subsample_gam.csv"
        error_rate.to_csv(filename)
        #plot_error_rate(error_rate, OUT_DIR + "error_rate/", tic + "_subsample_gamma")

    subsample_opt_param, gamma_opt_param = get_pivot_df_best_param(error_rate, 'subsample', 'gamma')

    # Get optimum value for param and param2, using prediction
    temp = error_rate[error_rate['prediction'] == error_rate['prediction'].max()]
    config.OPT_SUBSAMPLE = temp['subsample'].values[0]
    config.OPT_GAMMA = temp['gamma'].values[0]

    return subsample_opt_param, gamma_opt_param


def get_opt_param_colsample_bytree_colsample_bylevel(df, tic, OUT_DIR):
    print("Tuning subsample and gamma:")
    for pred_day in config.PRED_DAY_LIST:
        print("Predicting on day %d, with forecast horizon H = %d" % (pred_day, config.H))

        train = df[pred_day - config.TRAIN_VAL_SIZE:pred_day - config.VAL_SIZE].copy()
        val = df[pred_day - config.VAL_SIZE:pred_day].copy()
        train_val = df[pred_day - config.TRAIN_VAL_SIZE:pred_day].copy()
        test = df[pred_day:pred_day + config.H].copy()

        if (config.MODE_DEBUG == True):
            param_label = 'colsample_bytree'
            param_list = [0.5]

            param2_label = 'colsample_bylevel'
            param2_list = [0.5]
        else:
            param_label = 'colsample_bytree'
            param_list = [0.5, 0.8, 0.9, 1]
    
            param2_label = 'colsample_bylevel'
            param2_list = [0.5, 0.8, 0.9, 1]

        error_rate = defaultdict(list)

        for param in param_list:
            for param2 in param2_list:
                prediction_mean, recall_mean, f1_mean, accuracy_mean, _ = get_error_metrics(train_val,
                                                                                            config.TRAIN_SIZE,
                                                                                            config.N,
                                                                                            config.H,
                                                                                            seed=config.MODEL_SEED,
                                                                                            n_estimators=config.OPT_N_ESTIMATORS,
                                                                                            max_depth=config.OPT_MAX_DEPTH,
                                                                                            learning_rate=config.OPT_LEARNING_RATE,
                                                                                            min_child_weight=config.OPT_MIN_CHILD_WEIGHT,
                                                                                            subsample=config.OPT_SUBSAMPLE,
                                                                                            colsample_bytree=param,
                                                                                            colsample_bylevel=param2,
                                                                                            gamma=config.OPT_GAMMA)

                # Collect results
                error_rate[param_label].append(param)
                error_rate[param2_label].append(param2)
                error_rate['prediction'].append(prediction_mean)
                error_rate['recall'].append(recall_mean)
                error_rate['f1'].append(f1_mean)
                error_rate['accuracy'].append(accuracy_mean)
                if (config.PRINT_PARAM_TUNING):
                    print("colsample_bytree: ", param, "colsample_bylevel: ", param2, "prediction: ", prediction_mean, "recall: ", recall_mean, "f1: ", f1_mean, "accuracy: ", accuracy_mean)

    error_rate = pd.DataFrame(error_rate)

    if(config.ERROR_RATE_DISPLAY == True):
        if (os.path.isdir(OUT_DIR + "error_rate/") == False):
            print("new directory: ", OUT_DIR + "error_rate/")
            os.mkdir(OUT_DIR + "error_rate/")

        filename = OUT_DIR + "error_rate/" + tic + "_error_rate_colsample.csv"
        error_rate.to_csv(filename)
        #plot_error_rate(error_rate, OUT_DIR + "error_rate/", tic + "_colsample")

    colsample_bytree_opt_param, colsample_bylevel_opt_param = get_pivot_df_best_param(error_rate, 'colsample_bytree', 'colsample_bylevel')

    # Get optimum value for param and param2, using prediction
    temp = error_rate[error_rate['prediction'] == error_rate['prediction'].max()]
    config.OPT_COLSAMPLE_BYTREE = temp['colsample_bytree'].values[0]
    config.COLSAMPLE_BYLEVEL_OPT = temp['colsample_bylevel'].values[0]

    return colsample_bytree_opt_param, colsample_bylevel_opt_param