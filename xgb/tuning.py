import os
import numpy as np
import pandas as pd

import config

from collections import defaultdict
from error import get_error_metrics
from init import init_opt_param
from plot import plot_error_rate

def tuning_param(df, tic, OUT_DIR):

    init_opt_param()

    n_estimators_opt_param, max_depth_opt_param = get_opt_param_n_estimators_max_depth(df, tic, OUT_DIR)
    #df_param = pd.DataFrame(list(zip(n_estimators_opt_param, max_depth_opt_param)), columns =['n_estimators', 'max_depth'])

    learning_rate_opt_param, min_child_weight_opt_param = get_opt_param_learning_rate_min_child_weight(df, tic, OUT_DIR)
    #df_param.append(pd.DataFrame(list(zip(learning_rate_opt_param, min_child_weight_opt_param)), columns=['learning_rate', 'min_child_weight']))

    subsample_opt_param, gamma_opt_param = get_opt_param_subsample_gamma(df, tic, OUT_DIR)
    #df_param.append(pd.DataFrame(list(zip(subsample_opt_param, gamma_opt_param)), columns=['subsample', 'gamma']))

    colsample_bytree_opt_param, colsample_bylevel_opt_param = get_opt_param_colsample_bytree_colsample_bylevel(df, tic, OUT_DIR)
    #df_param.append(pd.DataFrame(list(zip(colsample_bytree_opt_param, colsample_bylevel_opt_param)), columns=['colsample_bytree', 'colsample_bylevel']))

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

    for pred_day in config.PRED_DAY_LIST:
        print("Predicting on day %d, date %s, with forecast horizon H = %d" % (pred_day, df.iloc[pred_day]['date'], config.H))

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
            # param_list = range(30, 61, 1)
            param_list = [50, 80, 100, 200, 500, 800, 1000]
            param2_label = 'max_depth'
            param2_list = [2, 3, 4, 5, 6, 7, 8, 9]

        error_rate = defaultdict(list)

        for i in param_list:
            param = i
            for param2 in param2_list:
                rmse_mean, mape_mean, mae_mean, accuracy_mean, _ = get_error_metrics(train_val,
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
                error_rate['rmse'].append(rmse_mean)
                error_rate['mape'].append(mape_mean)
                error_rate['mae'].append(mae_mean)
                error_rate['accuracy'].append(accuracy_mean)

    error_rate = pd.DataFrame(error_rate)

    if(config.ERROR_RATE_DISPLAY == True):
        if (os.path.isdir(OUT_DIR + "error_rate/") == False):
            print("new directory: ", OUT_DIR + "error_rate/")
            os.mkdir(OUT_DIR + "error_rate/")

        filename = OUT_DIR + "error_rate/" + tic + "_error_rate_n_esti_max_depth.csv"
        error_rate.to_csv(filename)
        plot_error_rate(error_rate, OUT_DIR + "error_rate/", tic + "_n_estimator_max_depth")

    n_estimators_opt_param = []
    max_depth_opt_param = []

    # Get optimum value for param and param2, using RMSE
    temp = error_rate[error_rate['rmse'] == error_rate['rmse'].min()]
    config.OPT_N_ESTIMATORS = temp['n_estimators'].values[0]
    config.OPT_MAX_DEPTH = temp['max_depth'].values[0]
    n_estimators_opt_param.append(temp['n_estimators'].values[0])
    max_depth_opt_param.append(temp['max_depth'].values[0])

    # Get optimum value for param and param2, using MAPE
    temp = error_rate[error_rate['mape'] == error_rate['mape'].min()]
    n_estimators_opt_param.append(temp['n_estimators'].values[0])
    max_depth_opt_param.append(temp['max_depth'].values[0])

    # Get optimum value for param and param2, using ACCURACY
    temp = error_rate[error_rate['accuracy'] == error_rate['accuracy'].max()]
    n_estimators_opt_param.append(temp['n_estimators'].values[0])
    max_depth_opt_param.append(temp['max_depth'].values[0])

    return n_estimators_opt_param, max_depth_opt_param


def get_opt_param_learning_rate_min_child_weight(df, tic, OUT_DIR):
    for pred_day in config.PRED_DAY_LIST:
        print("Predicting on day %d, date %s, with forecast horizon H = %d" % (
        pred_day, df.iloc[pred_day]['date'], config.H))

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

        for i in range(len(param_list)):
            param = param_list[i]
            for param2 in param2_list:
                rmse_mean, mape_mean, mae_mean, accuracy_mean, _ = get_error_metrics(train_val,
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
                error_rate['rmse'].append(rmse_mean)
                error_rate['mape'].append(mape_mean)
                error_rate['mae'].append(mae_mean)
                error_rate['accuracy'].append(accuracy_mean)

    error_rate = pd.DataFrame(error_rate)

    if(config.ERROR_RATE_DISPLAY == True):
        if (os.path.isdir(OUT_DIR + "error_rate/") == False):
            print("new directory: ", OUT_DIR + "error_rate/")
            os.mkdir(OUT_DIR + "error_rate/")

        filename = OUT_DIR + "error_rate/" + tic + "_error_rate_learn_rate_min_child_w.csv"
        error_rate.to_csv(filename)
        plot_error_rate(error_rate, OUT_DIR + "error_rate/", tic + "_learn_rate_min_child_w")

    learning_rate_opt_param = []
    min_child_weight_opt_param = []

    # Get optimum value for param and param2, using RMSE
    temp = error_rate[error_rate['rmse'] == error_rate['rmse'].min()]
    config.OPT_LEARNING_RATE = temp['learning_rate'].values[0]
    config.OPT_MIN_CHILD_WEIGHT = temp['min_child_weight'].values[0]
    learning_rate_opt_param.append(temp['learning_rate'].values[0])
    min_child_weight_opt_param.append(temp['min_child_weight'].values[0])

    # Get optimum value for param and param2, using MAPE
    temp = error_rate[error_rate['mape'] == error_rate['mape'].min()]
    learning_rate_opt_param.append(temp['learning_rate'].values[0])
    min_child_weight_opt_param.append(temp['min_child_weight'].values[0])

    # Get optimum value for param and param2, using MAPE
    temp = error_rate[error_rate['accuracy'] == error_rate['accuracy'].max()]
    learning_rate_opt_param.append(temp['learning_rate'].values[0])
    min_child_weight_opt_param.append(temp['min_child_weight'].values[0])

    return learning_rate_opt_param, min_child_weight_opt_param


def get_opt_param_subsample_gamma(df, tic, OUT_DIR):
    for pred_day in config.PRED_DAY_LIST:
        print("Predicting on day %d, date %s, with forecast horizon H = %d" % (
        pred_day, df.iloc[pred_day]['date'], config.H))

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
            param_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    
            param2_label = 'gamma'
            param2_list = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]

        error_rate = defaultdict(list)

        for i in range(len(param_list)):
            param = param_list[i]
            for param2 in param2_list:
                rmse_mean, mape_mean, mae_mean, accuracy_mean, _ = get_error_metrics(train_val,
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
                error_rate['rmse'].append(rmse_mean)
                error_rate['mape'].append(mape_mean)
                error_rate['mae'].append(mae_mean)
                error_rate['accuracy'].append(accuracy_mean)

    error_rate = pd.DataFrame(error_rate)

    if(config.ERROR_RATE_DISPLAY == True):
        if (os.path.isdir(OUT_DIR + "error_rate/") == False):
            print("new directory: ", OUT_DIR + "error_rate/")
            os.mkdir(OUT_DIR + "error_rate/")

        filename = OUT_DIR + "error_rate/" + tic + "_error_rate_subsample_gam.csv"
        error_rate.to_csv(filename)
        plot_error_rate(error_rate, OUT_DIR + "error_rate/", tic + "_subsample_gamma")



    subsample_opt_param = []
    gamma_opt_param = []

    # Get optimum value for param and param2, using RMSE
    temp = error_rate[error_rate['rmse'] == error_rate['rmse'].min()]
    config.OPT_SUBSAMPLE = temp['subsample'].values[0]
    config.OPT_GAMMA = temp['gamma'].values[0]
    subsample_opt_param.append(temp['subsample'].values[0])
    gamma_opt_param.append(temp['gamma'].values[0])

    # Get optimum value for param and param2, using MAPE
    temp = error_rate[error_rate['mape'] == error_rate['mape'].min()]
    subsample_opt_param.append(temp['subsample'].values[0])
    gamma_opt_param.append(temp['gamma'].values[0])

    # Get optimum value for param and param2, using ACCURACY
    temp = error_rate[error_rate['accuracy'] == error_rate['accuracy'].max()]
    subsample_opt_param.append(temp['subsample'].values[0])
    gamma_opt_param.append(temp['gamma'].values[0])

    return subsample_opt_param, gamma_opt_param


def get_opt_param_colsample_bytree_colsample_bylevel(df, tic, OUT_DIR):
    for pred_day in config.PRED_DAY_LIST:
        print("Predicting on day %d, date %s, with forecast horizon H = %d" % (
        pred_day, df.iloc[pred_day]['date'], config.H))

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
            param2_list = [0.5, 0.6, 0.7, 0.8, 0.9, 1]

        error_rate = defaultdict(list)

        for i in param_list:
            param = i
            for param2 in param2_list:
                rmse_mean, mape_mean, mae_mean, accuracy_mean, _ = get_error_metrics(train_val,
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
                error_rate['rmse'].append(rmse_mean)
                error_rate['mape'].append(mape_mean)
                error_rate['mae'].append(mae_mean)
                error_rate['accuracy'].append(accuracy_mean)

    error_rate = pd.DataFrame(error_rate)

    if(config.ERROR_RATE_DISPLAY == True):
        if (os.path.isdir(OUT_DIR + "error_rate/") == False):
            print("new directory: ", OUT_DIR + "error_rate/")
            os.mkdir(OUT_DIR + "error_rate/")

        filename = OUT_DIR + "error_rate/" + tic + "_error_rate_colsample.csv"
        error_rate.to_csv(filename)
        plot_error_rate(error_rate, OUT_DIR + "error_rate/", tic + "_colsample")

    colsample_bytree_opt_param = []
    colsample_bylevel_opt_param = []

    # Get optimum value for param and param2, using RMSE
    temp = error_rate[error_rate['rmse'] == error_rate['rmse'].min()]
    config.OPT_COLSAMPLE_BYTREE = temp['colsample_bytree'].values[0]
    config.COLSAMPLE_BYLEVEL_OPT = temp['colsample_bylevel'].values[0]
    colsample_bytree_opt_param.append(temp['colsample_bytree'].values[0])
    colsample_bylevel_opt_param.append(temp['colsample_bylevel'].values[0])

    # Get optimum value for param and param2, using MAPE
    temp = error_rate[error_rate['mape'] == error_rate['mape'].min()]
    colsample_bytree_opt_param.append(temp['colsample_bytree'].values[0])
    colsample_bylevel_opt_param.append(temp['colsample_bylevel'].values[0])

    # Get optimum value for param and param2, using ACCURACY
    temp = error_rate[error_rate['accuracy'] == error_rate['accuracy'].max()]
    colsample_bytree_opt_param.append(temp['colsample_bytree'].values[0])
    colsample_bylevel_opt_param.append(temp['colsample_bylevel'].values[0])

    return colsample_bytree_opt_param, colsample_bylevel_opt_param