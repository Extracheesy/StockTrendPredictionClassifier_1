import os
import numpy as np
import pandas as pd
import math

import config

from xgboost import XGBRegressor
from xgboost import XGBClassifier
from xgb import pred_xgboost

from tools import do_scaling
from tools import add_lags
from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score

def get_rmse(a, b):
    """
    Comp RMSE. a and b can be lists.
    Returns a scalar.
    """
    return math.sqrt(np.mean((np.array(a) - np.array(b)) ** 2))

def get_mae(a, b):
    """
    Comp mean absolute error e_t = E[|a_t - b_t|]. a and b can be lists.
    Returns a vector of len = len(a) = len(b)
    """
    return np.mean(abs(np.array(a) - np.array(b)))

def get_mape(y_true, y_pred):
    """
    Compute mean absolute percentage error (MAPE)
    """
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def get_accuracy_trend(y_test, pred):
    y_pred = pd.DataFrame(pred)
    y_test = y_test.reset_index(drop=True)
    len_y_test = len(y_test)
    len_y_pred = len(y_pred)

    # y_test_target_raw = (y_test.shift(-1) / y_test) - 1
    y_test_target_raw = y_test.shift(-1) - y_test
    y_test_target_raw[y_test_target_raw > 0] = 1
    y_test_target_raw[y_test_target_raw <= 0] = 0

    y_test_target_raw.drop([len_y_test - 1], axis=0, inplace=True)

    # y_pred_target_raw = (y_pred.shift(-1) / y_pred) - 1
    y_pred_target_raw = y_pred.shift(-1) - y_pred
    y_pred_target_raw[y_pred_target_raw > 0] = 1
    y_pred_target_raw[y_pred_target_raw <= 0] = 0

    y_pred_target_raw.drop([len_y_pred - 1], axis=0, inplace=True)

    len_y_pred = len(y_pred_target_raw)

    accuracy = round(accuracy_score(y_test_target_raw, y_pred_target_raw, normalize=False) / len_y_pred, 6)

    # print("==========> accuracy: ",accuracy)

    return accuracy

def get_mov_avg_std(df, col, N):
    """
    Given a dataframe, get mean and std dev at timestep t using values from t-1, t-2, ..., t-N.
    Inputs
        df         : dataframe. Can be of any length.
        col        : name of the column you want to calculate mean and std dev
        N          : get mean and std dev at timestep t using values from t-1, t-2, ..., t-N
    Outputs
        df_out     : same as df but with additional column containing mean and std dev
    """
    mean_list = df[col].rolling(window=N, min_periods=1).mean()  # len(mean_list) = len(df)
    std_list = df[col].rolling(window=N, min_periods=1).std()  # first value will be NaN, because normalized by N-1

    # Add one timestep to the predictions
    mean_list = np.concatenate((np.array([np.nan]), np.array(mean_list[:-1])))
    std_list = np.concatenate((np.array([np.nan]), np.array(std_list[:-1])))

    # Append mean_list to df
    df_out = df.copy()
    df_out[col + '_mean'] = mean_list
    df_out[col + '_std'] = std_list

    return df_out

def get_error_metrics(df,
                      train_size,
                      N,
                      H,
                      seed=100,
                      n_estimators=100,
                      max_depth=3,
                      learning_rate=0.1,
                      min_child_weight=1,
                      subsample=1,
                      colsample_bytree=1,
                      colsample_bylevel=1,
                      gamma=0):
    """
    Given a series consisting of both train+validation, do predictions of forecast horizon H on the validation set,
    at H/2 intervals.
    Inputs
        df                 : train + val dataframe. len(df) = train_size + val_size
        train_size         : size of train set
        N                  : for feature at day t, we use lags from t-1, t-2, ..., t-N as features
        H                  : forecast horizon
        seed               : model seed
        n_estimators       : number of boosted trees to fit
        max_depth          : maximum tree depth for base learners
        learning_rate      : boosting learning rate (xgb’s “eta”)
        min_child_weight   : minimum sum of instance weight(hessian) needed in a child
        subsample          : subsample ratio of the training instance
        colsample_bytree   : subsample ratio of columns when constructing each tree
        colsample_bylevel  : subsample ratio of columns for each split, in each level
        gamma              :

    Outputs
        mean of rmse, mean of mape, mean of mae, dictionary of predictions
    """
    rmse_list = []  # root mean square error
    mape_list = []  # mean absolute percentage error
    mae_list = []  # mean absolute error
    accuracy_list = []  # accuracy absolute error
    preds_dict = {}

    if config.ADD_LAGS == "adj_close":
        # Get mean and std dev at timestamp t using values from t-1, ..., t-N
        df = get_mov_avg_std(df, config.ADD_LAGS, N)
        # Do scaling
        df = do_scaling(df, N)

    # Get list of features
    """
    df_feature =  pd.read_csv('DF_FEATURE_LIST.csv')
    feature_ex = df_feature['attr'].tolist()
    features = df_feature['attr'].tolist()  # features contain all features, including adj_close_lags
    for n in range(N, 0, -1):
        if config.ADD_LAGS == "adj_close":
            features.append("adj_close_scaled_lag_" + str(n))
        else:
            features.append(config.ADD_LAGS + "_lag_" + str(n))
    """

    df_feature = pd.read_csv('DF_FEATURE_LIST.csv')
    features = df_feature['attr'].tolist()
    feature_ex = df_feature['attr'].tolist()
    for i in range(train_size, len(df) - H + 1, int(H / 2)):
        # Split into train and test
        train = df[i - train_size:i].copy()
        test = df[i:i + H].copy()

        # Drop the NaNs in train
        train.dropna(axis=0, how='any', inplace=True)

        # Split into X and y
        X_train_scaled = train[features]
        if config.ADD_LAGS == "adj_close":
            y_train_scaled = train['adj_close_scaled']
            X_test_ex_adj_close = test[features_ex_adj_close]
            y_test = test['adj_close']
            prev_vals = train[-N:]['adj_close'].to_numpy()
            prev_mean_val = test.iloc[0]['adj_close_mean']
            prev_std_val = test.iloc[0]['adj_close_std']
        else:
            y_train_scaled = train[config.ADD_LAGS]
            X_test = test[feature_ex]
            y_test = test[config.ADD_LAGS]
            prev_vals = train[-N:][config.ADD_LAGS].to_numpy()
            prev_mean_val = 0
            prev_std_val = 0


        rmse, mape, mae, accuracy, est, _ = train_pred_eval_model(X_train_scaled,
                                                                  y_train_scaled,
                                                                  X_test,
                                                                  y_test,
                                                                  N,
                                                                  H,
                                                                  prev_vals,
                                                                  prev_mean_val,
                                                                  prev_std_val,
                                                                  seed=seed,
                                                                  n_estimators=n_estimators,
                                                                  max_depth=max_depth,
                                                                  learning_rate=learning_rate,
                                                                  min_child_weight=min_child_weight,
                                                                  subsample=subsample,
                                                                  colsample_bytree=colsample_bytree,
                                                                  colsample_bylevel=colsample_bylevel,
                                                                  gamma=gamma)
        #         print("N = " + str(N) + ", i = " + str(i) + ", rmse = " + str(rmse) + ", mape = " + str(mape) + ", mae = " + str(mae))

        rmse_list.append(rmse)
        mape_list.append(mape)
        mae_list.append(mae)
        accuracy_list.append(accuracy)
        preds_dict[i] = est

    return np.mean(rmse_list), np.mean(mape_list), np.mean(mae_list), np.mean(accuracy_list), preds_dict

def train_pred_eval_model(X_train_scaled,
                          y_train_scaled,
                          X_test_ex_adj_close,
                          y_test,
                          N,
                          H,
                          prev_vals,
                          prev_mean_val,
                          prev_std_val,
                          seed=100,
                          n_estimators=100,
                          max_depth=3,
                          learning_rate=0.1,
                          min_child_weight=1,
                          subsample=1,
                          colsample_bytree=1,
                          colsample_bylevel=1,
                          gamma=0):
    '''
    Train model, do prediction, scale back to original range and do evaluation
    Use XGBoost here.
    Inputs
        X_train_scaled     : features for training. Scaled to have mean 0 and variance 1
        y_train_scaled     : target for training. Scaled to have mean 0 and variance 1
        X_test_ex_adj_close: features of the test set, excluding adj_close_scaled values
        y_test             : target for test. Actual values, not scaled.
        N                  : for feature at day t, we use lags from t-1, t-2, ..., t-N as features
        H                  : forecast horizon
        prev_vals          : numpy array. If predict at time t,
                             prev_vals will contain the N unscaled values at t-1, t-2, ..., t-N
        prev_mean_val      : the mean of the unscaled values at t-1, t-2, ..., t-N
        prev_std_val       : the std deviation of the unscaled values at t-1, t-2, ..., t-N
        seed               : model seed
        n_estimators       : number of boosted trees to fit
        max_depth          : maximum tree depth for base learners
        learning_rate      : boosting learning rate (xgb’s “eta”)
        min_child_weight   : minimum sum of instance weight(hessian) needed in a child
        subsample          : subsample ratio of the training instance
        colsample_bytree   : subsample ratio of columns when constructing each tree
        colsample_bylevel  : subsample ratio of columns for each split, in each level
        gamma              :
    Outputs
        rmse               : root mean square error of y_test and est
        mape               : mean absolute percentage error of y_test and est
        mae                : mean absolute error of y_test and est
        est                : predicted values. Same length as y_test
    '''

    if(config.ADD_LAGS == "adj_close"):
        model = XGBRegressor(objective='reg:squarederror',
                             seed=config.MODEL_SEED,
                             n_estimators=int(n_estimators),
                             max_depth=int(max_depth),
                             learning_rate=learning_rate,
                             min_child_weight=min_child_weight,
                             subsample=subsample,
                             colsample_bytree=colsample_bytree,
                             colsample_bylevel=colsample_bylevel,
                             gamma=gamma)
    else:
        model = XGBClassifier(objective='binary:logistic',
                              verbosity=0,
                              seed=config.MODEL_SEED,
                              n_estimators=int(n_estimators),
                              max_depth=int(max_depth),
                              learning_rate=learning_rate,
                              min_child_weight=min_child_weight,
                              subsample=subsample,
                              colsample_bytree=colsample_bytree,
                              colsample_bylevel=colsample_bylevel,
                              gamma=gamma)

    # Train the model
    model.fit(X_train_scaled, y_train_scaled)

    # Get predicted labels and scale back to original range
    est = pred_xgboost(model, X_test_ex_adj_close, N, H, prev_vals, prev_mean_val, prev_std_val)
    est_series = pd.Series(est)
    # Calculate RMSE, MAPE, MAE
    if(config.ADD_LAGS == "adj_close"):
        rmse = get_rmse(y_test, est)
        mape = get_mape(y_test, est)
        mae = get_mae(y_test, est)
        accuracy = get_accuracy_trend(y_test, est)
    else:
        rmse = 0
        mape = 0
        mae = 0
        accuracy = accuracy_score(y_test, est_series)

    return rmse, mape, mae, accuracy, est, model.feature_importances_

def get_error_metrics_one_pred(df,
                               train_size,
                               N,
                               H,
                               seed=100,
                               n_estimators=100,
                               max_depth=3,
                               learning_rate=0.1,
                               min_child_weight=1,
                               subsample=1,
                               colsample_bytree=1,
                               colsample_bylevel=1,
                               gamma=0):
    """
    Given a series consisting of both train+test, do one prediction of forecast horizon H on the test set.
    Inputs
        df                 : train + test dataframe. len(df) = train_size + test_size
        train_size         : size of train set
        N                  : for feature at day t, we use lags from t-1, t-2, ..., t-N as features
        H                  : forecast horizon
        seed               : model seed
        n_estimators       : number of boosted trees to fit
        max_depth          : maximum tree depth for base learners
        learning_rate      : boosting learning rate (xgb’s “eta”)
        min_child_weight   : minimum sum of instance weight(hessian) needed in a child
        subsample          : subsample ratio of the training instance
        colsample_bytree   : subsample ratio of columns when constructing each tree
        colsample_bylevel  : subsample ratio of columns for each split, in each level
        gamma              :

    Outputs
        rmse, mape, mae, predictions
    """

    if config.ADD_LAGS == "adj_close":
        # Get mean and std dev at timestamp t using values from t-1, ..., t-N
        df = get_mov_avg_std(df, config.ADD_LAGS, N)
        # Do scaling
        df = do_scaling(df, N)

    # Get list of features
    df_feature =  pd.read_csv('DF_FEATURE_LIST.csv')
    feature_ex = df_feature['Feature'].tolist()
    features = df_feature['Feature'].tolist()  # features contain all features, including adj_close_lags
    for n in range(N, 0, -1):
        if config.ADD_LAGS == "adj_close":
            features.append("adj_close_scaled_lag_" + str(n))
        else:
            features.append(config.ADD_LAGS + "_lag_" + str(n))

    # Split into train and test
    train = df[:train_size].copy()
    test = df[train_size:train_size + H].copy()

    # Drop the NaNs in train
    train.dropna(axis=0, how='any', inplace=True)

    # Split into X and y
    if config.ADD_LAGS == "adj_close":
        X_train_scaled = train[features]
        y_train_scaled = train['adj_close_scaled']
        X_test_ex_adj_close = test[features_ex_adj_close]
        y_test = test['adj_close']
        prev_vals = train[-N:]['adj_close'].to_numpy()
        prev_mean_val = test.iloc[0]['adj_close_mean']
        prev_std_val = test.iloc[0]['adj_close_std']
    else:
        X_train_scaled = train[features]
        y_train_scaled = train[config.ADD_LAGS]
        X_test_ex_adj_close = test[feature_ex]
        y_test = test[config.ADD_LAGS]
        prev_vals = train[-N:][config.ADD_LAGS].to_numpy()
        prev_mean_val = 0
        prev_std_val = 0

    rmse, mape, mae, accuracy, est, feature_importances = train_pred_eval_model(X_train_scaled,
                                                                                y_train_scaled,
                                                                                X_test_ex_adj_close,
                                                                                y_test,
                                                                                N,
                                                                                H,
                                                                                prev_vals,
                                                                                prev_mean_val,
                                                                                prev_std_val,
                                                                                seed=seed,
                                                                                n_estimators=n_estimators,
                                                                                max_depth=max_depth,
                                                                                learning_rate=learning_rate,
                                                                                min_child_weight=min_child_weight,
                                                                                subsample=subsample,
                                                                                colsample_bytree=colsample_bytree,
                                                                                colsample_bylevel=colsample_bylevel,
                                                                                gamma=gamma)

    return rmse, mape, mae, accuracy, est, feature_importances, features

def get_error_metrics_GS(df,
                         train_size,
                         N,
                         H,
                         seed=100,
                         n_estimators=100,
                         max_depth=3,
                         learning_rate=0.1,
                         min_child_weight=1,
                         subsample=1,
                         colsample_bytree=1,
                         colsample_bylevel=1,
                         gamma=0):
    """
    Given a series consisting of both train+test, do one prediction of forecast horizon H on the test set.
    Inputs
        df                 : train + test dataframe. len(df) = train_size + test_size
        train_size         : size of train set
        N                  : for feature at day t, we use lags from t-1, t-2, ..., t-N as features
        H                  : forecast horizon
        seed               : model seed
        n_estimators       : number of boosted trees to fit
        max_depth          : maximum tree depth for base learners
        learning_rate      : boosting learning rate (xgb’s “eta”)
        min_child_weight   : minimum sum of instance weight(hessian) needed in a child
        subsample          : subsample ratio of the training instance
        colsample_bytree   : subsample ratio of columns when constructing each tree
        colsample_bylevel  : subsample ratio of columns for each split, in each level
        gamma              :

    Outputs
        rmse, mape, mae, predictions
    """
    if config.ADD_LAGS == "adj_close":
        # Get mean and std dev at timestamp t using values from t-1, ..., t-N
        df = get_mov_avg_std(df, config.ADD_LAGS, N)
        # Do scaling
        df = do_scaling(df, N)

    # Get list of features
    df_feature =  pd.read_csv('DF_FEATURE_LIST.csv')
    feature_ex = df_feature['Feature'].tolist()
    features = df_feature['Feature'].tolist()  # features contain all features, including adj_close_lags

    for n in range(N, 0, -1):
        if config.ADD_LAGS == "adj_close":
            features.append("adj_close_scaled_lag_" + str(n))
        else:
            features.append(config.ADD_LAGS + "_lag_" + str(n))

    # Split into train and test
    train = df[:train_size].copy()
    test = df[train_size:train_size + H].copy()

    # Drop the NaNs in train
    train.dropna(axis=0, how='any', inplace=True)

    # Split into X and y
    if config.ADD_LAGS == "adj_close":
        X_train_scaled = train[features]
        y_train_scaled = train['adj_close_scaled']
        X_test_ex_adj_close = test[features_ex_adj_close]
        y_test = test['adj_close']
        prev_vals = train[-N:]['adj_close'].to_numpy()
        prev_mean_val = test.iloc[0]['adj_close_mean']
        prev_std_val = test.iloc[0]['adj_close_std']
    else:
        X_train_scaled = train[features]
        y_train_scaled = train[config.ADD_LAGS]
        X_test_ex_adj_close = test[feature_ex]
        y_test = test[config.ADD_LAGS]
        prev_vals = train[-N:][config.ADD_LAGS].to_numpy()
        prev_mean_val = 0
        prev_std_val = 0

    rmse, mape, mae, accuracy, est = train_pred_eval_model_GS(X_train_scaled,
                                                              y_train_scaled,
                                                              X_test_ex_adj_close,
                                                              y_test,
                                                              N,
                                                              H,
                                                              prev_vals,
                                                              prev_mean_val,
                                                              prev_std_val,
                                                              seed=seed,
                                                              n_estimators=n_estimators,
                                                              max_depth=max_depth,
                                                              learning_rate=learning_rate,
                                                              min_child_weight=min_child_weight,
                                                              subsample=subsample,
                                                              colsample_bytree=colsample_bytree,
                                                              colsample_bylevel=colsample_bylevel,
                                                              gamma=gamma)

    return rmse, mape, mae, accuracy, est


def train_pred_eval_model_GS(X_train_scaled,
                             y_train_scaled,
                             X_test_ex_adj_close,
                             y_test,
                             N,
                             H,
                             prev_vals,
                             prev_mean_val,
                             prev_std_val,
                             seed=100,
                             n_estimators=100,
                             max_depth=3,
                             learning_rate=0.1,
                             min_child_weight=1,
                             subsample=1,
                             colsample_bytree=1,
                             colsample_bylevel=1,
                             gamma=0):
    '''
    Train model, do prediction, scale back to original range and do evaluation
    Use XGBoost here.
    Inputs
        X_train_scaled     : features for training. Scaled to have mean 0 and variance 1
        y_train_scaled     : target for training. Scaled to have mean 0 and variance 1
        X_test_ex_adj_close: features of the test set, excluding adj_close_scaled values
        y_test             : target for test. Actual values, not scaled.
        N                  : for feature at day t, we use lags from t-1, t-2, ..., t-N as features
        H                  : forecast horizon
        prev_vals          : numpy array. If predict at time t,
                             prev_vals will contain the N unscaled values at t-1, t-2, ..., t-N
        prev_mean_val      : the mean of the unscaled values at t-1, t-2, ..., t-N
        prev_std_val       : the std deviation of the unscaled values at t-1, t-2, ..., t-N
        seed               : model seed
        n_estimators       : number of boosted trees to fit
        max_depth          : maximum tree depth for base learners
        learning_rate      : boosting learning rate (xgb’s “eta”)
        min_child_weight   : minimum sum of instance weight(hessian) needed in a child
        subsample          : subsample ratio of the training instance
        colsample_bytree   : subsample ratio of columns when constructing each tree
        colsample_bylevel  : subsample ratio of columns for each split, in each level
        gamma              :
    Outputs
        rmse               : root mean square error of y_test and est
        mape               : mean absolute percentage error of y_test and est
        mae                : mean absolute error of y_test and est
        est                : predicted values. Same length as y_test
    '''

    model = XGBRegressor(objective='reg:squarederror',
                         seed=config.MODEL_SEED,
                         n_estimators=int(n_estimators),
                         max_depth=int(max_depth),
                         learning_rate=learning_rate,
                         min_child_weight=min_child_weight,
                         subsample=subsample,
                         colsample_bytree=colsample_bytree,
                         colsample_bylevel=colsample_bylevel,
                         gamma=gamma)

    params = {
        'polynomialfeatures__degree': [2, 3],
        # 'selectkbest__k': [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18 ,19 ,20, 21],
    }
    model_XGB = make_pipeline(PolynomialFeatures(2, include_bias=False),
                              # SelectKBest(f_classif, k=21),
                              model)
    print("GridSearch...")
    Classifier_XGB = GridSearchCV(model_XGB, param_grid=params, cv=4)
    model = Classifier_XGB
    print("GridSearch completed")

    # Train the model
    model.fit(X_train_scaled, y_train_scaled)

    print("model :", model)
    print("best_param: ", model.best_params_)
    print("best_score: ", model.best_score_)

    # Get predicted labels and scale back to original range
    est = pred_xgboost(model, X_test_ex_adj_close, N, H, prev_vals, prev_mean_val, prev_std_val)

    # Calculate RMSE, MAPE, MAE
    rmse = get_rmse(y_test, est)
    mape = get_mape(y_test, est)
    mae = get_mae(y_test, est)
    accuracy = get_accuracy_trend(y_test, est)

    return rmse, mape, mae, accuracy, est

def get_accuracy_trends(df_y_test, df_prediction):
    trend_sum = 0

    if config.ADD_LAGS == "adj_close":
        for i in range(1,len(df_y_test), 1):
            trend_day_test = df_y_test['adj_close'].values[i] - df_y_test['adj_close'].values[0]
            if(trend_day_test >= 0):
                trend_day_test = 1
            else:
                trend_day_test = 0

            trend_day_pred = df_prediction['adj_close'].values[i] - df_prediction['adj_close'].values[0]

            if(trend_day_pred >= 0):
                trend_day_pred = 1
            else:
                trend_day_pred = 0

            if(trend_day_test == trend_day_pred):
                trend_day = 1
            else:
                trend_day = 0

            if(i == 1):
                trend_day_first = trend_day
                if(trend_day_test > 0):
                    first_trend_test_up_down  = 'up'
                else:
                    first_trend_test_up_down = 'down'
                if (trend_day_pred > 0):
                    first_trend_pred_up_down = 'up'
                else:
                    first_trend_pred_up_down = 'down'

            if(i == len(df_y_test) - 1):
                trend_day_end = trend_day
                if(trend_day_test > 0):
                    end_trend_test_up_down  = 'up'
                else:
                    end_trend_test_up_down = 'down'
                if (trend_day_pred > 0):
                    end_trend_pred_up_down = 'up'
                else:
                    end_trend_pred_up_down = 'down'

            trend_sum = trend_sum + trend_day

        trend_all_percent = round(trend_sum / len(df_y_test) * 100, 2)
    else:
        for i in range(1, len(df_y_test), 1):
            trend_day_test = df_y_test['adj_close'].values[i] - df_y_test['adj_close'].values[0]
            if (trend_day_test >= 0):
                trend_day_test = 1
            else:
                trend_day_test = 0

            trend_day_pred = df_prediction['adj_close'].values[i] - df_prediction['adj_close'].values[0]

            if (trend_day_pred >= 0):
                trend_day_pred = 1
            else:
                trend_day_pred = 0

            if (trend_day_test == trend_day_pred):
                trend_day = 1
            else:
                trend_day = 0

            if (i == 1):
                trend_day_first = trend_day
                if (trend_day_test > 0):
                    first_trend_test_up_down = 'up'
                else:
                    first_trend_test_up_down = 'down'
                if (trend_day_pred > 0):
                    first_trend_pred_up_down = 'up'
                else:
                    first_trend_pred_up_down = 'down'

            if (i == len(df_y_test) - 1):
                trend_day_end = trend_day
                if (trend_day_test > 0):
                    end_trend_test_up_down = 'up'
                else:
                    end_trend_test_up_down = 'down'
                if (trend_day_pred > 0):
                    end_trend_pred_up_down = 'up'
                else:
                    end_trend_pred_up_down = 'down'

            trend_sum = trend_sum + trend_day

        trend_all_percent = round(trend_sum / len(df_y_test) * 100, 2)

    return trend_day_first, trend_day_end, trend_all_percent, first_trend_test_up_down, first_trend_pred_up_down, end_trend_test_up_down, end_trend_pred_up_down