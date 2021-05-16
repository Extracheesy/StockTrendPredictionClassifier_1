import os
import numpy as np
import pandas as pd
import math
from tools import do_scaling
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

    # Add lags up to N number of days to use as features
    df = add_lags(df, N, ['adj_close'])

    # Get mean and std dev at timestamp t using values from t-1, ..., t-N
    df = get_mov_avg_std(df, 'adj_close', N)

    # Do scaling
    df = do_scaling(df, N)

    # Get list of features
    features_ex_adj_close = [
        'year',
        'month',
        'week',
        'day',
        'dayofweek',
        'dayofyear',
        'is_month_end',
        'is_month_start',
        'is_quarter_end',
        'is_quarter_start',
        'is_year_end'
    ]
    features = features_ex_adj_close  # features contain all features, including adj_close_lags
    for n in range(N, 0, -1):
        features.append("adj_close_scaled_lag_" + str(n))

    for i in range(train_size, len(df) - H + 1, int(H / 2)):
        # Split into train and test
        train = df[i - train_size:i].copy()
        test = df[i:i + H].copy()

        # Drop the NaNs in train
        train.dropna(axis=0, how='any', inplace=True)

        # Split into X and y
        X_train_scaled = train[features]
        y_train_scaled = train['adj_close_scaled']
        X_test_ex_adj_close = test[features_ex_adj_close]
        y_test = test['adj_close']
        prev_vals = train[-N:]['adj_close'].to_numpy()
        prev_mean_val = test.iloc[0]['adj_close_mean']
        prev_std_val = test.iloc[0]['adj_close_std']

        rmse, mape, mae, accuracy, est, _ = train_pred_eval_model(X_train_scaled,
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
        #         print("N = " + str(N) + ", i = " + str(i) + ", rmse = " + str(rmse) + ", mape = " + str(mape) + ", mae = " + str(mae))

        rmse_list.append(rmse)
        mape_list.append(mape)
        mae_list.append(mae)
        accuracy_list.append(accuracy)
        preds_dict[i] = est

    return np.mean(rmse_list), np.mean(mape_list), np.mean(mae_list), np.mean(accuracy_list), preds_dict


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
    # Add lags up to N number of days to use as features
    df = add_lags(df, N, ['adj_close'])

    # Get mean and std dev at timestamp t using values from t-1, ..., t-N
    df = get_mov_avg_std(df, 'adj_close', N)

    # Do scaling
    df = do_scaling(df, N)

    # Get list of features
    features_ex_adj_close = [
        'year',
        'month',
        'week',
        'day',
        'dayofweek',
        'dayofyear',
        'is_month_end',
        'is_month_start',
        'is_quarter_end',
        'is_quarter_start',
        'is_year_end'
    ]
    features = features_ex_adj_close  # features contain all features, including adj_close_lags
    for n in range(N, 0, -1):
        features.append("adj_close_scaled_lag_" + str(n))

    # Split into train and test
    train = df[:train_size].copy()
    test = df[train_size:train_size + H].copy()

    # Drop the NaNs in train
    train.dropna(axis=0, how='any', inplace=True)

    # Split into X and y
    X_train_scaled = train[features]
    y_train_scaled = train['adj_close_scaled']
    X_test_ex_adj_close = test[features_ex_adj_close]
    y_test = test['adj_close']
    prev_vals = train[-N:]['adj_close'].to_numpy()
    prev_mean_val = test.iloc[0]['adj_close_mean']
    prev_std_val = test.iloc[0]['adj_close_std']

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