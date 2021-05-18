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
from xgboost import XGBRegressor

def pred_xgboost(model, X_test_ex_adj_close, N, H, prev_vals, prev_mean_val, prev_std_val):
    """
    Do recursive forecasting using xgboost
    Inputs
        model              : the xgboost model
        X_test_ex_adj_close: features of the test set, excluding adj_close_scaled values
        N                  : for feature at day t, we use lags from t-1, t-2, ..., t-N as features
        H                  : forecast horizon
        prev_vals          : numpy array. If predict at time t,
                             prev_vals will contain the N unscaled values at t-1, t-2, ..., t-N
        prev_mean_val      : the mean of the unscaled values at t-1, t-2, ..., t-N
        prev_std_val       : the std deviation of the unscaled values at t-1, t-2, ..., t-N
    Outputs
        Times series of predictions. Numpy array of shape (H,). This is unscaled.
    """
    forecast = prev_vals.copy()

    for n in range(H):
        forecast_scaled = (forecast[-N:] - prev_mean_val) / prev_std_val

        # Create the features dataframe
        X = X_test_ex_adj_close[n:n + 1].copy()
        for n in range(N, 0, -1):
            X.loc[:, "adj_close_scaled_lag_" + str(n)] = forecast_scaled[-n]

        # Do prediction
        est_scaled = model.predict(X)

        # Unscale the prediction
        forecast = np.concatenate([forecast,
                                   np.array((est_scaled * prev_std_val) + prev_mean_val).reshape(1, )])

        # Comp. new mean and std
        prev_mean_val = np.mean(forecast[-N:])
        prev_std_val = np.std(forecast[-N:])

    return forecast[-H:]


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

    model = XGBRegressor(objective='reg:squarederror',
                         seed=config.MODEL_SEED,
                         n_estimators=n_estimators,
                         max_depth=max_depth,
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

    # Calculate RMSE, MAPE, MAE
    rmse = get_rmse(y_test, est)
    mape = get_mape(y_test, est)
    mae = get_mae(y_test, est)
    accuracy = get_accuracy_trend(y_test, est)

    return rmse, mape, mae, accuracy, est, model.feature_importances_

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
    # df = add_lags(df, N, ['adj_close'])

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
    # df = add_lags(df, N, ['adj_close'])

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
