import os
import numpy as np
import pandas as pd
import config

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

    if(config.MODE_DEBUG == False):
        est_scaled = model.predict(X_test_ex_adj_close)
        return est_scaled
    else:
        if(config.ADD_LAGS == "adj_close"):
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
        else:
            for n in range(H):
                forecast_scaled = forecast[-N:]

                # Create the features dataframe
                X = X_test_ex_adj_close[n:n + 1].copy()
                for n in range(N, 0, -1):
                    X.loc[:, config.ADD_LAGS + "_lag_" + str(n)] = forecast_scaled[-n]

                # Do prediction
                est_scaled = model.predict(X)
                est_scaled = [round(value) for value in est_scaled]

                # Unscale the prediction
                forecast = np.concatenate([forecast, np.array(est_scaled).reshape(1, )])


        return forecast[-H:]