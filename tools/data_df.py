import os
import numpy as np
import pandas as pd

import config
from tools import addRow

def set_data_df_param(df,
                      tic,
                      day,
                      type,
                      rmse,
                      mape,
                      mae,
                      accuracy,
                      trend_day_first,
                      trend_day_end,
                      trend_all_percent,
                      first_trend_test,
                      first_trend_pred,
                      end_trend_test,
                      end_trend_pred,
                      n_estimators,
                      max_depth,
                      learning_rate,
                      min_child_weight,
                      subsample,
                      colsample_bytree,
                      colsample_bylevel,
                      gamma):
    list_results = []

    list_results.append(day)
    list_results.append(tic)
    list_results.append(type)
    list_results.append(rmse)
    list_results.append(mape)
    list_results.append(mae)
    list_results.append(accuracy)
    list_results.append(trend_day_first)
    list_results.append(trend_day_end)
    list_results.append(trend_all_percent)
    list_results.append(first_trend_test)
    list_results.append(first_trend_pred)
    list_results.append(end_trend_test)
    list_results.append(end_trend_pred)
    list_results.append(n_estimators)
    list_results.append(max_depth)
    list_results.append(learning_rate)
    list_results.append(min_child_weight)
    list_results.append(subsample)
    list_results.append(colsample_bytree)
    list_results.append(colsample_bylevel)
    list_results.append(gamma)

    df = addRow(df, list_results)

    return df

def set_bet_proba(df):


    #df['daily_proba_strategy'] = np.where((df['first_day_trend_pred'] == 'up')  &  (df['first_day_trend_test'] == 'up'), 'win', 'loss')


    df.loc[(df['first_day_trend_pred'] == 'up') & (df['first_day_trend_test'] == 'up'), 'daily_strategy'] = 'win'
    df.loc[(df['first_day_trend_pred'] == 'up') & (df['first_day_trend_test'] == 'down'), 'daily_strategy'] = 'loss'
    df.loc[(df['first_day_trend_pred'] == 'down'), 'daily_strategy'] = 'no_bet'


    df.loc[(df['first_day_trend_pred'] == 'up')  &  (df['end_day_trend_pred'] == 'up') & (df['first_day_trend_test'] == 'up'), 'AND_strategy'] = 'win'
    df.loc[(df['first_day_trend_pred'] == 'up')  &  (df['end_day_trend_pred'] == 'up') & (df['first_day_trend_test'] == 'down'), 'AND_strategy'] = 'loss'
    df.loc[(df['first_day_trend_pred'] == 'down'), 'AND_strategy'] = 'no_bet'
    df.loc[(df['end_day_trend_pred'] == 'down'), 'AND_strategy'] = 'no_bet'

    return df

