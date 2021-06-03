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

def raw_summary_ticker(df, df_summary, tic):

    df_tic = df[df['tic'] == tic]
    list_results = []


    list_results.append(tic)

    iter_pred = df_tic['day'].count()
    list_results.append(iter_pred)

    rmse = pd.to_numeric(df_tic['rmse'], errors = 'coerce')
    rmse_avg = round(rmse.mean(),2)
    list_results.append(rmse_avg)

    mape = pd.to_numeric(df_tic['mape'], errors = 'coerce')
    mape_avg = round(mape.mean(),2)
    list_results.append(mape_avg)

    mae = pd.to_numeric(df_tic['mae'], errors = 'coerce')
    mae_avg = round(mae.mean(),2)
    list_results.append(mae_avg)

    accuracy = pd.to_numeric(df_tic['accuracy'], errors = 'coerce')
    accuracy_avg = round(accuracy.mean(),2)
    list_results.append(accuracy_avg)

    trend_day_first = pd.to_numeric(df_tic['trend_day_first'], errors = 'coerce')
    trend_day_first_accuracy =  round(trend_day_first.sum() / iter_pred * 100,2)
    list_results.append(trend_day_first_accuracy)

    trend_day_end = pd.to_numeric(df_tic['trend_day_end'], errors = 'coerce')
    trend_day_end_accuracy =  round(trend_day_end.sum() / iter_pred * 100,2)
    list_results.append(trend_day_end_accuracy)

    trend_all_percent = pd.to_numeric(df_tic['trend_all_percent'], errors = 'coerce')
    trend_all_accuracy =  round(trend_all_percent.sum() / iter_pred, 2)
    list_results.append(trend_all_accuracy)


    daily_strategy = df_tic.groupby(by=['daily_strategy']).count()
    daily_strategy_win =  round(daily_strategy['day']['win'] / iter_pred * 100, 2)
    list_results.append(daily_strategy_win)

    daily_strategy_no_bet =  round(daily_strategy['day']['no_bet'] / iter_pred * 100, 2)
    list_results.append(daily_strategy_no_bet)

    daily_strategy_loss =  round(daily_strategy['day']['loss'] / iter_pred * 100, 2)
    list_results.append(daily_strategy_loss)


    AND_strategy = df_tic.groupby(by=['AND_strategy']).count()
    daily_and_trend_strategy_win =  round(AND_strategy['day']['win'] / iter_pred * 100, 2)
    list_results.append(daily_and_trend_strategy_win)

    daily_and_trend_strategy_no_bet =  round(AND_strategy['day']['no_bet'] / iter_pred * 100, 2)
    list_results.append(daily_and_trend_strategy_no_bet)

    daily_and_trend_strategy_loss =  round(AND_strategy['day']['loss'] / iter_pred * 100, 2)
    list_results.append(daily_and_trend_strategy_loss)


    first_day_trend_test = df_tic.groupby(by=['first_day_trend_test']).count()
    first_day_trend_test_up =  round(first_day_trend_test['day']['up'] / iter_pred * 100, 2)
    list_results.append(first_day_trend_test_up)

    first_day_trend_pred = df_tic.groupby(by=['first_day_trend_pred']).count()
    first_day_trend_pred_up =  round(first_day_trend_pred['day']['up'] / iter_pred * 100, 2)
    list_results.append(first_day_trend_pred_up)

    end_day_trend_test  = df_tic.groupby(by=['end_day_trend_test']).count()
    end_day_trend_test_up =  round(end_day_trend_test['day']['up'] / iter_pred * 100, 2)
    list_results.append(end_day_trend_test_up)

    end_day_trend_pred  = df_tic.groupby(by=['end_day_trend_pred']).count()
    end_day_trend_pred_up =  round(end_day_trend_pred['day']['up'] / iter_pred * 100, 2)
    list_results.append(end_day_trend_pred_up)

    df_summary = addRow(df_summary, list_results)

    return df_summary