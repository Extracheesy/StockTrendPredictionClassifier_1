import os
import numpy as np
import pandas as pd
from tools import addRow
from tools import merge_df
from compute_results import get_strategy_results

import config


def parse_data_vs_pred(tic,
                       day,
                       strategy,
                       up_win,
                       hold,
                       down_lost,
                       df_pred,
                       df_data):

    df_results = pd.DataFrame(columns=config.LIST_COLUMNS_RESULTS_DASHBOARD)





    return df_results




def compute_df_dasboard(df_prediction, df_raw_data, tic, OUT_DIR):

    df_results = pd.DataFrame(columns=config.LIST_COLUMNS_RESULTS_DASHBOARD)

    for pred_day in config.PRED_DAY_LIST:
        df_prediction['date'] = df_prediction['date'].astype(int)
        df_filtered = df_prediction[df_prediction['date'] == pred_day]
        df_filtered = df_filtered.drop(columns=['date', 'precision', 'recall', 'f1', 'accuracy','iter','id'])
        df_filtered = df_filtered.astype(float)
        df_transposed = df_filtered.transpose()

        lst_column = df_transposed.columns

        for i in range(0, len(lst_column) - 2, 1):
            pred_pos = lst_column[i]
            test_pos = lst_column[len(lst_column) - 1]

            df_strategy = df_prediction[df_prediction['date'] == pred_day]
            strategy = df_strategy['id'][i]

            up_win, hold, down_lost = get_strategy_results(df_transposed[pred_pos], df_transposed[test_pos])

            df_raw_data_filtered = df_raw_data[pred_day -1: pred_day + config.H -1]

            df_tmp_result = parse_data_vs_pred(tic,
                                               pred_day,
                                               strategy,
                                               up_win,
                                               hold,
                                               down_lost,
                                               df_transposed[pred_pos],
                                               df_raw_data_filtered)

            df_results = merge_df(df_results, df_tmp_result)






