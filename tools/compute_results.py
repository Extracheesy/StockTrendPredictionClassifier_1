import os
import numpy as np
import pandas as pd
from tools import addRow

import config

def read_csv_results(tic, OUT_DIR):
    OUT_DIR = OUT_DIR + "after_tuning/"
    filename = OUT_DIR + str(tic) + "_raw_results_tuned.csv"
    df_prediction = pd.read_csv(filename)

    return df_prediction


def merge_pred_column(merged, merged_with):
    count_results = []
    for i in range(0, len(merged), 1):
        if (merged[i] == merged_with[i]):
            if(merged[i] == 1):
                res = 1
            else:
                res = 0
        else:
            res = 0
        count_results.append(res)

    df_result = pd.DataFrame(count_results, columns =['merged'])
    return df_result.squeeze()


def get_strategy_results(pred, test):
    count_results = []
    for i in range(0, len(pred), 1):
        if (pred[i] == test[i]):
            if(pred[i] == 1):
                res = 'win'
            else:
                res = 'hold'
        else:
            if(test[i] == 0):
                res = 'lost'
            else:
                res = 'hold'
        count_results.append(res)

    count_win = round(count_results.count('win') / len(count_results),2)
    count_hold = round(count_results.count('hold') / len(count_results),2)
    count_lost = round(count_results.count('lost') / len(count_results),2)

    return count_win, count_hold, count_lost


def compute_df_results(df_prediction, tic, OUT_DIR):

    df_results = pd.DataFrame(columns=config.LIST_COLUMNS_RESULTS)

    for pred_day in config.PRED_DAY_LIST:
        df_filtered = df_prediction[df_prediction['date'] == pred_day]
        df_filtered = df_filtered.drop(columns=['date', 'precision', 'recall', 'f1', 'accuracy','iter','id'])

        df_transposed = df_filtered.transpose()

        lst_column = df_transposed.columns
        pred_pos = lst_column[0]
        test_pos = lst_column[len(lst_column) - 1]
        yesterday_trend_pos = lst_column[len(lst_column) - 2]
        up_win, hold, down_lost = get_strategy_results(df_transposed[pred_pos], df_transposed[test_pos])

        new_row_lst = []
        new_row_lst.append(pred_day)
        new_row_lst.append("no_merge")
        new_row_lst.append(up_win)
        new_row_lst.append(hold)
        new_row_lst.append(down_lost)
        df_results = addRow(df_results, new_row_lst)


        merged = df_transposed[pred_pos].copy()

        for j in range(1, len(lst_column) - 2, 1):
            merged = merge_pred_column(merged, df_transposed[lst_column[j]])
            up_win, hold, down_lost = get_strategy_results(merged, df_transposed[test_pos])

            new_row_lst = []
            new_row_lst.append(pred_day)
            new_row_lst.append("merge_" + str(j))
            new_row_lst.append(up_win)
            new_row_lst.append(hold)
            new_row_lst.append(down_lost)
            df_results = addRow(df_results, new_row_lst)

        up_win, hold, down_lost = get_strategy_results(df_transposed[yesterday_trend_pos], df_transposed[test_pos])

        new_row_lst = []
        new_row_lst.append(pred_day)
        new_row_lst.append("yesterday_trend")
        new_row_lst.append(up_win)
        new_row_lst.append(hold)
        new_row_lst.append(down_lost)
        df_results = addRow(df_results, new_row_lst)

    OUT_DIR = OUT_DIR + "after_tuning/"
    filename = OUT_DIR + str(tic) + "_final_results_stats.csv"
    df_results.to_csv(filename)

    filename = config.RESULTS_DIR + str(tic) + "_final_results_stats.csv"
    df_results.to_csv(filename)

