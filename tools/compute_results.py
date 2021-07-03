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

def get_trend_count(test):
    count_win = round(test.sum() / len(test),2)
    count_hold = 0
    count_lost = round((len(test) - test.sum()) / len(test),2)

    return count_win, count_hold, count_lost


def get_gross_return(df_pred, df_data):
    # 253 trade day per year
    # 0,03 => 7%
    # 0,04 => 10%
    # 0,06 => 15%
    df_pred.reset_index(drop=True, inplace=True)
    df_data.reset_index(drop=True, inplace=True)

    sum_close_day = 0
    gain_7_percent = 0
    gain_10_percent = 0
    gain_15_percent = 0

    for i in range(0, len(df_pred), 1):
        delta_day = (df_data['close'][i] - df_data['open'][i]) * 100 / df_data['open'][i]

        if(delta_day < config.STOP_LOSS) & (config.STOP_LOSS_ACTIVE == True):
            delta_day = config.STOP_LOSS

        delta_high = (df_data['high'][i] - df_data['open'][i]) * 100 / df_data['open'][i]

        sum_close_day = sum_close_day + df_pred[i] * delta_day

        if (delta_high >= 0.03):
            gain_7_percent = gain_7_percent + df_pred[i] * 0.03
        else:
            gain_7_percent = gain_7_percent + df_pred[i] * delta_day

        if (delta_high >= 0.04):
            gain_10_percent = gain_10_percent + df_pred[i] * 0.04
        else:
            gain_10_percent = gain_10_percent + df_pred[i] * delta_day

        if (delta_high >= 0.06):
            gain_15_percent = gain_15_percent + df_pred[i] * 0.06
        else:
            gain_15_percent = gain_15_percent + df_pred[i] * delta_day

        #print("i:              ",i,              " delta day =      ", round(delta_day,2),       " delta high:      ", round(delta_high,2), " sum_close_day: ", round(sum_close_day,2))
        #print("gain_7_percent: ", round(gain_7_percent,2)," gain_10_percent: ", round(gain_10_percent,2), " gain_15_percent: ", round(gain_15_percent,2))

    return round(sum_close_day,2), round(gain_7_percent,2), round(gain_10_percent,2), round(gain_15_percent,2)


def compute_df_results(df_prediction, df_raw_data,tic, OUT_DIR):

    df_results = pd.DataFrame(columns=config.LIST_COLUMNS_RESULTS)

    for pred_day in config.PRED_DAY_LIST:
        df_prediction['date'] = df_prediction['date'].astype(int)
        df_filtered = df_prediction[df_prediction['date'] == pred_day]
        df_filtered = df_filtered.drop(columns=['date', 'precision', 'recall', 'f1', 'accuracy','iter','id'])
        df_filtered = df_filtered.astype(float)
        df_transposed = df_filtered.transpose()

        lst_column = df_transposed.columns
        pred_pos = lst_column[0]
        test_pos = lst_column[len(lst_column) - 1]
        yesterday_trend_pos = lst_column[len(lst_column) - 2]
        up_win, hold, down_lost = get_strategy_results(df_transposed[pred_pos], df_transposed[test_pos])

        df_raw_data_filtered = df_raw_data[pred_day : pred_day + config.H]
        sum_close_day, gain_7_percent, gain_10_percent, gain_15_percent = get_gross_return(df_transposed[pred_pos], df_raw_data_filtered)

        new_row_lst = []
        new_row_lst.append(tic)
        new_row_lst.append(pred_day)
        new_row_lst.append("no_merge")
        new_row_lst.append(up_win)
        new_row_lst.append(hold)
        new_row_lst.append(down_lost)

        new_row_lst.append(sum_close_day)
        new_row_lst.append(gain_7_percent)
        new_row_lst.append(gain_10_percent)
        new_row_lst.append(gain_15_percent)

        df_results = addRow(df_results, new_row_lst)

        merged = df_transposed[pred_pos].copy()

        for j in range(1, len(lst_column) - 2, 1):
            merged = merge_pred_column(merged, df_transposed[lst_column[j]])
            up_win, hold, down_lost = get_strategy_results(merged, df_transposed[test_pos])

            sum_close_day, gain_7_percent, gain_10_percent, gain_15_percent = get_gross_return(merged, df_raw_data_filtered)

            new_row_lst = []
            new_row_lst.append(tic)
            new_row_lst.append(pred_day)
            new_row_lst.append("merge_" + str(j))
            new_row_lst.append(up_win)
            new_row_lst.append(hold)
            new_row_lst.append(down_lost)

            new_row_lst.append(sum_close_day)
            new_row_lst.append(gain_7_percent)
            new_row_lst.append(gain_10_percent)
            new_row_lst.append(gain_15_percent)

            df_results = addRow(df_results, new_row_lst)

        up_win, hold, down_lost = get_strategy_results(df_transposed[yesterday_trend_pos], df_transposed[test_pos])
        sum_close_day, gain_7_percent, gain_10_percent, gain_15_percent = get_gross_return(df_transposed[yesterday_trend_pos], df_raw_data_filtered)

        new_row_lst = []
        new_row_lst.append(tic)
        new_row_lst.append(pred_day)
        new_row_lst.append("yesterday_strategy")
        new_row_lst.append(up_win)
        new_row_lst.append(hold)
        new_row_lst.append(down_lost)

        new_row_lst.append(sum_close_day)
        new_row_lst.append(gain_7_percent)
        new_row_lst.append(gain_10_percent)
        new_row_lst.append(gain_15_percent)

        df_results = addRow(df_results, new_row_lst)

        up_win, hold, down_lost = get_trend_count(df_transposed[test_pos])
        sum_close_day, gain_7_percent, gain_10_percent, gain_15_percent = get_gross_return(df_transposed[test_pos], df_raw_data_filtered)

        new_row_lst = []
        new_row_lst.append(tic)
        new_row_lst.append(pred_day)
        new_row_lst.append("trend")
        new_row_lst.append(up_win)
        new_row_lst.append(hold)
        new_row_lst.append(down_lost)

        new_row_lst.append(sum_close_day)
        new_row_lst.append(gain_7_percent)
        new_row_lst.append(gain_10_percent)
        new_row_lst.append(gain_15_percent)

        df_results = addRow(df_results, new_row_lst)

    OUT_DIR = OUT_DIR + "after_tuning/"
    filename = OUT_DIR + str(tic) + "_final_results_stats.csv"
    df_results.to_csv(filename)

    filename = config.RESULTS_DIR + str(tic) + "_final_results_stats.csv"
    df_results.to_csv(filename)

    return df_results

