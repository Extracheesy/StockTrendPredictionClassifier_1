import os
import numpy as np
import pandas as pd
import pandas_datareader as web
import xgboost as xgb
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt2
from xgboost import plot_importance, plot_tree
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, GridSearchCV

import config
from scraping import get_df_DJI
from plot import plot_seasonal
from plot import plot_price
from indicators import add_indicator
from tools import filter_df_date_year
from tools import filter_df_rm_n_last_raw
from tools import format_df
from tools import drop_unused_df_feature
from tools import merge_df
from corr import get_RFECV_features
from corr import get_CORR_features
from corr import get_PCA_features
from predict import predict_df_before_tuning
from predict import predict_df_before_tuning_one_pred
from predict import predict_df_val_tuned_param
from predict import predict_tuning_param
from predict import predict_test_set_with_tuned_param
from compute_results import compute_df_results
from compute_results import read_csv_results
from dataset import balance_df_dataset

from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)
simplefilter(action='ignore', category=DeprecationWarning)
simplefilter(action='ignore', category=UserWarning)

# Mute pandas warnings
from pandas.core.common import SettingWithCopyWarning
simplefilter(action="ignore", category=SettingWithCopyWarning)



if (os.path.isdir(config.TRACES_DIR) == False):
    print("new traces directory: ", config.TRACES_DIR)
    os.mkdir(config.TRACES_DIR)

if (os.path.isdir(config.RESULTS_DIR) == False):
    print("new results directory: ", config.RESULTS_DIR)
    os.mkdir(config.RESULTS_DIR)

if (os.path.isdir(config.FEATURE_DIRECTORY) == False):
    print("new traces directory: ", config.FEATURE_DIRECTORY)
    os.mkdir(config.FEATURE_DIRECTORY)

LIST_TICKER_DJI = get_df_DJI()

df_summary = pd.DataFrame(columns=config.LIST_COLUMNS_TIC_SUMMARY)
df_results = pd.DataFrame(columns=config.LIST_COLUMNS_RESULTS)

for tic in LIST_TICKER_DJI['Symbol']:
    STOCK_NAME = tic
    OUT_DIR = config.TRACES_DIR + "out_" + STOCK_NAME + "/"

    print("Ticker: ", STOCK_NAME)

    if(os.path.isdir(OUT_DIR) == False):
        print("new directory: ", OUT_DIR)
        os.mkdir(OUT_DIR)

    if(config.READ_DATA_FILE == True):
        #df = pd.read_csv(os.path.join(config.STOCKS_DIRECTORY, STOCK_NAME.lower() + '.us.txt'), sep=',')
        df = pd.read_csv(config.STOCKS_DJI_DIRECTORY + STOCK_NAME + '.csv')
    else:
        df = web.DataReader(STOCK_NAME, data_source = 'yahoo', start = '2000-01-01')
        df['Date'] = df.index
        #df.to_csv(config.STOCKS_DJI_DIRECTORY + tic + '.csv', index = False)

    df = format_df(df)

    # save fig decomposed data
    plot_seasonal(df, tic, OUT_DIR)

    if (config.ADD_INDICATORS == True):
        df = add_indicator(df)

    if (config.FILTERS == True):
        df = filter_df_date_year(df, config.START_YEAR)
        df = filter_df_rm_n_last_raw(df, config.H)

    if(config.PLOT_PRICE == True):
        plot_price(df, tic, OUT_DIR)

    df = drop_unused_df_feature(df)

    if (config.CORR_MATRIX == True):
        df_feature = get_CORR_features(df, tic, OUT_DIR, 'XGB', 'accuracy')

    if (config.PCA_MATRIX == True):
        df = get_PCA_features(df)

    if (config.RFECV_MATRIX == True):
        #df = get_RFECV_features(df, tic, OUT_DIR, 'XGB', 'accuracy')
        #df = get_RFECV_features(df, tic, OUT_DIR, 'XGB', 'f1')
        #df_feature = get_RFECV_features(df, tic, OUT_DIR, 'XGB', 'jaccard')
        df = get_RFECV_features(df, tic, OUT_DIR, 'XGB', 'precision')
        #df_feature = get_RFECV_features(df, tic, OUT_DIR, 'XGB', 'recall')

        #df_feature = get_RFECV_features(df, tic, OUT_DIR, 'Forest', 'precision')
        #df_feature = get_RFECV_features(df, tic, OUT_DIR, 'Forest', 'recall')
        #df_feature = get_RFECV_features(df, tic, OUT_DIR, 'Forest', 'f1')
        #df_feature = get_RFECV_features(df, tic, OUT_DIR, 'Forest', 'jaccard')
        #df_feature = get_RFECV_features(df, tic, OUT_DIR, 'Forest', 'accuracy')

        #df_feature = get_RFECV_features(df, tic, OUT_DIR, 'SVC', 'precision')
        #df_feature = get_RFECV_features(df, tic, OUT_DIR, 'SVC', 'recall')
        #df_feature = get_RFECV_features(df, tic, OUT_DIR, 'SVC', 'f1')
        #df_feature = get_RFECV_features(df, tic, OUT_DIR, 'SVC', 'jaccard')
        #df_feature = get_RFECV_features(df, tic, OUT_DIR, 'SVC', 'accuracy')

    if (config.PREDICT_BEFORE_TUNING == True):
        predict_df_before_tuning(df, tic, OUT_DIR)

    if (config.PREDICT_BEFORE_TUNING_ONE_PRED == True):
        predict_df_before_tuning_one_pred(df, tic, OUT_DIR)

    if (config.PREDICT_TUNING_PARAM == True):
        # Search exhaustive list of hyper param
        predict_tuning_param(df, tic, OUT_DIR)

    if (config.PREDICT_VALID_WITH_PARAM == True):
        # Search the optimized combination of hyper param
        predict_df_val_tuned_param(df, tic, OUT_DIR)

    if (config.PREDICT_TEST_SET_WITH_PARAM == True):
        df_prediction = predict_test_set_with_tuned_param(df, tic, OUT_DIR, df_summary)
    else:
        df_prediction = read_csv_results(tic, OUT_DIR)

    if(config.COMPUTE_RESULT == True):
        df_results_tmp = compute_df_results(df_prediction, tic, OUT_DIR)
        df_results = merge_df(df_results, df_results_tmp)
        filename = config.RESULTS_DIR + "global_final_result.csv"
        df_results.to_csv(filename)


