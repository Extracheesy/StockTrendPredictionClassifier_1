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
from tools import format_df
from corr import get_corr_matrix
from predict import predict_df_before_tuning
from predict import predict_df_before_tuning_one_pred
from predict import predict_df_val_tuned_param
from predict import predict_tuning_param

# Mute sklearn warnings
from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)
simplefilter(action='ignore', category=DeprecationWarning)
simplefilter(action='ignore', category=UserWarning)

if (os.path.isdir(config.TRACES_DIR) == False):
    print("new traces directory: ", config.TRACES_DIR)
    os.mkdir(config.TRACES_DIR)

if (os.path.isdir(config.RESULTS_DIR) == False):
    print("new results directory: ", config.RESULTS_DIR)
    os.mkdir(config.RESULTS_DIR)

LIST_TICKER_DJI = get_df_DJI()

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

    df = filter_df_date_year(df, config.START_YEAR)

    if(config.PLOT_PRICE == True):
        plot_price(df, tic, OUT_DIR)

    if (config.CORR_MATRIX == True):
        get_corr_matrix(df, tic, OUT_DIR)

    if (config.PREDICT_BEFORE_TUNING == True):
        predict_df_before_tuning(df, tic, OUT_DIR)

    if (config.PREDICT_BEFORE_TUNING_ONE_PRED == True):
        predict_df_before_tuning_one_pred(df, tic, OUT_DIR)

    if (config.PREDICT_TUNING_PARAM == True):
        predict_tuning_param(df, tic, OUT_DIR)

    if (config.PREDICT_VALID_WITH_PARAM == True):
        predict_df_val_tuned_param(df, tic, OUT_DIR)

    if (config.PREDICT_TEST_SET_WITH_PARAM == True):
        run_validation_set_with_tuned_param(df, tic, OUT_DIR)




