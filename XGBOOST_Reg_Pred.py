# %% [markdown]
# ### Introduction
#
# In this kernel I use XGBRegressor from XGBoost library to predict future prices of stocks using technical indicator as features. If you are looking for an explanation of indicators (e.g. moving averages, RSI, MACD) used below, please refer to [articles on Investopedia](https://www.investopedia.com/technical-analysis-4689657) or [this notebook of mine](https://www.kaggle.com/mtszkw/analysis-and-technical-indicators-for-trading-etfs) where I introduce and visualize various technical analysis concepts.

# %% [code] {"_kg_hide-input":false,"_kg_hide-output":true}
import os
import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt2
from xgboost import plot_importance, plot_tree
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, GridSearchCV

import mplfinance as mpf

# Time series decomposition
#!pip install stldecompose
from stldecompose import decompose

# Chart drawing
import plotly as py
import plotly.io as pio
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

import pandas_datareader as web

# Mute sklearn warnings
from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)
simplefilter(action='ignore', category=DeprecationWarning)
simplefilter(action='ignore', category=UserWarning)

# Show charts when running kernel
#init_notebook_mode(connected=True)

# Change default background color for all visualizations
#layout=go.Layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(250,250,250,0.8)')
#fig = go.Figure(layout=layout)
#templated_fig = pio.to_templated(fig)
#pio.templates['my_template'] = templated_fig.layout.template
#pio.templates.default = 'my_template'

# %% [markdown]
# ### Read historical prices
#
# I read historical data frame for a chosen stock (e.g. CERN) which I am going to analyze. New York Stock Exchange dataset provides day by day price history gathered over more than 10 years. I decided to crop the time frame and start it from a year 2010 to reduce amount of data to be processed.
#
# Removing rows is then followed by reindexing the data frame to keep it clean.

SAVE_PLT = True
#READ_DATA_FILE = False
READ_DATA_FILE = True

#ORIGIN_CODE = True
ORIGIN_CODE = False

TRACES_DIR = "./traces/"
RESULTS_DIR = TRACES_DIR + "results/"

if (os.path.isdir(TRACES_DIR) == False):
    print("new traces directory: ", TRACES_DIR)
    os.mkdir(TRACES_DIR)

if (os.path.isdir(RESULTS_DIR) == False):
    print("new results directory: ", RESULTS_DIR)
    os.mkdir(RESULTS_DIR)

# %% [code]
def relative_strength_idx(df, n=14):
    close = df['Close']
    delta = close.diff()
    delta = delta[1:]
    pricesUp = delta.copy()
    pricesDown = delta.copy()
    pricesUp[pricesUp < 0] = 0
    pricesDown[pricesDown > 0] = 0
    rollUp = pricesUp.rolling(n).mean()
    rollDown = pricesDown.abs().rolling(n).mean()
    rs = rollUp / rollDown
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi

def get_list_DJI():

    df_html = pd.read_html('https://en.wikipedia.org/wiki/Dow_Jones_Industrial_Average')
    df_dji = df_html[1]
    list_dji = df_dji["Symbol"]

    return list_dji

if (READ_DATA_FILE == True):
    LIST_TICKER_DJI = pd.read_csv('DJI_TICKER_LIST.csv')
    index_drop = LIST_TICKER_DJI[LIST_TICKER_DJI['Symbol'] == 'DOW'].index
    LIST_TICKER_DJI = LIST_TICKER_DJI.drop(index_drop)
else:
    LIST_TICKER_DJI = get_list_DJI()
    #df_LIST_TICKER_DJI = pd.DataFrame(LIST_TICKER_DJI, columns =['ticker'])
    df_LIST_TICKER_DJI = LIST_TICKER_DJI.to_frame()
    df_LIST_TICKER_DJI.to_csv('DJI_TICKER_LIST.csv',index = False)

df_pred_true_DJI = pd.DataFrame()
df_score = pd.DataFrame()
df_best_param = pd.DataFrame()

for tic in LIST_TICKER_DJI['Symbol']:
    # %% [code]
    #ETF_NAME = 'CERN'
    #ETF_NAME = 'NKE'
    #ETF_NAME = 'AAPL'
    #ETF_DIRECTORY = '/kaggle/input/price-volume-data-for-all-us-stocks-etfs/Data/Stocks/'
    ETF_NAME = tic
    ETF_DIRECTORY = './Data/Stocks/'
    OUT_DIR = TRACES_DIR + "out_" + ETF_NAME + "/"

    print("Ticker: ", ETF_NAME)
    if(os.path.isdir(OUT_DIR) == False):
        print("new directory: ", OUT_DIR)
        os.mkdir(OUT_DIR)

    if(READ_DATA_FILE == True):
        df = pd.read_csv(os.path.join(ETF_DIRECTORY, ETF_NAME.lower() + '.us.txt'), sep=',')
        df['Date'] = pd.to_datetime(df['Date'])
    else:
        df = web.DataReader('AAPL', data_source = 'yahoo', start = '2000-01-01')
        df['Date'] = df.index

    df = df[(df['Date'].dt.year >= 2010)].copy()
    df.index = range(len(df))

    print(df.head())

    # %% [markdown]
    # ### OHLC Chart
    #
    # I start with drawing an OHLC (open/high/low/close) chart to get a sense of historical prices. Below OHLC I draw Volume chart which shows number of stocks traded each day. In my previous notebook (linked above) I explain importance of OHLC and Volume charts in technical analysis.

    # %% [code] {"_kg_hide-input":false}
    if(ORIGIN_CODE == True):
        fig = make_subplots(rows=2, cols=1)

        fig.add_trace(go.Ohlc(x=df.Date,
                              open=df.Open,
                              high=df.High,
                              low=df.Low,
                              close=df.Close,
                              name='Price'), row=1, col=1)

        fig.add_trace(go.Scatter(x=df.Date, y=df.Volume, name='Volume'), row=2, col=1)

        fig.update(layout_xaxis_rangeslider_visible=False)
        fig.show()
    else:
        fig = plt.figure()
        fig.set_size_inches(20, 8)

        plt.title('Close')

        plt.plot(df.Date, df.Close, linewidth=0.5)

        filename = OUT_DIR + ETF_NAME + "_1.png"
        plt.savefig(filename, dpi=500)

        df.index = range(len(df))


    # %% [markdown]
    # ### Decomposition

    # %% [code]
    df_close = df[['Date', 'Close']].copy()
    df_close = df_close.set_index('Date')
    df_close.head()

    decomp = decompose(df_close, period=365)
    fig = decomp.plot()
    fig.set_size_inches(20, 8)

    filename = OUT_DIR + ETF_NAME + "_2.png"
    plt.savefig(filename, dpi=500)

    #plt.figure(True)
    #plt.clf()



    # %% [markdown]
    # ### Technical indicators

    # %% [markdown]
    # #### Moving Averages
    #
    # I'm calculating few moving averages to be used as features: $SMA_{5}$, $SMA_{10}$, $SMA_{15}$, $SMA_{30}$ and $EMA_{9}$.

    # %% [code]
    df['EMA_9'] = df['Close'].ewm(9).mean().shift()
    df['SMA_5'] = df['Close'].rolling(5).mean().shift()
    df['SMA_10'] = df['Close'].rolling(10).mean().shift()
    df['SMA_15'] = df['Close'].rolling(15).mean().shift()
    df['SMA_30'] = df['Close'].rolling(30).mean().shift()

    if(ORIGIN_CODE == True):
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df.Date, y=df.EMA_9, name='EMA 9'))
        fig.add_trace(go.Scatter(x=df.Date, y=df.SMA_5, name='SMA 5'))
        fig.add_trace(go.Scatter(x=df.Date, y=df.SMA_10, name='SMA 10'))
        fig.add_trace(go.Scatter(x=df.Date, y=df.SMA_15, name='SMA 15'))
        fig.add_trace(go.Scatter(x=df.Date, y=df.SMA_30, name='SMA 30'))
        fig.add_trace(go.Scatter(x=df.Date, y=df.Close, name='Close', opacity=0.2))
        fig.show()
    else:
        fig = plt.figure()
        fig.set_size_inches(20, 8)

        plt.title('EMA_9 - SMA_5 - SMA_10 - SMA_15 - SMA_30 - Close')

        plt.plot(df.Date, df.EMA_9, linewidth=0.5)
        plt.plot(df.Date, df.SMA_5, linewidth=0.5)
        plt.plot(df.Date, df.SMA_10, linewidth=0.5)
        plt.plot(df.Date, df.SMA_15, linewidth=0.5)
        plt.plot(df.Date, df.SMA_30, linewidth=0.5)
        plt.plot(df.Date, df.Close, linewidth=0.5)

        filename = OUT_DIR + ETF_NAME + "_3.png"
        plt.savefig(filename, dpi=500)

    # %% [markdown]
    # #### Relative Strength Index
    #
    # I'll add RSI indicator to predict whether a stock is overbought/oversold.



    df['RSI'] = relative_strength_idx(df).fillna(0)

    if(ORIGIN_CODE == True):
        fig = go.Figure(go.Scatter(x=df.Date, y=df.RSI, name='RSI'))
        fig.show()
    else:
        fig = plt.figure()
        fig.set_size_inches(20, 8)
        plt.title('RSI')
        plt.plot(df.Date, df.RSI, linewidth=0.7)
        filename = OUT_DIR + ETF_NAME + "_4.png"
        plt.savefig(filename, dpi=500)

    # %% [markdown]
    # #### MACD

    # %% [code]
    EMA_12 = pd.Series(df['Close'].ewm(span=12, min_periods=12).mean())
    EMA_26 = pd.Series(df['Close'].ewm(span=26, min_periods=26).mean())
    df['MACD'] = pd.Series(EMA_12 - EMA_26)
    df['MACD_signal'] = pd.Series(df.MACD.ewm(span=9, min_periods=9).mean())

    if(ORIGIN_CODE == True):
        fig = make_subplots(rows=2, cols=1)
        fig.add_trace(go.Scatter(x=df.Date, y=df.Close, name='Close'), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.Date, y=EMA_12, name='EMA 12'), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.Date, y=EMA_26, name='EMA 26'), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.Date, y=df['MACD'], name='MACD'), row=2, col=1)
        fig.add_trace(go.Scatter(x=df.Date, y=df['MACD_signal'], name='Signal line'), row=2, col=1)
        fig.show()
    else:
        fig = plt.figure()
        fig.set_size_inches(20, 8)

        plt.title('EMA_12 - SMA_26 - Close')

        plt.plot(df.Date, df.Close, linewidth=0.5)
        plt.plot(df.Date, EMA_12, linewidth=0.5)
        plt.plot(df.Date, EMA_26, linewidth=0.5)
        filename = OUT_DIR + ETF_NAME + "_5.png"

        fig = plt.figure()
        fig.set_size_inches(20, 8)
        plt.title('MACD - MACD_signal')
        plt.plot(df.Date, df['MACD'], linewidth=0.5)
        plt.plot(df.Date, df['MACD_signal'], linewidth=0.5)

        filename = OUT_DIR + ETF_NAME + "_6.png"
        plt.savefig(filename, dpi=500)


    # %% [markdown]
    # ### Shift label column
    #
    # Because I want to predict the next day price, after calculating all features for day $D_{i}$, I shift Close price column by -1 rows. After doing that, for day $D_{i}$ we have features from the same timestamp e.g. $RSI_{i}$, but the price $C_{i+1}$ from upcoming day.

    # %% [code]
    df['Close'] = df['Close'].shift(-1)

    # %% [markdown]
    # ### Drop invalid samples
    #
    # Because of calculating moving averages and shifting label column, few rows will have invalid values i.e. we haven't calculated $SMA_{10}$ for the first 10 days. Moreover, after shifting Close price column, last row price is equal to 0 which is not true. Removing these samples should help.

    # %% [code]
    df = df.iloc[33:] # Because of moving averages and MACD line
    df = df[:-1]      # Because of shifting close price

    df.index = range(len(df))

    # %% [markdown]
    # Here I split stock data frame into three subsets: training ($70\%$), validation ($15\%$) and test ($15\%$) sets. I calculated split indices and create three separate frames (train_df, valid_df, test_df). All three frames have been ploted in the chart below.

    # %% [code]
    if (ORIGIN_CODE == True):
        test_size  = 0.15
        valid_size = 0.15
    else:
        test_size  = 0.01
        valid_size = 0.01

    test_split_idx  = int(df.shape[0] * (1-test_size))
    valid_split_idx = int(df.shape[0] * (1-(valid_size+test_size)))

    train_df  = df.loc[:valid_split_idx].copy()
    valid_df  = df.loc[valid_split_idx+1:test_split_idx].copy()
    test_df   = df.loc[test_split_idx+1:].copy()


    print("train_df size: ", len(train_df))
    print("valid_df size: ", len(valid_df))
    print("test_df size: ", len(test_df))

    if(ORIGIN_CODE == True):
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=train_df.Date, y=train_df.Close, name='Training'))
        fig.add_trace(go.Scatter(x=valid_df.Date, y=valid_df.Close, name='Validation'))
        fig.add_trace(go.Scatter(x=test_df.Date,  y=test_df.Close,  name='Test'))
        fig.show()
    else:
        fig = plt.figure()
        fig.set_size_inches(20, 8)

        plt.title('Training - Validation - Test')
        plt.plot(train_df.Date, train_df.Close, linewidth=0.5)
        plt.plot(valid_df.Date, valid_df.Close, linewidth=0.5)
        plt.plot(test_df.Date, test_df.Close, linewidth=0.5)

        filename = OUT_DIR + ETF_NAME + "_7.png"
        plt.savefig(filename, dpi=500)

    # %% [markdown]
    # ### Drop unnecessary columns

    # %% [code]
    if(READ_DATA_FILE == True):
        drop_cols = ['Date', 'Volume', 'Open', 'Low', 'High', 'OpenInt']
    else:
        drop_cols = ['Date', 'Volume', 'Open', 'Low', 'High']

    train_df = train_df.drop(drop_cols, 1)
    valid_df = valid_df.drop(drop_cols, 1)
    test_df  = test_df.drop(drop_cols, 1)

    # %% [markdown]
    # ### Split into features and labels

    # %% [code]
    y_train = train_df['Close'].copy()
    X_train = train_df.drop(['Close'], 1)

    y_valid = valid_df['Close'].copy()
    X_valid = valid_df.drop(['Close'], 1)

    y_test  = test_df['Close'].copy()
    X_test  = test_df.drop(['Close'], 1)

    X_train.info()

    # %% [markdown]
    # ### Fine-tune XGBoostRegressor

    # %% [code]
    # %%time

    parameters = {
        'n_estimators': [100, 300, 500, 600],
        'learning_rate': [0.001, 0.005, 0.01, 0.05],
        'max_depth': [8, 10, 12, 15],
        'gamma': [0.001, 0.005, 0.01, 0.02],
        'random_state': [42]
    }
    """
    parameters = {
        'n_estimators': [600],
        'learning_rate': [0.01],
        'max_depth': [12],
        'gamma': [0.01],
        'random_state': [42]
    }
    """

    eval_set = [(X_train, y_train), (X_valid, y_valid)]
    #model = xgb.XGBRegressor(eval_set=eval_set, objective='reg:squarederror', verbose=False)
    model = xgb.XGBRegressor(objective='reg:squarederror', verbosity=0,silent=True)
    clf = GridSearchCV(model, parameters)

    clf.fit(X_train, y_train)

    print(f'Best params: {clf.best_params_}')
    print(f'Best validation score = {clf.best_score_}')

    # %% [code]
    # %%time

    df_param = pd.DataFrame.from_dict(clf.best_params_, orient='index')
    #df_param.columns.values[0] = tic
    df_param.columns = [tic]
    df_best_param[tic] = df_param[tic]

    model = xgb.XGBRegressor(**clf.best_params_, objective='reg:squarederror')
    model.fit(X_train, y_train, eval_set=eval_set, verbose=False)

    # %% [code]
    plot_importance(model)

    # %% [markdown]
    # ### Calculate and visualize predictions

    # %% [code]
    y_pred = model.predict(X_test)
    print(f'y_true = {np.array(y_test)[:5]}')
    print(f'y_pred = {y_pred[:5]}')

    print("#####################")
    print(f'y_true = {np.array(y_test)}')
    print(f'y_pred = {y_pred}')
    df_pred_true_DJI[tic + "_true"] = y_test
    df_pred_true_DJI[tic + "_pred"] = y_pred

    # %% [code]
    print(f'mean_squared_error = {mean_squared_error(y_test, y_pred)}')
    df_score[tic + '_MSE'] = [mean_squared_error(y_test, y_pred)]
    df_score[tic + '_val'] = [clf.best_score_]

    df_pred_true_DJI.to_csv(RESULTS_DIR + tic + '_tmp_PRED.csv')
    df_score.to_csv(RESULTS_DIR + tic + '_tmp_SCORE.csv')
    df_best_param.to_csv(RESULTS_DIR + tic + '_tmp_PARAM.csv')

    # %% [code]
    predicted_prices = df.loc[test_split_idx+1:].copy()
    predicted_prices['Close'] = y_pred

    if(ORIGIN_CODE == True):
        fig = make_subplots(rows=2, cols=1)
        fig.add_trace(go.Scatter(x=df.Date, y=df.Close,
                                 name='Truth',
                                 marker_color='LightSkyBlue'), row=1, col=1)

        fig.add_trace(go.Scatter(x=predicted_prices.Date,
                                 y=predicted_prices.Close,
                                 name='Prediction',
                                 marker_color='MediumPurple'), row=1, col=1)

        fig.add_trace(go.Scatter(x=predicted_prices.Date,
                                 y=y_test,
                                 name='Truth',
                                 marker_color='LightSkyBlue',
                                 showlegend=False), row=2, col=1)

        fig.add_trace(go.Scatter(x=predicted_prices.Date,
                                 y=y_pred,
                                 name='Prediction',
                                 marker_color='MediumPurple',
                                 showlegend=False), row=2, col=1)

        fig.show()
    else:
        fig = plt.figure()
        fig.set_size_inches(20, 8)

        plt.title('df - Predict')
        plt.plot(df.Date, df.Close, linewidth=0.5)
        plt.plot(predicted_prices.Date, predicted_prices.Close, linewidth=0.5)

        filename = OUT_DIR + ETF_NAME + "_9.png"
        plt.savefig(filename, dpi=500)

        fig = plt.figure()
        fig.set_size_inches(20, 8)

        plt.title('y_test - y_pred')
        plt.plot(predicted_prices.Date, y_test, linewidth=0.5)
        plt.plot(predicted_prices.Date, y_pred, linewidth=0.5)

        filename = OUT_DIR + ETF_NAME + "_8.png"
        plt.savefig(filename, dpi=500)

        filename = RESULTS_DIR + ETF_NAME + "_result.png"
        plt.savefig(filename, dpi=500)

        plt.close('all')

df_pred_true_DJI.to_csv(RESULTS_DIR + 'PRED.csv')
df_score.to_csv(RESULTS_DIR + 'SCORE.csv')
df_best_param.to_csv(RESULTS_DIR + 'PARAM.csv')