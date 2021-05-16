import config
import pandas as pd

def get_list_DJI():

    df_html = pd.read_html('https://en.wikipedia.org/wiki/Dow_Jones_Industrial_Average')
    df_dji = df_html[1]
    list_dji = df_dji["Symbol"]

    return list_dji


def get_df_DJI():

    if (config.READ_DATA_FILE == True):
        df_LIST_TICKER_DJI = pd.read_csv('DJI_TICKER_LIST.csv')
        index_drop = df_LIST_TICKER_DJI[df_LIST_TICKER_DJI['Symbol'] == 'DOW'].index
        df_LIST_TICKER_DJI = df_LIST_TICKER_DJI.drop(index_drop)
    else:
        LIST_TICKER_DJI = get_list_DJI()
        #df_LIST_TICKER_DJI = pd.DataFrame(LIST_TICKER_DJI, columns =['ticker'])
        df_LIST_TICKER_DJI = LIST_TICKER_DJI.to_frame()
        df_LIST_TICKER_DJI.to_csv('DJI_TICKER_LIST.csv',index = False)

    return df_LIST_TICKER_DJI