import config
import pandas as pd

def init_opt_param():
    config.OPT_N_ESTIMATORS = config.N_ESTIMATORS
    config.OPT_MAX_DEPTH = config.MAX_DEPTH
    config.OPT_LEARNING_RATE = config.LEARNING_RATE
    config.OPT_MIN_CHILD_WEIGHT = config.MIN_CHILD_WEIGHT
    config.OPT_SUBSAMPLE = config.SUBSAMPLE
    config.OPT_COLSAMPLE_BYTREE = config.COLSAMPLE_BYTREE
    config.OPT_COLSAMPLE_BYLEVEL = config.COLSAMPLE_BYLEVEL
    config.OPT_GAMMA = config.GAMMA


def init_pred_day_list(df):
    config.PRED_DAY_LIST = []

    #half = int(len(df) / 2)
    half = config.TRAIN_VAL_SIZE
    half_10 = int( (len(df) - half) / config.PRED_DAY_NB_ITEM)

    for iter in range(config.PRED_DAY_NB_ITEM + 1):
        if(iter == config.PRED_DAY_NB_ITEM):
            item = half + iter * half_10  - config.H
        else:
            item = half + iter * half_10
        config.PRED_DAY_LIST.append(item)

def build_df_importance_feature(df):
    df_feature = pd.DataFrame(columns = df.columns)
    df_feature.insert(0, 'results', [])
    df_feature.insert(0, 'scoring', [])
    df_feature.insert(0, 'estimator', [])

    #df_feature = df_feature.drop(config.PREDICT_TARGET, axis=1)

    return df_feature

