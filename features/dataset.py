import config
import pandas as pd
import numpy as np
from collections import Counter
from sklearn.datasets import make_classification
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import ADASYN
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
from matplotlib import pyplot
from numpy import where

from tools import drop_unused_df_feature

def balance_df_dataset(df, target):
    # input DataFrame
    # X →Independent Variable in DataFrame\
    # y →dependent Variable in Pandas DataFrame format

    data = df.copy()

    #data = drop_unused_df_feature(data)

    #print("balance dataset:")
    #print("data len: ", len(data[target]))
    #print("count of 1:", data[target].sum())
    #print("count of 0:", len(data[target]) - data[target].sum())
    #print("balance dataset...")

    X = data.drop(target, axis=1)
    X_columns = X.columns
    y = data[target]

    X = X.to_numpy()
    y = y.to_numpy()

    if(config.SMOTE == True):
        oversampling = SMOTE()
    else:
        oversampling = ADASYN()

    X, y = oversampling.fit_resample(X, y)

    X_df = pd.DataFrame(X, columns = X_columns)
    y_df = pd.DataFrame(y, columns=[target])

    #print("data len: ", len(y_df[target]))
    #print("count of 1:", y_df[target].sum())
    #print("count of 0:", len(y_df[target]) - data[target].sum())

    df = pd.concat([X_df, y_df], axis=1)

    return df

def get_pivot_df_best_param(df,param1,param2):

    lst_param1 = []
    lst_param2 = []

    table_0 = pd.pivot_table(df, values=['prediction', 'recall', 'f1', 'accuracy'], index=[param1, param2],
                             aggfunc={'prediction': np.mean,
                                      'recall': np.mean,
                                      'f1': np.mean,
                                      'accuracy': np.mean})

    # Get optimum value for param and param2, using prediction
    temp = table_0[table_0['prediction'] == table_0['prediction'].max()]
    lst_param1.append(temp.index[0][0])
    lst_param2.append(temp.index[0][1])

    temp = table_0[table_0['recall'] == table_0['recall'].max()]
    lst_param1.append(temp.index[0][0])
    lst_param2.append(temp.index[0][1])

    temp = table_0[table_0['f1'] == table_0['f1'].max()]
    lst_param1.append(temp.index[0][0])
    lst_param2.append(temp.index[0][1])

    temp = table_0[table_0['accuracy'] == table_0['accuracy'].max()]
    lst_param1.append(temp.index[0][0])
    lst_param2.append(temp.index[0][1])

    table_1 = pd.pivot_table(df, values=['prediction', 'recall', 'f1', 'accuracy'], index=[param1],
                             aggfunc={'prediction': np.mean,
                                     'recall': np.mean,
                                     'f1': np.mean,
                                     'accuracy': np.mean})

    # Get optimum value for param and param2, using prediction
    temp = table_1[table_1['prediction'] == table_1['prediction'].max()]
    lst_param1.append(temp.index[0])

    temp = table_1[table_1['recall'] == table_1['recall'].max()]
    lst_param1.append(temp.index[0])

    temp = table_1[table_1['f1'] == table_1['f1'].max()]
    lst_param1.append(temp.index[0])

    temp = table_1[table_1['accuracy'] == table_1['accuracy'].max()]
    lst_param1.append(temp.index[0])

    table_2 = pd.pivot_table(df, values=['prediction', 'recall', 'f1', 'accuracy'], index=[param2],
                             aggfunc={'prediction': np.mean,
                                      'recall': np.mean,
                                      'f1': np.mean,
                                      'accuracy': np.mean})

    # Get optimum value for param and param2, using prediction
    temp = table_2[table_2['prediction'] == table_2['prediction'].max()]
    lst_param2.append(temp.index[0])

    temp = table_2[table_2['recall'] == table_2['recall'].max()]
    lst_param2.append(temp.index[0])

    temp = table_2[table_2['f1'] == table_2['f1'].max()]
    lst_param2.append(temp.index[0])

    temp = table_2[table_2['accuracy'] == table_2['accuracy'].max()]
    lst_param2.append(temp.index[0])

    #lst_param1 = list(set(lst_param1))
    #lst_param2 = list(set(lst_param2))

    return lst_param1, lst_param2
