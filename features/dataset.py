import config
import pandas as pd
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

    data = drop_unused_df_feature(data)

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

    df = pd.concat([X_df, y_df], axis=1)

    return df