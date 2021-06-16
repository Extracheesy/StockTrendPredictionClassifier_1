from plot import plot_corr_matrix


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import RFECV
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier

from xgboost import XGBClassifier

from tools import drop_unused_df_feature
import config

def get_corr_matrix(df, tic, OUT_DIR):

    #features = df.columns
    #corr_matrix = df[features].corr()
    #corr_matrix = df.corr()
    #corr_matrix["adj_close"].sort_values(ascending=False)
    #plot_corr_matrix(corr_matrix, tic, OUT_DIR)

    print("Feature selection...")

    data = df.copy()

    # data = drop_unused_df_feature(data)

    X = data.drop(config.ADD_LAGS, axis=1)

    target = data[config.ADD_LAGS]

    #rfc = RandomForestClassifier(random_state=101)
    rfc = XGBClassifier(random_state=101, verbosity = 0)
    rfecv = RFECV(estimator=rfc, step=1, cv=StratifiedKFold(10), scoring='accuracy')
    rfecv.fit(X, target)

    print('Optimal number of features: {}'.format(rfecv.n_features_))

    plt.figure(figsize=(16, 9))
    plt.title('Recursive Feature Elimination with Cross-Validation', fontsize=18, fontweight='bold', pad=20)
    plt.xlabel('Number of features selected', fontsize=14, labelpad=20)
    plt.ylabel('% Correct Classification', fontsize=14, labelpad=20)
    plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_, color='#303F9F', linewidth=3)
    #plt.show()
    plt.savefig(OUT_DIR + tic + '_feature_selected_1.png')

    print(np.where(rfecv.support_ == False)[0])

    X.drop(X.columns[np.where(rfecv.support_ == False)[0]], axis=1, inplace=True)

    print(rfecv.estimator_.feature_importances_)

    dset = pd.DataFrame()
    dset['attr'] = X.columns
    dset['importance'] = rfecv.estimator_.feature_importances_
    dset = dset.sort_values(by='importance', ascending=False)

    filename = config.FEATURE_DIRECTORY + tic + "_feature_selected.csv"
    dset.to_csv(filename)

    plt.figure(figsize=(16, 10))
    plt.barh(y=dset['attr'], width=dset['importance'], color='#1976D2')
    plt.title('RFECV - Feature importances', fontsize=20, fontweight='bold', pad=20)
    plt.xlabel('Importance', fontsize=14, labelpad=20)
    #plt.show()
    plt.savefig(OUT_DIR + tic + '_feature_selected_2.png')