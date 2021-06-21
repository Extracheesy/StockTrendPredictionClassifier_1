from plot import plot_corr_matrix


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import RFECV
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from xgboost import XGBClassifier

from tools import drop_unused_df_feature
import config

def get_corr_matrix(df, df_feature, tic, OUT_DIR, model_type, scoring):

    #features = df.columns
    #corr_matrix = df[features].corr()
    #corr_matrix = df.corr()
    #corr_matrix["adj_close"].sort_values(ascending=False)
    #plot_corr_matrix(corr_matrix, tic, OUT_DIR)

    print("Feature selection...")
    print("model: ",model_type)
    print("scoring: ", scoring)


    data = df.copy()

    # data = drop_unused_df_feature(data)

    X = data.drop('target', axis=1)

    target = data['target']

    print("target prediction: ",config.PREDICT_TARGET)
    print("data column: ", X.columns)

    if (model_type == 'XGB'):
        """rfc = XGBClassifier(random_state=101,
                            n_estimators=800,
                            max_depth=9,
                            learning_rate=0.1,
                            min_child_weight=10,
                            subsample=1,
                            colsample_bytree=0.7,
                            colsample_bylevel=1,
                            gamma=0.3,
                            verbosity=0)
        """
        rfc = XGBClassifier(random_state=101,
                            verbosity=0)
    if (model_type == 'Forest'):
        rfc = RandomForestClassifier(random_state=101)
    if (model_type == 'SVC'):
        rfc = SVC(kernel="linear")

    #rfecv = RFECV(estimator=rfc, step=1, cv=StratifiedKFold(10), scoring=scoring)
    rfecv = RFECV(estimator=rfc, scoring=scoring, min_features_to_select = config.MIN_FEATURE)
    rfecv.fit(X, target)

    #[model_type]
    #rfecv.n_features_
    #rfecv.ranking_

    print('Optimal number of features: {}'.format(rfecv.n_features_))

    plt.figure(figsize=(16, 9))
    plt.title('Recursive Feature Elimination with Cross-Validation', fontsize=18, fontweight='bold', pad=20)
    plt.xlabel('Number of features selected', fontsize=14, labelpad=20)
    plt.ylabel('% Correct Classification', fontsize=14, labelpad=20)
    plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_, color='#303F9F', linewidth=3)

    plt.savefig(OUT_DIR + tic + '_feature_selected_' + str(model_type) + '_' + str(scoring) + '_1.png')

    plt.clf()

    print(np.where(rfecv.support_ == False)[0])

    X.drop(X.columns[np.where(rfecv.support_ == False)[0]], axis=1, inplace=True)

    if (model_type == 'SVC'):
        print("feature_importances_: none")
        print("rankin: ", rfecv.ranking_)
        print("grid_scores_: ", rfecv.grid_scores_)
        print("columns: ", X.columns)

        return df_feature
    else:
        print("feature_importances_: ",rfecv.estimator_.feature_importances_)
    print("rankin: ",rfecv.ranking_)
    print("grid_scores_: ",rfecv.grid_scores_)
    print("columns: ",X.columns)

    dset = pd.DataFrame()
    dset['attr'] = X.columns
    dset['importance'] = rfecv.estimator_.feature_importances_
    dset = dset.sort_values(by='importance', ascending=False)

    filename = config.FEATURE_DIRECTORY + tic + '_feature_selected_' + str(model_type) + '_' + str(scoring) + '.csv'


    dset.to_csv(filename)

    plt.figure(figsize=(16, 10))
    plt.barh(y=dset['attr'], width=dset['importance'], color='#1976D2')
    plt.title('RFECV - Feature importances', fontsize=20, fontweight='bold', pad=20)
    plt.xlabel('Importance', fontsize=14, labelpad=20)

    plt.savefig(OUT_DIR + tic + '_feature_selected_' + str(model_type) + '_' + str(scoring) + '_2.png')

    plt.clf()

    return df_feature