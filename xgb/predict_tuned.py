import os
import numpy as np
import pandas as pd

import config

from error import get_error_metrics

def run_validation_set_with_tuned_param(df, tic, OUT_DIR):

    for pred_day in config.PRED_DAY_LIST:
        print("Get error metrics on validation set after hyperparameter tuning")

        train = df[pred_day - config.TRAIN_VAL_SIZE:pred_day - config.VAL_SIZE].copy()
        val = df[pred_day - config.VAL_SIZE:pred_day].copy()
        train_val = df[pred_day - config.TRAIN_VAL_SIZE:pred_day].copy()
        test = df[pred_day:pred_day + config.H].copy()

        filename = OUT_DIR + tic + "_opt_param.csv"
        df_param = pd.read_csv(filename)

        n_estimators_opt_param = df_param['n_estimators']
        n_estimators_opt_param = list(set(n_estimators_opt_param))

        max_depth_opt_param = df_param['max_depth']
        max_depth_opt_param = list(set(max_depth_opt_param))

        learning_rate_opt_param = df_param['learning_rate']
        learning_rate_opt_param = list(set(learning_rate_opt_param))

        min_child_weight_opt_param = df_param['min_child_weight']
        min_child_weight_opt_param = list(set(min_child_weight_opt_param))

        subsample_opt_param = df_param['subsample']
        subsample_opt_param = list(set(subsample_opt_param))

        colsample_bytree_opt_param = df_param['colsample_bytree']
        colsample_bytree_opt_param = list(set(colsample_bytree_opt_param))

        colsample_bylevel_opt_param = df_param['colsample_bylevel']
        colsample_bylevel_opt_param = list(set(colsample_bylevel_opt_param))

        gamma_opt_param = df_param['gamma']
        gamma_opt_param = list(set(gamma_opt_param))

        best_rmse = 100
        for n_estimators_opt in n_estimators_opt_param:
            for max_depth_opt in max_depth_opt_param:
                for learning_rate_opt in learning_rate_opt_param:
                    for min_child_weight_opt in min_child_weight_opt_param:
                        for subsample_opt in subsample_opt_param:
                            for colsample_bytree_opt in colsample_bytree_opt_param:
                                for colsample_bylevel_opt in colsample_bylevel_opt_param:
                                    for gamma_opt in gamma_opt_param:
                                        rmse_aft_tuning, mape_aft_tuning, mae_aft_tuning, accuracy_aft_tuning, preds_dict = get_error_metrics(train_val,
                                                                                                                                              config.TRAIN_SIZE,
                                                                                                                                              config.N,
                                                                                                                                              config.H,
                                                                                                                                              seed=config.MODEL_SEED,
                                                                                                                                              n_estimators=n_estimators_opt,
                                                                                                                                              max_depth=max_depth_opt,
                                                                                                                                              learning_rate=learning_rate_opt,
                                                                                                                                              min_child_weight=min_child_weight_opt,
                                                                                                                                              subsample=subsample_opt,
                                                                                                                                              colsample_bytree=colsample_bytree_opt,
                                                                                                                                              colsample_bylevel=colsample_bylevel_opt,
                                                                                                                                              gamma=gamma_opt)

                                        if rmse_aft_tuning < best_rmse:
                                            best_accuracy = accuracy_aft_tuning
                                            best_rmse = rmse_aft_tuning
                                            best_mape = mape_aft_tuning
                                            best_mae = mae_aft_tuning
                                            best_est_dict = preds_dict

                                            best_n_estimators = n_estimators_opt
                                            best_max_depth = max_depth_opt
                                            best_learning_rate = learning_rate_opt
                                            best_min_child_weight = min_child_weight_opt
                                            best_subsample = subsample_opt
                                            best_colsample_bytree = colsample_bytree_opt
                                            best_colsample_bylevel = colsample_bylevel_opt
                                            best_gamma = gamma_opt

        print("Best VAL -",tic ," RMSE = %0.3f" % best_rmse)
        print("Best VAL -",tic ," MAPE = %0.3f%%" % best_mape)
        print("Best VAL -",tic ," MAE = %0.3f" % best_mae)
        print("Best VAL -",tic ," ACCURACY = %0.3f" % best_accuracy)

        print("Best VAL -",tic ," n_estimators_opt: ", best_n_estimators)
        print("Best VAL -",tic ," max_depth_opt: ", best_max_depth)
        print("Best VAL -",tic ," learning_rate_opt: ", best_learning_rate)
        print("Best VAL -",tic ," min_child_weight_opt: ", best_min_child_weight)
        print("Best VAL -",tic ," subsample_opt: ", best_subsample)
        print("Best VAL -",tic ," colsample_bytree_opt: ", best_colsample_bytree)
        print("Best VAL -",tic ," colsample_bylevel_opt: ", best_colsample_bylevel)
        print("Best VAL -",tic ," gamma_opt: ", best_gamma)

        plot_preds_after_tuning(train, val, test, train_val, config.H, best_est_dict, tic, OUT_DIR)

