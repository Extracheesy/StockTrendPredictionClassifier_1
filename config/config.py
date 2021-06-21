N = 5  # for feature at day t, we use lags from t-1, t-2, ..., t-N as features
#H = 21  # Forecast horizon, in days. Note there are about 252 trading days in a year
H = 5  # Forecast horizon, in days. Note there are about 252 trading days in a year
START_YEAR = 2010
TRAIN_SIZE = 252 * 3  # Use 3 years of data as train set. Note there are about 252 trading days in a year
VAL_SIZE = 252  # Use 1 year of data as validation set
TRAIN_VAL_SIZE = TRAIN_SIZE + VAL_SIZE

PRED_DAY_LIST = [1008, 1050, 1092, 1134, 1176, 1218, 1260, 1302, 1344, 1386, 1428, 1470]
PRED_DAY_NB_ITEM = 3
PRED_DAY = 1008  # Predict for this day, for the next H-1 days. Note indexing of days start from 0.

TRACES_DIR = "./traces/"
RESULTS_DIR = TRACES_DIR + "results/"
FEATURE_DIRECTORY = TRACES_DIR + "feature/"
STOCKS_DIRECTORY = './Data/Stocks/'
STOCKS_DJI_DIRECTORY = './Data/Stocks_dji/'

#ADD_LAGS = 'adj_close'
ADD_LAGS = 'trend'
#PREDICT_TARGET = 'trend'
PREDICT_TARGET = 'target_day+1'
DROP_LAGS = True
#DROP_LAGS = False
MIN_FEATURE = 3

BALANCE_DATASET = True
#BALANCE_DATASET = False
#SMOTE = True
SMOTE = True
ADASYN = False
#ADASYN = False

CORR_MATRIX = True
#CORR_MATRIX = False

READ_DATA_FILE = True
#READ_DATA_FILE = False
MODE_DEBUG = False
#MODE_DEBUG = True
ERROR_RATE_DISPLAY = True
#ERROR_RATE_DISPLAY = False

ADD_INDICATORS = True
#ADD_INDICATORS = False
FILTERS = True
#FILTERS = False
PLOT_PRICE = False
#PLOT_PRICE = True

#PREDICT_BEFORE_TUNING = True
PREDICT_BEFORE_TUNING = False
#PREDICT_BEFORE_TUNING_ONE_PRED = True
PREDICT_BEFORE_TUNING_ONE_PRED = False
PREDICT_TUNING_PARAM = True
#PREDICT_TUNING_PARAM = False
#PREDICT_VALID_WITH_PARAM = True
PREDICT_VALID_WITH_PARAM = False
RUN_VALID_WITH_PARAM = False
PREDICT_TEST_SET_WITH_PARAM = True
#PREDICT_TEST_SET_WITH_PARAM = False
PREDICT_GRID_SEARCH = False

PRINT_SHAPE = False
#PRINT_SHAPE = True

GENERIC_PARAM_FOR_TEST = True

N_ESTIMATORS = 100  # Number of boosted trees to fit. default = 100
MAX_DEPTH = 3  # Maximum tree depth for base learners. default = 3
LEARNING_RATE = 0.1  # Boosting learning rate (xgb’s “eta”). default = 0.1
MIN_CHILD_WEIGHT = 1  # Minimum sum of instance weight(hessian) needed in a child. default = 1
SUBSAMPLE = 1  # Subsample ratio of the training instance. default = 1
COLSAMPLE_BYTREE = 1  # Subsample ratio of columns when constructing each tree. default = 1
COLSAMPLE_BYLEVEL = 1  # Subsample ratio of columns for each split, in each level. default = 1
GAMMA = 0  # Minimum loss reduction required to make a further partition on a leaf node of the tree. default=0
MODEL_SEED = 100

OPT_N_ESTIMATORS = N_ESTIMATORS
OPT_MAX_DEPTH = MAX_DEPTH
OPT_LEARNING_RATE = LEARNING_RATE
OPT_MIN_CHILD_WEIGHT = MIN_CHILD_WEIGHT
OPT_SUBSAMPLE = SUBSAMPLE
OPT_COLSAMPLE_BYTREE = COLSAMPLE_BYTREE
OPT_COLSAMPLE_BYLEVEL = COLSAMPLE_BYLEVEL
OPT_GAMMA = GAMMA

LIST_COLUMNS = ['rmse',
                'mape',
                'mae',
                'accuracy',
                'n_estimators',
                'max_depth',
                'learning_rate',
                'min_child_weight',
                'subsample',
                'colsample_bytree',
                'colsample_bylevel',
                'gamma']

LIST_COLUMNS_RESULTS = ['day',
                        'tic',
                        'type',
                        'rmse',
                        'mape',
                        'mae',
                        'accuracy',
                        'trend_day_first',
                        'trend_day_end',
                        'trend_all_percent',
                        'first_day_trend_test',
                        'first_day_trend_pred',
                        'end_day_trend_test',
                        'end_day_trend_pred',
                        'n_estimators',
                        'max_depth',
                        'learning_rate',
                        'min_child_weight',
                        'subsample',
                        'colsample_bytree',
                        'colsample_bylevel',
                        'gamma']

LIST_COLUMNS_TIC_SUMMARY = ['tic',
                            'iter_pred',

                            'rmse_avg',
                            'mape_avg',
                            'mae_avg',
                            'accuracy_avg',

                            'trend_day_first_accuracy',
                            'trend_day_end_accuracy',
                            'trend_all_accuracy',

                            'daily_strategy_win',
                            'daily_strategy_no_bet',
                            'daily_strategy_loss',

                            'daily_and_trend_strategy_win',
                            'daily_and_trend_strategy_no_bet',
                            'daily_and_trend_strategy_loss',

                            'first_day_trend_test_up',
                            'first_day_trend_pred_up',

                            'end_day_trend_test_up',
                            'end_day_trend_pred_up']

LIST_COLUMNS_FEATURE = ['year',
                        'month',
                        'week',
                        'day',
                        'dayofweek',
                        'dayofyear'
                        #'is_month_end',
                        #'is_month_start',
                        #'is_quarter_end',
                        #'is_quarter_start',
                        #'is_year_end'
                        ]