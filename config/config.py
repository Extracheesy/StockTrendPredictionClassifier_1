N = 10  # for feature at day t, we use lags from t-1, t-2, ..., t-N as features
H = 21  # Forecast horizon, in days. Note there are about 252 trading days in a year
START_YEAR = 2010
TRAIN_SIZE = 252 * 3  # Use 3 years of data as train set. Note there are about 252 trading days in a year
VAL_SIZE = 252  # Use 1 year of data as validation set
TRAIN_VAL_SIZE = TRAIN_SIZE + VAL_SIZE

PRED_DAY_LIST = [1008, 1050, 1092, 1134, 1176, 1218, 1260, 1302, 1344, 1386, 1428, 1470]
PRED_DAY = 1008  # Predict for this day, for the next H-1 days. Note indexing of days start from 0.

TRACES_DIR = "./traces/"
RESULTS_DIR = TRACES_DIR + "results/"
STOCKS_DIRECTORY = './Data/Stocks/'
STOCKS_DJI_DIRECTORY = './Data/Stocks_dji/'

READ_DATA_FILE = True
#READ_DATA_FILE = False

MODE_DEBUG = False

ADD_INDICATORS = True
PLOT_PRICE = False
#PLOT_PRICE = True
CORR_MATRIX = False
#CORR_MATRIX = True
#PREDICT_BEFORE_TUNING = True
PREDICT_BEFORE_TUNING = False
#PREDICT_BEFORE_TUNING_ONE_PRED = True
PREDICT_BEFORE_TUNING_ONE_PRED = False
TUNING_PARAM = True
PRINT_SHAPE = False
#PRINT_SHAPE = True

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