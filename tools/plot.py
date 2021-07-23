import matplotlib.pyplot as plt
import seaborn as sn

from stldecompose import decompose

def plot_seasonal(df, tic, OUT_DIR):
    df_close = df[['date', 'close']].copy()
    df_close = df_close.set_index('date')

    decomp = decompose(df_close, period=365)
    fig = decomp.plot()
    fig.set_size_inches(20, 8)

    filename = OUT_DIR + tic + "_decompose.png"
    plt.savefig(filename, dpi=500)
    plt.close('all')

def plot_close(df, tic, OUT_DIR):
    fig = plt.figure()
    fig.set_size_inches(20, 8)

    plt.title('close')
    plt.plot(df.date, df.close, linewidth=0.5)

    filename = OUT_DIR + tic + "_close.png"
    plt.savefig(filename, dpi=500)
    plt.close('all')

def plot_df(df, X, Y, filename):
    fig = plt.figure()
    fig.set_size_inches(20, 8)

    plt.title('close')
    plt.plot(df[X], df[Y], linewidth=0.5)
    plt.grid(True)

    plt.savefig(filename, dpi=500)
    plt.close('all')

def plot_avg_price_month(df, tic, OUT_DIR):
    # Compute the average price for each month
    avg_price_mth = df.groupby("month").agg({'close': 'mean'}).reset_index()

    plot_df(avg_price_mth, 'month', 'close', OUT_DIR + tic + "_avg_month.png")

def plot_avg_price_day(df, tic, OUT_DIR):
    # Compute the average price for each month
    avg_price_day = df.groupby("day").agg({'close': 'mean'}).reset_index()

    plot_df(avg_price_day, 'day', 'close', OUT_DIR + tic + "_avg_day.png")

def plot_avg_price_dayofweek(df, tic, OUT_DIR):
    # Compute the average price for each month
    avg_price_dayofweek = df.groupby("dayofweek").agg({'close': 'mean'}).reset_index()

    plot_df(avg_price_dayofweek, 'dayofweek', 'close', OUT_DIR + tic + "_avg_dayofweek.png")

def plot_corr_matrix(corr_matrix, tic, OUT_DIR):
    fig = plt.figure()
    fig.set_size_inches(20, 20)
    sn.heatmap(corr_matrix, annot=True)
    filename = OUT_DIR + tic + "_corrmatrix.png"
    plt.savefig(filename, dpi=500)
    plt.close('all')

def plot_price(df, tic, OUT_DIR):
    plot_close(df, tic, OUT_DIR)
    plot_avg_price_month(df, tic, OUT_DIR)
    plot_avg_price_day(df, tic, OUT_DIR)
    plot_avg_price_dayofweek(df, tic, OUT_DIR)
    plt.close('all')

def plot_preds_before_tuning(train, val, test, train_val, H,preds_dict, tic, OUT_DIR):
    fig = plt.figure()
    fig.set_size_inches(20, 8)

    plt.plot(train['date'], train['adj_close'], linewidth=0.5)
    plt.plot(val['date'], val['adj_close'], linewidth=0.5)
    plt.plot(test['date'], test['adj_close'], linewidth=0.5)

    # Plot the predictions
    n = 0
    for key in preds_dict:
        plt.plot(train_val[key:key + H]['date'], preds_dict[key], linewidth=0.5)

    plt.grid(True)
    filename = OUT_DIR + tic + "_prds_b_tuning.png"
    plt.savefig(filename, dpi=500)
    plt.close('all')

    fig = plt.figure()
    fig.set_size_inches(20, 8)

    plt.plot(val['date'], val['adj_close'], linewidth=0.5)
    plt.plot(test['date'], test['adj_close'], linewidth=0.5)

    # Plot the predictions
    n = 0
    for key in preds_dict:
        plt.plot(train_val[key:key + H]['date'], preds_dict[key], linewidth=0.5)

    plt.grid(True)
    filename = OUT_DIR + tic + "_prds_b_tuning_zoom.png"
    plt.savefig(filename, dpi=500)
    plt.close('all')


def plot_preds_before_tuning_one_pred(train, val, test, est, H, tic, OUT_DIR):
    fig = plt.figure()
    fig.set_size_inches(20, 8)

    plt.plot(train['date'], train['adj_close'], linewidth=0.5)
    plt.plot(val['date'], val['adj_close'], linewidth=0.5)
    plt.plot(test['date'], test['adj_close'], linewidth=0.5)
    plt.plot(test[:H]['date'], est, linewidth=0.5)

    plt.grid(True)
    filename = OUT_DIR + tic + "_prds_b_tuning_onepred.png"
    plt.savefig(filename, dpi=500)
    plt.close('all')

    fig = plt.figure()
    fig.set_size_inches(20, 8)

    plt.plot(test['date'], test['adj_close'], linewidth=0.5)
    plt.plot(test[:H]['date'], est, linewidth=0.5)

    plt.grid(True)
    filename = OUT_DIR + tic + "_prds_b_tuning_onepred_zoom.png"
    plt.savefig(filename, dpi=500)
    plt.close('all')



def plot_importance_features_no_tuning(imp, tic, OUT_DIR):
    fig = plt.figure()
    fig.set_size_inches(20, 8)

    x_val = [x[0] for x in imp]
    y_val = [x[1] for x in imp]
    plt.barh(x_val, y_val)

    plt.grid(True)
    filename = OUT_DIR + tic + "_feature_importance_no_tuning.png"
    plt.savefig(filename, dpi=500)
    plt.close('all')

def plot_error_rate(df, OUT_DIR, tic):
    fig = plt.figure()
    fig.set_size_inches(20, 8)

    plt.plot(df[df.columns[0]], df['precision'], linewidth=0.5)

    plt.grid(True)
    filename = OUT_DIR + tic + "_rmse_1.png"
    plt.savefig(filename, dpi=500)
    plt.close('all')

    plt.plot(df[df.columns[1]], df['rmse'], linewidth=0.5)

    plt.grid(True)
    filename = OUT_DIR + tic + "_rmse_2.png"
    plt.savefig(filename, dpi=500)
    plt.close('all')

    plt.plot(df[df.columns[0]], df['mape'], linewidth=0.5)

    plt.grid(True)
    filename = OUT_DIR + tic + "_mape_1.png"
    plt.savefig(filename, dpi=500)
    plt.close('all')

    plt.plot(df[df.columns[1]], df['mape'], linewidth=0.5)

    plt.grid(True)
    filename = OUT_DIR + tic + "_mape_2.png"
    plt.savefig(filename, dpi=500)
    plt.close('all')

    plt.plot(df[df.columns[0]], df['mae'], linewidth=0.5)

    plt.grid(True)
    filename = OUT_DIR + tic + "_mae_1.png"
    plt.savefig(filename, dpi=500)
    plt.close('all')

    plt.plot(df[df.columns[1]], df['mae'], linewidth=0.5)

    plt.grid(True)
    filename = OUT_DIR + tic + "_mae_2.png"
    plt.savefig(filename, dpi=500)
    plt.close('all')

    plt.plot(df[df.columns[0]], df['accuracy'], linewidth=0.5)

    plt.grid(True)
    filename = OUT_DIR + tic + "_accuracy_1.png"
    plt.savefig(filename, dpi=500)
    plt.close('all')

    plt.plot(df[df.columns[1]], df['accuracy'], linewidth=0.5)

    plt.grid(True)
    filename = OUT_DIR + tic + "_accuracy_2.png"
    plt.savefig(filename, dpi=500)
    plt.close('all')

def plot_preds_after_tuning(train, val, test, train_val, H,preds_dict, tic, OUT_DIR):
    fig = plt.figure()
    fig.set_size_inches(20, 8)

    plt.plot(train['date'], train['adj_close'], linewidth=0.5)
    plt.plot(val['date'], val['adj_close'], linewidth=0.5)
    plt.plot(test['date'], test['adj_close'], linewidth=0.5)

    # Plot the predictions
    n = 0
    for key in preds_dict:
        plt.plot(train_val[key:key + H]['date'], preds_dict[key], linewidth=0.5)

    plt.grid(True)
    filename = OUT_DIR + tic + "_prds_a_tuning.png"
    plt.savefig(filename, dpi=500)
    plt.close('all')

    fig = plt.figure()
    fig.set_size_inches(20, 8)

    plt.plot(val['date'], val['adj_close'], linewidth=0.5)
    plt.plot(test['date'], test['adj_close'], linewidth=0.5)

    # Plot the predictions
    n = 0
    for key in preds_dict:
        plt.plot(train_val[key:key + H]['date'], preds_dict[key], linewidth=0.5)

    plt.grid(True)
    filename = OUT_DIR + tic + "_prds_a_tuning_zoom.png"
    plt.savefig(filename, dpi=500)
    plt.close('all')


































