from plot import plot_corr_matrix

def get_corr_matrix(df, tic, OUT_DIR):

    #features = df.columns
    #corr_matrix = df[features].corr()
    corr_matrix = df.corr()
    corr_matrix["adj_close"].sort_values(ascending=False)
    plot_corr_matrix(corr_matrix, tic, OUT_DIR)