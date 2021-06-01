import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
plt.rcParams["figure.figsize"] = (20,5)

def pairwise_plot(streamflow, target_site, limit = 10):
    """plot streamflow pairwisely (one by one)

    Args:
        streamflow (pd.DataFrame): the DataFrame of streamflow
        target_site (string): the target site for comparison
        limit (int, optional): the number of plots to plot. Defaults to 10.
    """    
    for idx, col in enumerate(streamflow.drop(target_site, axis=1).columns):
        if idx == limit:
            break
        streamflow[[target_site, col]].plot()
        plt.title(f"{target_site} vs. {col}, \n from {streamflow.index[0].strftime('%Y-%m-%d')} to {streamflow.index[-1].strftime('%Y-%m-%d')}")
        plt.show()

def plot_cvs(streamflow, train_idx, target_col, val_idx=None, test_idx=None):
    """plot cross validation

    Args:
        streamflow (pd.DataFrame): the DataFrame of streamflow
        train_idx (list): list of np.array of training indices
        target_site (string): the target site for comparison
        val_idx (list, optional): list of np.array of validation indices
        test_idx (list, optional): list of np.array of test indices
    """ 
    start_date, end_date = streamflow.sort_index().index[0], streamflow.sort_index().index[-1]
    train_df = streamflow.iloc[train_idx, :]
    train_df[target_col].reindex(pd.date_range(start_date, end_date), fill_value=np.NaN).plot(label='train')
    if val_idx is not None:
        val_df = streamflow.iloc[val_idx, :]
        val_df[target_col].reindex(pd.date_range(start_date, end_date), fill_value=np.NaN).plot(label='validation')
    if test_idx is not None:
        test_df = streamflow.iloc[test_idx, :]
        test_df[target_col].reindex(pd.date_range(start_date, end_date), fill_value=np.NaN).plot(label='validation')
    plt.title(f'cross validation of streamflow of station {target_col}')
    plt.legend()
    plt.show()