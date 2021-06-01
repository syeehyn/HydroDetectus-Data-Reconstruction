import numpy as np

def cv_splits(streamflow, test_size=.2):
    """create cross validation splits for streamflow data

    Args:
        streamflow (pd.DataFrame): dataframe of streamflow
        test_size (float, optional): proprotion of test size in fraction. Defaults to .2.

    Returns:
        list: list of cross validation splits with i.e. [(train_indices, test_indices)]
    """    
    num_folds = int(1/test_size)
    offset_idx = [int(i * len(streamflow) * test_size) for i in range(num_folds)]
    offset_idx.append(len(streamflow))
    df = streamflow.reset_index()
    cv_indices = []
    for s, e in zip(offset_idx[1:-2], offset_idx[2:-1]):
        val_df = df.iloc[s: e]
        train_df = df.drop(val_df.index)
        cv_indices.append((np.array(train_df.index.tolist()), np.array(val_df.index.tolist())))
    return cv_indices