def split_xy(df, target_col):
    """perform conventional X y splits without sequence information included

    Args:
        df (pd.DataFrame): the dataset with feature and target column indicated
        target_col (string): the name of target column

    Returns:
        tuple: (X, y) in numpy array.
    """ 
    return df.drop(target_col, axis = 1).to_numpy() , df[target_col].to_numpy()