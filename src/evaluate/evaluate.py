from sklearn import metrics
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from ..data.preproc import split_xy

plt.rcParams["figure.figsize"] = (20,5)

def evaluate_normal(streamflow, train_idx, val_idx, target_site, model):
    train, val = streamflow.iloc[train_idx, :], streamflow.iloc[val_idx, :]
    X_train, y_train = split_xy(train, target_site)
    X_val, y_val = split_xy(val, target_site)
    model.fit(X_train, y_train)
    y_val_pred = model.predict(X_val)
    print(f'validation root mean squared error: {np.sqrt(metrics.mean_squared_error(y_val, y_val_pred))}')
    print(f'validation r2 score {metrics.r2_score(y_val, y_val_pred)} \n')
    print(f'visualize the prediction for validation set: ')
    
    pd.DataFrame({
        'true': val[target_site],
        'pred': model.predict(X_val)
    }).plot()
    plt.show()