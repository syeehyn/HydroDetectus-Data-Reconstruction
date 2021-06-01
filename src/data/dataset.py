import torch
import pandas as pd
from torch.utils.data import Dataset

class streamflowDataset(Dataset):
    def __init__(self, streamflow, 
                    target_col=None
                ):
        self.streamflow = streamflow
        self.target_col = target_col
        self.dates = streamflow.index.tolist()
        self.break_points = self._split_date_sequence()
    def __getitem__(self, idx):
        if self.target_col != None:
            X = torch.FloatTensor(self.streamflow.drop(self.target_col, axis=1).to_numpy())[idx, :]
            y = torch.FloatTensor(self.streamflow[self.target_col].to_numpy())[idx]
            
            
            return (X, y)
        else:
            X = torch.FloatTensor(self.streamflow.drop(self.target_col, axis=1).iloc[idx, :].to_numpy())
            return X
    def __len__(self):
        return len(self.streamflow)
    def _split_date_sequence(self):
        break_points = []
        for idx, (s, e) in enumerate(zip(self.dates[:-1], self.dates[1:])):
            if e - s > pd.Timedelta('1D'):
                break_points.append(idx+1)
        return break_points