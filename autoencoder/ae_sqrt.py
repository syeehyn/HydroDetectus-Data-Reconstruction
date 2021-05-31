import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import torch
from tqdm import tqdm
from .models import LINEAR_AE, LSTM_AE, CONV_LSTM_AE


class AE_sqrt():
    def __init__(self, fp, params, num_epoch):
        self.df = pd.read_csv(fp, index_col='Date')
        self.params = params
        self.sc, self.train_set = self._preprocessing(self.df)
        self.train_seqs, self.seq_len, _ = self._create_dataset(self.train_set)
        self.model = LSTM_AE(
                input_dim=1,
                encoding_dim=params['encoding_dim'],
                h_dims=params['h_dims'],
                h_activ=None,
                out_activ=None
                )
        self.num_epoch = num_epoch
    def train(self):
        self.models, self.losses = self.train_model(
                                            self.model, \
                                            self.train_seqs, \
                                            verbose=True, \
                                            lr=1e-3, \
                                            epochs=self.num_epoch,\
                                            denoise=False
                                            )
        self.model = self.models[np.argmin(np.array(self.losses))]
    def predict(self):
        preds = self.get_preds(self.model, self.train_seqs, self.seq_len)
        pred = np.hstack(preds).T
        pred = self.sc.inverse_transform(pred)
        pred = np.square(pred)
        pred[pred<0] = 0
        return pd.DataFrame(pred, index=self.df.index, columns=self.df.columns)

    def _preprocessing(self, df):
        train = df.fillna(0).apply(np.sqrt).values
        sc = MinMaxScaler()
        train = sc.fit_transform(train)
        return sc, train
    def _create_dataset(self, df):
        sequences = df.astype(np.float32).tolist()
        dataset = [torch.tensor(s).unsqueeze(1).float() for s in sequences]
        n_seq, seq_len, n_features = torch.stack(dataset).shape
        return dataset, seq_len, n_features


    def train_model(self, model, train_set, verbose, lr, epochs, denoise):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = torch.nn.SmoothL1Loss(size_average=False)
        mean_losses = []
        models = []
        for epoch in range(1, epochs + 1):
            model.train()
            losses = []
            for x in tqdm(train_set):
                optimizer.zero_grad()
                # Forward pass
                x_prime = model(x.to(device))
                loss = criterion(x_prime, x.to(device))
                # Backward pass
                loss.backward()
                optimizer.step()
                losses.append(loss.item())
            mean_loss = np.mean(losses)
            mean_losses.append(mean_loss)
            models.append(model)
            if verbose:
                print(f"Epoch: {epoch}, Loss: {mean_loss}")
        return models,mean_losses
    def get_preds(self, model, train_set, seq_len):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        model.eval()
        
        pred = []
        for seq in tqdm(train_set):
            z = model.encoder(seq.to(device))
            x_prime = model.decoder(z, seq_len=seq_len)
            pred.append(x_prime.detach().cpu().numpy())
        return pred
    
    def _create_dataset(self, df):
        sequences = df.astype(np.float32).tolist()
        dataset = [torch.tensor(s).unsqueeze(1).float() for s in sequences]
        n_seq, seq_len, n_features = torch.stack(dataset).shape
        return dataset, seq_len, n_features