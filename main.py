from autoencoder import AE_log, AE_sqrt
import os
import json
import torch
datapath = './data/interim'
fp = os.path.join(datapath, 'dummy_filled.csv')
encoding_dims = [5]
h_dims = [[32], [32, 16], [16, 8], [8, 4], [4, 2]]
op = './data/final'
param_list = []
num_epoch = 20
for e_dim in encoding_dims:
    for h_dim in h_dims:
        param_list.append({
                        'encoding_dim': e_dim,
                        'h_dims': h_dim
                        })
if __name__=='__main__':
    results = []
    counter = 1
    total = len(param_list)
    for params in param_list:
        print('{}/{}'.format(counter, total))
        model = AE_sqrt(fp, params, num_epoch)
        model.train()
        loss = min(model.losses)
        prediction = model.predict()
        prediction.to_csv(os.path.join(op,'results', '{}_{}_sqrt_prediction.csv'.format(params['encoding_dim'], params['h_dims'])))
        params['sqrt_loss'] = loss
        torch.save(model.model, os.path.join(op,'models', '{}_{}_sqrt_prediction.pt'.format(params['encoding_dim'], params['h_dims'])))
        results.append(params)
        counter += 1
    with open(os.path.join(op, 'tuning_results.json'), 'w') as f:
        json.dump(results, f)

