# HydroDetectus-Data-Reconstruction

This project aims to build models to reconstruct missing data of California Water Stations.

## Description of Content

```
├── LICENSE
├── README.md
├── autoencoder
│   ├── ae_log.py #log preprocessor
│   ├── ae_sqrt.py #sqrt preprocessor
│   ├── models #models
│   │   ├── __init__.py
│   │   ├── linear_ae.py
│   │   └── lstm_ae.py
│   └── train.py # train util functions
├── main.py #training script
└── notebook #notebook for analysis
```

## Missing Data analysis/EDA

- [ ] a case to case study of following questions:
  - [ ] what type of missing of the station? MNAR, MCAR, MAR
    - if MNAR, find a data mapping function
    - if MCAR/MAR, what ML model should apply
  - [ ] proportion of missing data among different stations
    - [ ] what should we do if missing in days count?
    - [ ] what should we do if missing in months count?
    - [ ] what should we do if missing in years count?

## Models

### Univariate Models

the models only consider `streamflow` as features to train in LSTM/Linear model

- [x] Linear Based Unisite Autoencoder
- [x] LSTM Based Unisite Autoencoder
- [ ] LSTM Based Attention Unisite Autoencoder
- [ ] LSTM Based Attention Multisite (site embedding) Autoencoder

### Multivariate Models

the models feed in additional features like `temperature` in data filling model
- [ ] LSTM Based Covariate Autoencoder
- [ ] LSTM Based Covariate Attention Unisite Autoencoder
- [ ] LSTM Based Multisite Attention Multisite (site embedding) Autoencoder
- [ ] Tree Based/XGBoost model
- [ ] ... 


## Preprocessing Techniques

- [x] Apply log Streamflow
- [x] Apply square root of Streamflow
- [ ] What need to do for preprocessing in terms of covariate variables 

## *Missing Data Initiation

In our case, `0` is meaningful in the real dataset, and variance of the streamflow is large from time to time. Unlike traditional data imputation techniques which can initialize value with `mean` or `0`, we need to find different ways to initialize value of autoencoder.

- [x] mean of adjacent sites who have data
- [ ] *unsupervised learning techniques to cluster sites.
- [ ] ...

## Cross Validation/Model Checking

Since we do not have ground-truth of missing data, what should we do to select models?

- [ ] What metrics?
- [ ] If metrics not available, then what's the golden standard?


## Significance of filling missing data

TO BE FILLED