# -*- coding: utf-8 -*-
# graph reduction to main components 
# from
# import packages
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from numpy.polynomial import polynomial as P
from scipy import signal
from distutils.dir_util import copy_tree
import pandas as pd
import math
import shutil
import glob
import os, sys
import matplotlib.pyplot as plt
# plt.style.use('dark_background')
# plt.style.use('default')
from importlib import reload
import insight
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import math
import copy
import torch as th
sys.path.append("..") 
sys.path.append("../..") 
# %% load data
X_raw = pd.read_csv('../data_prep/Xr.csv').iloc[::,2:]
y_raw = pd.read_csv('../data_prep/yr.csv').iloc[::,1:]
# %% see them 
X = torch.tensor(X_raw.values)
y = torch.tensor(y_raw.values)
print(X.shape)
print(y.shape)
# %%
train_size = int(len(X) * 0.67)
test_size = len(X) - train_size
from sklearn.model_selection import train_test_split
def ds_splitting(X,y):
    Xtrain, Xtest, ytrain, ytest = train_test_split(X,y, test_size=.3)
    
    Xtrain = th.tensor(Xtrain , dtype=th.float)
    ytrain = th.tensor(ytrain, dtype=th.float)
    
    Xtest = th.tensor(Xtest , dtype=th.float)
    ytest = th.tensor(ytest, dtype=th.float)
    
    train_ds = TensorDataset(Xtrain,ytrain) 
    test_ds = TensorDataset(Xtest,ytest) 
    
    train_dl = DataLoader(train_ds, batch_size = 3, shuffle=False)
    test_dl = DataLoader(test_ds, batch_size = 3, shuffle= False)
    return train_dl, test_dl
train_loader, test_loader = ds_splitting(X,y)
# plt.imshow(X_raw.values)
# y_raw.plot()
# %%
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# %%
from models_store import TRF1
epochs = 2
model = TRF1()
model.eval()
batchsize = 3
num_hist = 100
optimizer = torch.optim.RAdam(model.parameters(), lr=1e-8)
for _ in range(epochs):
    # train model
    model.train()
    for xb,yb in train_loader: 
        optimizer.zero_grad()
        pred = model(xb.to(device))
    # for i in range(batchsize):
        # x
        # yhat = model()
        break
        # n = np.random.randint(num_points-num_hist)
