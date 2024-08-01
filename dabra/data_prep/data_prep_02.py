# -*- coding: utf-8 -*-
# %% import packages
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
sys.path.append("..") 
sys.path.append("../..") 
# %% put full directory 
# report on what has been put 
# press button consolidate and save essential 3D tracks
# process waves 
# show volumes 
# a = athlete
# condition_ = ''
# ins['a'] = insight.emts(path2emts = '../../data/0409/')
# b = athlete
condition_ = '1201_Loose'
# condition_ = '1201_normal'
# condition_ = '1201_tightbra'
# via  direct 
ins = {}
ins['b'] = insight.emts(path2emts = f'../../data/1201/{condition_}/3.min/')
# ins['c'] = insight.emts(path2emts = f'../../data/1201/{condition_}/6.min/')
# ins['d'] = insight.emts(path2emts = f'../../data/1201/{condition_}/9.min/')
# ins['e'] = insight.emts(path2emts = f'../../data/1201/{condition_}/12.min/')
# ins['f'] = insight.emts(path2emts = f'../../data/1201/{condition_}/15.min/')
# ins['g'] = insight.emts(path2emts = f'../../data/1201/{condition_}/18.min/')
# ins['h'] = insight.emts(path2emts = f'../../data/1201/{condition_}/20.min/')
# ins['j'] = insight.emts(path2emts = f'../../data/1201/{condition_}/recovery/')
# ins['k'] = insight.emts(path2emts = f'../../data/1201/{condition_}/rest/')
# .... 

# %%
i = 0
d1 = []
X_l, y_l = [], []
plt.figure()
for _ in ins:
    d_ = ins[_].get_emt_dict()
    d1.append(d_)
    X_df = d_['3DU'].iloc[::,1::]
    
    y_scl_df = d_['Scl'].iloc[::,2:5]
    y_vol_df = d_['Vol'].iloc[::,2:7]
    # y_vol_df = d_['Vol'].iloc[::,2:7]
    
    # y_df = pd.concat([y_scl_df,y_vol_df], axis = 1)
    y_df = y_scl_df
    # break 
    X = X_df.values.astype('float32')
    y = y_df.values.astype('float32')
    X_l.append(X)
    y_l.append(y)
    # break
    
    # plt.subplot(len(ins), 1, i+1)
    plt.figure()
    plt.subplot(2, 1, 1) # 
    plt.imshow(X.T)
    plt.subplot(2, 1, 2) # 
    plt.plot(pd.DataFrame(y_df))
    
    # plt.imshow(y.T)
    # plt.xaxis()
    
    # plt.subplot(len(ins), 1, i+1)
    # i+=1
    # plt.tight_layout()
# plt.tight_layout()
#%% 
plt.plot(y_df)
y_scl_df.plot()
y_vol_df.plot()
# %%
X_all = np.concatenate(X_l, axis=0)
# %%
X_a1 = pd.DataFrame(X_all)
# D = pd.concat([X_a1, y_a1])
# D = D.dropna()
# plt.imshow(D.values)
# X_a1.to_csv('Xr.csv')
# %%
X_a2 = X_a1.dropna()
X_a3 = X_a1.ffill(axis = 0)
X_a3.to_csv('Xr.csv')
# %%
y_all = np.concatenate(y_l, axis=0)
# y_a1 = y_all[:,0:3]
y_a1 = y_all
pd.DataFrame(y_a1).to_csv('yr.csv')
# %%
plt.figure()
plt.plot(pd.DataFrame(y_a1))
# %%

num_plts = 3

plt.subplot(num_plts, 1, 1) # 
plt.imshow(X_all.T)
plt.subplot(num_plts, 1, 2) # 
plt.imshow(X_a3.T)
plt.subplot(num_plts, 1, 3) # 
plt.plot(pd.DataFrame(y_a1))

