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
# %% vypnout pro zacatek 
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# %%
pos_enc = torch.unsqueeze(torch.tensor(np.linspace(0.01, 1, num=100), dtype=torch.float32, device=device), 1)
# %%
src_mask = (torch.triu(torch.ones((10, 10), device=device)) == 1)
src_mask = src_mask.transpose(0, 1).float()
src_mask = src_mask.masked_fill(src_mask == 0, float('-inf'))
src_mask = src_mask.masked_fill(src_mask == 1, float(0.0))
src_mask_np = src_mask.detach().cpu().numpy()
# %% transformer layer 
d_model=512
# encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=1, ).to(device)
encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=1, ).to(device)
src = torch.rand(10, 32, 512).to(device)
out = encoder_layer(src)
print(out.shape)
out_np = out.detach().cpu().numpy()
# %%

d_model=3
dim_feedforward=2048
encoder_layer1 = nn.TransformerEncoderLayer(d_model=d_model, nhead=1, dim_feedforward=dim_feedforward, dropout=0.0, batch_first=True, norm_first=True).to(device)
linear_out = nn.Linear(3*d_model, 1).to(device)

# %%
from models_store import TRF1
model = TRF1()
optimizer = torch.optim.RAdam(model.parameters(), lr=1e-8)
for xb,yb in train_loader: 
    optimizer.zero_grad()
    # pred = model(xb.to(device))
# for i in range(batchsize):
    # x
    # yhat = model()
    break
# %%
s = xb.to(device).reshape(267, 3)
# %%
pos_enc = pos_enc[:s.size()[1]].repeat(s.size()[0], 1, 1)
# %%
pos_enc_np = pos_enc.detach().cpu().numpy()
# %%
v1 = encoder_layer1(s, src_mask[:s.size()[1],:s.size()[1]])
# v2 = encoder_layer2(s, src_mask[:s.size()[1],:s.size()[1]])
# v3 = encoder_layer3(s, src_mask[:s.size()[1],:s.size()[1]])
'''
RuntimeError: Given normalized_shape=[3], 
expected input with shape [*, 3], but got input of size[3, 267]whe
RuntimeError: The shape of the 2D attn_mask is torch.Size([3, 3]), 
but should be (267, 267).
'''
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
# %%
import torch
import torch.nn as nn

class SimpleTransformer(nn.Module):
    def __init__(self, d_model, nhead, num_encoder_layers, dim_feedforward, dropout):
        super(SimpleTransformer, self).__init__()
        self.encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_encoder_layers)
        
    def forward(self, src, src_mask):
        return self.transformer_encoder(src, src_mask)

# Parameters
d_model = 128  # Dimension of the model
nhead = 8  # Number of attention heads
num_encoder_layers = 2  # Number of encoder layers
dim_feedforward = 512  # Dimension of the feedforward network
dropout = 0.1  # Dropout rate

# Create the model
model = SimpleTransformer(d_model, nhead, num_encoder_layers, dim_feedforward, dropout)

# Create a sample input tensor with shape [seq_length, batch_size, d_model]
seq_length = 267
batch_size = 3
input_tensor = torch.randn(seq_length, batch_size, d_model)

# Create an attention mask with shape [seq_length, seq_length]
attn_mask = torch.zeros(seq_length, seq_length)

# Example: Making the mask to ignore future positions (for causal attention)
# attn_mask[i, j] = -inf if j > i else 0
attn_mask = torch.triu(torch.ones(seq_length, seq_length) * float('-inf'), diagonal=1)

print("Input Tensor Shape:", input_tensor.shape)  # Should be [267, 3, 128]
print("Attention Mask Shape:", attn_mask.shape)  # Should be [267, 267]

# Pass the input tensor and attention mask through the model
output = model(input_tensor, attn_mask)

print("Output Shape:", output.shape)  # Should be [267, 3, 128]
# %%
# Parameters
d_model = 128  # Dimension of the model
nhead = 8  # Number of attention heads
num_encoder_layers = 2  # Number of encoder layers
dim_feedforward = 512  # Dimension of the feedforward network
dropout = 0.1  # Dropout rate

# Create the model
model = SimpleTransformer(d_model, nhead, num_encoder_layers, dim_feedforward, dropout)

# Hyperparameters
learning_rate = 0.001
num_epochs = 5

# Loss and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Dummy dataset
seq_length = 267
batch_size = 3
num_batches = 10  # Number of batches in the dummy dataset
# Training loop
for epoch in range(num_epochs):
    for _ in range(num_batches):
        # Generate random input and target sequences
        input_tensor = torch.randn(seq_length, batch_size, d_model)
        target_tensor = torch.roll(input_tensor, shifts=1, dims=0)  # Shifted input as target

        # Create an attention mask with shape [seq_length, seq_length]
        attn_mask = torch.triu(torch.ones(seq_length, seq_length) * float('-inf'), diagonal=1)

        # Forward pass
        output = model(input_tensor, attn_mask)

        # Compute loss
        loss = criterion(output, target_tensor)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

print("Training completed")
