# -*- coding: utf-8 -*-
"""
https://pytorch.org/tutorials/beginner/transformer_tutorial.html
https://towardsdatascience.com/how-to-run-inference-with-a-pytorch-time-series-transformer-394fd6cbe16c
"""
import torch.optim as optim
import torch.utils.data as data
import torch
import torch.nn as nn
import numpy as np

device = 'GPU'
class AirModel(nn.Module):
    def __init__(self, lookback, hidden_size_nodes):
        super().__init__()
        # self.lstm = nn.LSTM(input_size=1, hidden_size=50, num_layers=1, batch_first=True)
        self.lstm = nn.LSTM(input_size=lookback, hidden_size=hidden_size_nodes, num_layers=1, batch_first=True)
        self.lstm2 = nn.LSTM(input_size=hidden_size_nodes, hidden_size=hidden_size_nodes, num_layers=1, batch_first=True)
        # self.linear = nn.Linear(hidden_size_nodes, hidden_size_nodes)
        # self.linear2 = nn.Linear(hidden_size_nodes, hidden_size_nodes)
        self.linear3 = nn.Linear(hidden_size_nodes, 1)
    def forward(self, x):
        x, _ = self.lstm(x)
        x, _ = self.lstm2(x)
        # x = self.linear(x)
        # x = self.linear2(x)
        x = self.linear3(x)
        return x
class LSTM1(nn.Module):
    # input_size = 
    # output_size = 
    # Initialize the layers
    def __init__(self, input_size, output_size ):
        self.name_model = 'SimpleNet'
class ModelT1(nn.Module):
    def __init__(self, d_model=45, dim_feedforward=2048, device=device):
        super(ModelT1, self).__init__()

        # encode time
        self.pos_enc = torch.unsqueeze(torch.tensor(np.linspace(0.01, 1, num=100), dtype=torch.float32, device=device), 1)
        
        # create mask for hiding future inputs
        self.src_mask = (torch.triu(torch.ones((100, 100), device=device)) == 1)
        self.src_mask = self.src_mask.transpose(0, 1).float()
        self.src_mask = self.src_mask.masked_fill(self.src_mask == 0, float('-inf'))
        self.src_mask = self.src_mask.masked_fill(self.src_mask == 1, float(0.0))
        
        # extract temporal features using transformers
        #IDEA: It is import to set the flag norm_first to True, otherwise the expected gradients
        # close near the output layer are larger hindering optimization
        # For more information, see https://arxiv.org/pdf/2002.04745.pdf
        self.encoder_layer1 = nn.TransformerEncoderLayer(d_model=d_model, nhead=1, dim_feedforward=dim_feedforward, dropout=0.0, batch_first=True, norm_first=True).to(device)
        self.encoder_layer2 = nn.TransformerEncoderLayer(d_model=d_model, nhead=1, dim_feedforward=dim_feedforward, dropout=0.0, batch_first=True, norm_first=True).to(device)
        self.encoder_layer3 = nn.TransformerEncoderLayer(d_model=d_model, nhead=1, dim_feedforward=dim_feedforward, dropout=0.0, batch_first=True, norm_first=True).to(device)
        
        # compute output based on temporal features using a linear layer
        self.linear_out = nn.Linear(3*d_model, 1).to(device)

