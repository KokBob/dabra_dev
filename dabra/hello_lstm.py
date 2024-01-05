# -*- coding: utf-8 -*-
"""
https://www.youtube.com/watch?v=RCUrpCpGZ5o&t=1156s
https://www.youtube.com/watch?v=-Mx89Jcn2E4&list=PLh3I780jNsiTXlWYiNWjq2rBgg3UsL1Ub&index=5
stav: nejaky erro v dashi, neprehava tak jak je treba
https://www.youtube.com/playlist?list=PLh3I780jNsiTXlWYiNWjq2rBgg3UsL1Ub
https://community.plotly.com/t/slider-with-play-button-for-animations-independent-of-plotly/53188
https://dash.plotly.com/dash-core-components/slider
https://community.plotly.com/t/slider-with-play-button-for-animations-independent-of-plotly/53188/2
https://www.youtube.com/watch?v=d9SmpNfMg7U
https://towardsdatascience.com/how-to-create-animated-scatter-maps-with-plotly-and-dash-f10bb82d357a play button
https://stackoverflow.com/questions/71906091/python-plotly-dash-automatically-iterate-through-slider-play-button
"""
#******* CLEANED AND CLOSED *******
import sys
import pandas as pd
import matplotlib.pyplot as plt
sys.path.append("..") 
import insight
import prep
import numpy as np
import torch.optim as optim
import torch.utils.data as data
import torch
import torch.nn as nn
from pathlib import Path
from datetime import datetime
import models_store
# %%
# ins = insight.emts()
ins = insight.emts(path2emts = '../../data/0409/')
# ins = insight.emts(path2emts = '../../data/1201/1201_Loose/3. min/')
# ins = insight.emts(path2emts = '../../data/1201/1201_Loose/6.min/')
# ins = insight.emts(path2emts = '../../data/1201/1201_tightbra/3.min/')
# ins = insight.emts(path2emts = '../../data/1201/1201_tightbra/6.min/')
a_wi = ins.list_emts
# tady je potreba uz pocitat s nejakou hiarchickou rekurenci
if len(a_wi) > 17:
    print(f'hiearchicka mereni: {len(a_wi)/3}')
ins.get_emt_list()
a_wi = ins.list_emts
# print(f'{a_wi}')
# ins.print_emts_stems()
emt_dct = ins.get_emt_dict()
# train-test split for time series
timeseries = emt_dct['1DU']['Point1d'].values.astype('float32')
train_size = int(len(timeseries) * 0.067)
test_size = len(timeseries) - train_size
train, test = timeseries[:train_size], timeseries[train_size:]

lookback = 1
X_train, y_train = prep.create_dataset(train, lookback=lookback)
X_test, y_test = prep.create_dataset(test, lookback=lookback)
print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)
# %%
hidden_size_nodes = 50
model = models_store.AirModel(lookback, hidden_size_nodes)
optimizer = optim.Adam(model.parameters())
loss_fn = nn.MSELoss()
loader = data.DataLoader(data.TensorDataset(X_train, y_train), shuffle=True, batch_size=8)

# n_epochs = 2000
n_epochs = 500
for epoch in range(n_epochs):
    model.train()
    for X_batch, y_batch in loader:
        y_pred = model(X_batch)
        loss = loss_fn(y_pred, y_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    # Validation
    # if epoch % 100 != 0:
    if epoch % 10 != 0:        
        continue
    model.eval()
    with torch.no_grad():
        y_pred = model(X_train)
        train_rmse = np.sqrt(loss_fn(y_pred, y_train))
        y_pred = model(X_test)
        test_rmse = np.sqrt(loss_fn(y_pred, y_test))
    print("Epoch %d: train RMSE %.4f, test RMSE %.4f" % (epoch, train_rmse, test_rmse))
# %%
from datetime import datetime
with torch.no_grad():
    train_plot = np.ones_like(timeseries) * np.nan
    y_pred = model(X_train)
    train_plot[lookback:train_size] = y_pred.cpu().numpy().squeeze()
    test_plot = np.ones_like(timeseries) * np.nan
    test_plot[train_size+lookback:len(timeseries)] = model(X_test).cpu().numpy().squeeze()


current_time = datetime.now().strftime("%d%m%Y_%H%M%S")
Figure_name_ = f'Figure_g_{current_time}'
plt.figure()
stri = f' lookBack:{lookback} \n LSTM: AM3 \n epochs: {n_epochs} \n hidden size: {hidden_size_nodes} \n {str(model.state_dict)}'
plt.text(100, -10050, stri, fontsize=12)
plt.plot(timeseries, c='b', linewidth = 0.1)
plt.plot(train_plot, c='r')
plt.plot(test_plot, c='g')
plt.show()
plt.savefig(f'C:/Users/cim2bj/CAEs/caes/dabra/pics/{Figure_name_}.png')
