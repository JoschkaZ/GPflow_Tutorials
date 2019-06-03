import os, sys
os.chdir( r'C:\\Users\\Joschka\\github\\TM7')
import numpy as np
import pandas as pd
import csv
import util
import random

def load_hist_prices():
    my_path = r'C:\\Users\\Joschka\\github\\TM7'
    data_path = my_path+r'\\data'
    file_names = [f for f in os.listdir(data_path) if os.path.isfile(os.path.join(data_path, f))]
    file_names.remove('Historie.csv')
    hist_prices = {}
    for file_name in file_names:
        print('reading ', file_name, '...')

        with open(data_path + '\\' + file_name) as f:
            reader = csv.reader(f)
            data = list(reader)

        sub_dic = {}
        keys = data[0][2:10]
        for row in data[1::]:
            d_date = row[0]
            d_time = row[1]
            unix = util.get_unix(d_date, d_time)
            values = row[2:10]
            values = [float(i) for i in values]
            sub_dic[unix] = values

        instrument_name = file_name.split('_')[0]
        hist_prices[instrument_name] = sub_dic

    print('available keys: ', list(hist_prices.keys()))
    print('order of entries: ', keys)
    return hist_prices

hist_prices = load_hist_prices()

#%%

hist_returns = {}
for instrument in hist_prices:
    times = sorted(list(hist_prices[instrument].keys()))
    rs = []
    for time in times:
        if time-60 in hist_prices[instrument]:
            r = (hist_prices[instrument][time][0] - hist_prices[instrument][(time-60)][0]) / hist_prices[instrument][(time-60)][0]
            rs.append(r)
    hist_returns[instrument] = rs

#%%

period = 1000

l_from = 50
l_to = 50
l_step = 0.1
max_loss = 1e-10
data = []

for i_master in range(1000):

    # select instrument
    idx = np.random.randint(0, len(list(hist_prices.keys())))
    instrument = list(hist_prices.keys())[idx]

    # select data:
    ifrom = random.randint(0,len(hist_returns[instrument])-period-1)
    d = np.array(hist_returns[instrument][ifrom:ifrom+period])

    # get sharpe
    mean = np.mean(d)
    std = np.std(d)
    sharpe = mean/std

    for l in np.linspace(l_from, l_to, num=1000):
        score = np.sum(np.log(np.maximum(np.ones(period) + d*l,max_loss)))
        data.append([score, sharpe, l])

print(len(data))

#%%
