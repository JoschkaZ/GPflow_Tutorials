import requests, json
import os, sys
from os import listdir
from os.path import isfile, join
import csv
import datetime as dt
import fxcmpy
import math
import numpy as np
import pickle as pkl
import time
import matplotlib.pyplot as plt
import calendar
import pandas as pd
import json

TOKEN = '0d2336e2d419bb60e3e3310a84dda3359d24f25c'
con = None

con = fxcmpy.fxcmpy(access_token = TOKEN, log_level = 'error')

print(con.is_connected)


#%%

instruments = con.get_instruments_for_candles()
for i in range(int(len(instruments)/4)):
    print(instruments[i*4:(i+1)*4])
print(instruments[(i+1)*4:])


#%%

con.get_candles('USD/JPY', period='D1')  # daily data


#%%

start = dt.datetime(2019, 3, 18)
end = dt.datetime(2019, 3, 21)

start=dt.datetime(2019, 3, 18)
end = dt.datetime.utcnow()
con.get_candles('AUD/CHF', period='m1',
                start=start, end=end)


#%%

def get_histdata(instrument, start, stop, con):
    print(start)
    print(stop)
    data = con.get_candles(instrument, period='m1', start = start, stop=stop)
    return data


#%%

instrument = 'AUD/CHF'

#03/01/2019 22 0 21:59:00
#print(ld, lhour, lmin, lt)
ld = '03/01/2019'
lhour = 22
lmin = 0
lt = '21:59:00'
print(ld, lhour, lmin, lt)
start = dt.datetime(int(ld.split('/')[2]), int(ld.split('/')[0]), int(ld.split('/')[1]), lhour, lmin, int(lt.split(':')[2]))
print('start: ',start)

start = '2019-03-01 12:12:50.074674'

#start=dt.datetime(2019, 3, 18)
end = dt.datetime.utcnow()

end = '2019-03-16 12:12:50.074674'
data = get_histdata(instrument, start, end, con)
print(len(data))

#%%



ft0 = dt.datetime.utcnow()
delta2 = 0.

# get all instruments
my_path = r'C:\\Users\\Joschka\\github\\TM9'
data_path = my_path+r'\data'
file_names = [f for f in listdir(data_path) if isfile(join(data_path, f))]
if 'Historie.csv' in file_names: file_names.remove('Historie.csv')
hist = {}

if os.path.isfile('last_updates.pkl') == False:
    last_updates = {}
else:
    with open('last_updates.pkl', 'rb') as f:
        last_updates =  pkl.load(f)

print(last_updates)
for file_name in file_names:

    instrument = file_name.split('_')[0]
    if instrument in ['EURUSD','AUDCHF','GBPJPY','AUDJPY','EURCHF','EURGBP','EURJPY']:
        instrument=  instrument[0:3] + '/' + instrument[3:6]

    if instrument in last_updates:
        ld = last_updates[instrument][0]
        lt = last_updates[instrument][1]
    else:
        with open(data_path + '\\' + file_name) as f:
            reader = csv.reader(f)
            data = list(reader)
            ld = data[-1][0]
            lt =  data[-1][1]


    print('Updating ', instrument)
    print('Last data @', ld, lt)

    lhour = int(lt.split(':')[0])
    lmin = int(lt.split(':')[1])
    if lmin == 59:
        lmin = 0
        lhour += 1
        if lhour == 24: lhour = 0
    else:
        lmin += 1

    print(ld, lhour, lmin, lt)
    start = dt.datetime(int(ld.split('/')[2]), int(ld.split('/')[0]), int(ld.split('/')[1]), lhour, lmin, int(lt.split(':')[2]))
    stop = dt.datetime.utcnow()

    print('updating ', math.floor((stop-start).total_seconds()/60), 'minutes...')
    if math.floor((stop-start).total_seconds()/60) > 0:
        ft1 = dt.datetime.utcnow()
        print(start, stop)
        new_data = get_histdata(instrument, start, stop,con)
        delta2 += (dt.datetime.utcnow()-ft1).microseconds
        new_data['aaa'] = new_data.index

        new_data = new_data.values.tolist()
        new_data_shifted = []
        ndate = -1
        print(len(new_data), ' datapoints found!')
        for r in new_data:
            ndate = str(r[9]).split(" ")[0]
            ndate = ndate.split('-')[1] + '/' + ndate.split('-')[2] + '/' + ndate.split('-')[0]
            ntime = str(r[9]).split(" ")[1]
            nbidopen = r[0]
            nbidclose = r[1]
            nbidhigh = r[2]
            nbidlow = r[3]
            naskopen = r[4]
            naskclose = r[5]
            naskhigh = r[6]
            nasklow = r[7]
            ntickqty = r[8]
            new_data_shifted.append([ndate,ntime,nbidopen,nbidhigh,nbidlow,nbidclose,naskopen,naskhigh,nasklow,naskclose,ntickqty])

            #data.append([ndate,ntime,nbidopen,nbidhigh,nbidlow,nbidclose,naskopen,naskhigh,nasklow,naskclose,ntickqty])
        if ndate == -1:
            last_updates[instrument] = [ld,lt]
            if len(new_data) != 0:
                print('WARNING: 23699')
                break
        else:
            last_updates[instrument] = [ndate, ntime]
        #print('SHIFTED')
        #print(data)

        #with open("test_" + instrument.replace('/','') + ".csv", "w", delimiter=',', lineterminator='\n') as f:
        #    writer = csv.writer(f)
        #    writer.writerows(data)

        #with open("test_" + instrument.replace('/','') + ".csv", "a") as fp:
        #    wr = csv.writer(fp, dialect='excel')
        #    wr.writerow(data)

        #with open(data_path + '\\' + file_name,'a',newline='') as fd:
        #    writer = csv.writer(fd)
        #    for row in new_data_shifted:
        #         writer.writerow(row)
    #else:
    #    print('nothing to update...')
#with open('last_updates.pkl', 'wb') as f:
#    pkl.dump(last_updates, f)

delta1 = (dt.datetime.utcnow()-ft0).microseconds
print('time efficiency: ', np.round(delta2 / (delta2+delta1)*100,2), "%")

return 'Instruments updated!'
