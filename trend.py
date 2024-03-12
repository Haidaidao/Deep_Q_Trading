# Copyright 2019-2020 Alvaro Bartolome
# See LICENSE for details.

from investpy import get_stock_historical_data

import numpy as np
import pandas as pd

from statistics import mean

from unidecode import unidecode

import gym
from gym import spaces

#datetime is the library used to manipulate time and date
from datetime import datetime

from decimal import Decimal

import string
import global_config

MK = global_config.MK

def identify_df_trends(df, prices, window_size = 5):

    df_result = pd.DataFrame(index=df.index, columns=['trend'])

    trends = []  # Danh sách để lưu trữ xu hướng của mỗi cửa sổ
    
    # Duyệt qua mỗi cửa sổ con trong tập dữ liệu
    for i in range(len(prices) - window_size + 1):
        # Lấy cửa sổ con hiện tại
        window = prices[i:i+window_size]
        
        
        # Kiểm tra xu hướng tăng
        if all(window[j] < window[j+1] for j in range(window_size - 1)):
            trends.append(1)
        # Kiểm tra xu hướng giảm
        elif all(window[j] > window[j+1] for j in range(window_size - 1)):
            trends.append(-1)
        # Nếu không phải tăng hoặc giảm, xu hướng là Sideway
        else:
            trends.append(0)

    count = 0 
    for i in range(window_size - 1):
        trends = [0] + trends
    for index, row in df_result.iterrows():
        if count >= window_size -1: 
            df_result.at[index, 'trend'] = trends[count]
        else:
            df_result.at[index, 'trend'] = 0
        count = count + 1

    
    return df_result
    

class Trend:
    def __init__(self, iteration = None, minLimit=None, maxLimit=None, name = "Week", type = "test", columnName = "trend"):
        self.name = name
        self.spTimeserie = pd.read_csv('./datasets/'+MK+self.name+'.csv')[minLimit:maxLimit+1]
        self.minlimit = minLimit
        self.maxLimit = maxLimit
        self.Date = self.spTimeserie.loc[:, 'Date'].tolist()
        self.Time = self.spTimeserie.loc[:, 'Time'].tolist()
        self.Open = self.spTimeserie.loc[:, 'Open'].tolist()
        self.High = self.spTimeserie.loc[:, 'High'].tolist()
        self.Low = self.spTimeserie.loc[:, 'Low'].tolist()
        self.Close = self.spTimeserie.loc[:, 'Close'].tolist()

        self.columnName = columnName
        self.name = name
        self.iteration = iteration
        self.type = type


    def trend(self):
        trendResult = []
        macd , signal = self.calculate_MACD()
        for i in range(0,len(self.Date)):
            trendResult.append(self.analyze_market_trend(macd[i], signal[i]))
        return pd.DataFrame({'ensemble': trendResult}, index=pd.to_datetime(self.Date))

    def writeFile(self):
        ensambleValid=pd.DataFrame()
        ensambleValid.index.name='Date'
        self.spTimeserie.set_index('Date', inplace=True)
        trendResult = identify_df_trends(df = self.spTimeserie, prices = self.Close , window_size=5)
        for i in range(0,len(self.Date)):
            ensambleValid.at[trendResult.index[i],self.columnName]=trendResult['trend'][i]
        ensambleValid.to_csv("./Output/ensemble/"+"ensembleFolder"+"/walk"+self.name+str(self.iteration)+"ensemble_"+self.type+".csv")