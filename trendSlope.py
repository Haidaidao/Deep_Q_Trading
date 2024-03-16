# Copyright 2019-2020 Alvaro Bartolome
# See LICENSE for details.

import math
from investpy import get_stock_historical_data

import numpy as np
import pandas as pd

from statistics import mean
import numpy
from unidecode import unidecode

import gym
from gym import spaces

#datetime is the library used to manipulate time and date
from datetime import datetime

from decimal import Decimal
from sklearn.linear_model import LinearRegression
import string
import global_config

MK = global_config.MK

def identify_df_trends(df, prices, window_size = 5):

    df_result = pd.DataFrame(index=df.index, columns=['trend'])

    trends = []  
    
    for i in range(len(prices) - window_size + 1):
        window = prices[i:i+window_size]
        
        if all(window[j] < window[j+1] for j in range(window_size - 1)):
            trends.append(1)
        
        elif all(window[j] > window[j+1] for j in range(window_size - 1)):
            trends.append(-1)
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
    

class TrendSlope:
    def __init__(self, name = "Week", columnName = "trend"):
        self.name = name
        self.spTimeserie = pd.read_csv('./datasets/'+MK+self.name+'.csv')

        self.Date = self.spTimeserie.loc[:, 'Date'].tolist()
        self.Time = self.spTimeserie.loc[:, 'Time'].tolist()
        self.Open = self.spTimeserie.loc[:, 'Open'].tolist()
        self.High = self.spTimeserie.loc[:, 'High'].tolist()
        self.Low = self.spTimeserie.loc[:, 'Low'].tolist()
        self.Close = self.spTimeserie.loc[:, 'Close'].tolist()

        self.columnName = columnName
        self.name = name


    def findDelta(self, begin, end):
        price1 = self.Close[begin]
        price2 = self.Close[end]

        return (price2-price1)/5

    def trendAddDelta(self, trendArr):

        for i in range(0,len(self.Date)):  
            if trendArr[i]!=0:
                if i-4>=0:
                    delta = self.findDelta(i-4,i)
                    trendArr[i] = math.tanh(delta)

        return trendArr


    def writeFile(self):
        
        ensambleValid=pd.DataFrame()
        ensambleValid.index.name='Date'
        self.spTimeserie.set_index('Date', inplace=True)
        trendResult = identify_df_trends(df = self.spTimeserie, prices = self.Close , window_size=5)
        trendResult['trend'] = self.trendAddDelta(trendResult['trend'].tolist())
        for i in range(0,len(self.Date)):
            ensambleValid.at[trendResult.index[i],self.columnName]=trendResult['trend'][i]
        ensambleValid['close'] = self.Close
        ensambleValid.to_csv("./Output/ensemble/"+"ensembleFolder"+self.name+".csv")