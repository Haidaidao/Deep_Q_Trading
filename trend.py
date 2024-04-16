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

class TrendGenerator:
    def __init__(self, name = "Week", type = "test", columnName = "trend"):
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
        self.type = type


    def EMA(self, period=12):
        # print(pd.Series(self.Close).ewm(span=period, adjust=False).mean())
        return pd.Series(self.Close).ewm(span=period, adjust=False).mean()

    # Tính toán MACD và đường tín hiệu MACD
    def calculate_MACD(self, fast_period=12, slow_period=26, signal_period=9):
        ema_fast = self.EMA(period=fast_period)
        ema_slow = self.EMA(period=slow_period)
        # print(ema_fast - ema_slow)
        a = ema_fast - ema_slow
        b = a.ewm(span=signal_period, adjust=False).mean()
        return a.tolist(), b.tolist()

    def analyze_market_trend(self, macd, macd_signal):
        # UP
        if macd > macd_signal:
            return 1
        # DOWN
        elif macd < macd_signal:
            return -1
        # SIDEWAY
        else:
            return 0


    def trend(self):
        trendResult = []
        macd , signal = self.calculate_MACD()
        for i in range(0,len(self.Date)):
            trendResult.append(self.analyze_market_trend(macd[i], signal[i]))
        return pd.DataFrame({'trend': trendResult}, index=pd.to_datetime(self.Date))

    def writeFile(self, file_name):
        ensambleValid=pd.DataFrame()
        ensambleValid.index.name='Date'
        self.spTimeserie.set_index('Date', inplace=True)
        trendResult = self.trend()
        for i in range(0,len(self.Date)):
            ensambleValid.at[trendResult.index[i],self.columnName]=trendResult['trend'][i]
        ensambleValid['close'] = self.Close
        ensambleValid.index = ensambleValid.index.strftime('%m/%d/%Y')
        ensambleValid.to_csv(file_name)