# Copyright 2019-2020 Alvaro Bartolome
# See LICENSE for details.

import math
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
import trendStrengthIdentifier

MK = global_config.MK
trend_type = global_config.trend_type

trend_add_delta_linear = trendStrengthIdentifier.linear_regression_slope
trend_add_delta_two_point = trendStrengthIdentifier.two_point_slope

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

    def _EMA(self, period=12):
        # print(pd.Series(self.Close).ewm(span=period, adjust=False).mean())
        return pd.Series(self.Close).ewm(span=period, adjust=False).mean()


    def _calculate_MACD(self, fast_period=12, slow_period=26, signal_period=9):
        ema_fast = self._EMA(period=fast_period)
        ema_slow = self._EMA(period=slow_period)
        # print(ema_fast - ema_slow)
        a = ema_fast - ema_slow
        b = a.ewm(span=signal_period, adjust=False).mean()
        return a.tolist(), b.tolist()

    def _analyze_market_trend(self, macd, macd_signal):
        # UP
        if macd > macd_signal:
            return 1
        # DOWN
        elif macd < macd_signal:
            return -1
        # SIDEWAY
        else:
            return 0
        
    def MACD_signal(self):
        trendResult = []
        macd , signal = self._calculate_MACD()
        for i in range(0,len(self.Date)):
            trendResult.append(self._analyze_market_trend(macd[i], signal[i]))
        return pd.DataFrame({'trend': trendResult}, index=self.spTimeserie.index)
    
    def trendWA(self, df, prices, window_size=5):
        df_result = pd.DataFrame(index=df.index, columns=['trend'])
        # df_result['close'] = prices

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
    
    def trendWAWithLinear(self, trendArr):
        for i in range(0,len(self.Date)):  
            if trendArr[i]!=0:
                if i-4>=0:
                    delta = trend_add_delta_linear(self.Close, i - 4, i, False)
                    trendArr[i] = delta
                else:
                    if trendArr[i] != 0:
                        trendArr[i] = 0

        return trendArr
    
    def trendWAWithLinearScaler(self, trendArr):
        for i in range(0,len(self.Date)):  
            if trendArr[i]!=0:
                if i-4>=0:
                    delta = trend_add_delta_linear(self.Close, i - 4, i, False)
                    trendArr[i] = delta
                    normalized_m = np.arctan(delta) / (np.pi / 2)
                    trendArr[i] = normalized_m
                else:
                    if trendArr[i] != 0:
                        trendArr[i] = 0

        return trendArr
    
    def trendWAWithTwoPoint(self, trendArr):
        for i in range(0,len(self.Date)):  
            if trendArr[i]!=0:
                if i-4>=0:
                    delta = trend_add_delta_two_point(self.Close, i - 4, i, False)
                    trendArr[i] = delta
                else:
                    if trendArr[i] != 0:
                        trendArr[i] = 0

        return trendArr
    
    def trendWAWithTwoPointScaler(self, trendArr):
        for i in range(0,len(self.Date)):  
            if trendArr[i]!=0:
                if i-4>=0:
                    delta = trend_add_delta_two_point(self.Close, i - 4, i, False)
                    trendArr[i] = math.tanh(delta)

                else:
                    if trendArr[i] != 0:
                        trendArr[i] = 0

        return trendArr
    
    def writeFile(self, file_name):
        ensambleValid=pd.DataFrame()
        ensambleValid.index.name='Date'
        self.spTimeserie.set_index('Date', inplace=True)

        assert trend_type == "MACD" or trend_type == "TrendWA" or trend_type == "TrendWAWithTwoPointScaler" or  trend_type == "TrendWAWithLinearScaler" or trend_type == "TrendWAWithTwoPoint" or trend_type == "TrendWAWithLinear", "trend_type must be \"MACD\" or \"TrendWA\""

        if trend_type == "TrendWA":
            trendResult = self.trendWA(df = self.spTimeserie, prices = self.Close , window_size=5)

        if trend_type == "MACD":
            trendResult = self.MACD_signal()

        if trend_type == "TrendWAWithLinear":
            trendResult = self.trendWA(df = self.spTimeserie, prices = self.Close , window_size=5)
            trendResult['trend'] = self.trendWAWithLinear(trendResult['trend'].tolist())

        if trend_type == "TrendWAWithTwoPoint":
            trendResult = self.trendWA(df = self.spTimeserie, prices = self.Close , window_size=5)
            trendResult['trend'] = self.trendWAWithTwoPoint(trendResult['trend'].tolist())

        if trend_type == "TrendWAWithLinearScaler":
            trendResult = self.trendWA(df = self.spTimeserie, prices = self.Close , window_size=5)
            trendResult['trend'] = self.trendWAWithLinearScaler(trendResult['trend'].tolist())

        if trend_type == "TrendWAWithTwoPointScaler":
            trendResult = self.trendWA(df = self.spTimeserie, prices = self.Close , window_size=5)
            trendResult['trend'] = self.trendWAWithTwoPointScaler(trendResult['trend'].tolist())

        for i in range(0,len(self.Date)):
            ensambleValid.at[trendResult.index[i],self.columnName]=trendResult['trend'][i]
        ensambleValid['close'] = self.Close
        ensambleValid['open'] = self.Open
        ensambleValid.to_csv(file_name)