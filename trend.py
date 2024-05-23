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
    
    def check_trend(self, lst):
        if all(lst[i] <= lst[i + 1] for i in range(len(lst) - 1)):
            return 1
        elif all(lst[i] >= lst[i + 1] for i in range(len(lst) - 1)):
            return -1
        else:
            return 0
    
    def trendWA(self, df, prices, window_size=5):
        results = []
        df.index = pd.to_datetime(df.index)

        unique_days = pd.to_datetime(df.index , format='%m/%d/%Y').unique()

        for day in unique_days:
            day = day.strftime('%-m/%-d/%Y')
  
            first_occurrence_index = df.index.get_loc(day)

            if self.name == "Hour":
                window_data = prices[first_occurrence_index.start-window_size:first_occurrence_index.start]
            else:
                window_data = prices[first_occurrence_index-window_size:first_occurrence_index]

            if len(window_data) < window_size:
                trend = 0  
            else:
                trend = self.check_trend(window_data)

            results.append({'Date': day, 'Trend': trend})
            # print("=============")
        return results


    
    def writeFile(self, file_name):

        self.spTimeserie.set_index('Date', inplace=True)

        assert trend_type == "MACD" or trend_type == "TrendWA" or trend_type == "TrendWAWithTwoPointScaler" or  trend_type == "TrendWAWithLinearScaler" or trend_type == "TrendWAWithTwoPoint" or trend_type == "TrendWAWithLinear", "trend_type must be \"MACD\" or \"TrendWA\""
        
        if trend_type == "TrendWA":
            trendResult = self.trendWA(df = self.spTimeserie, prices = self.Close , window_size=5)
            # print(trendResult)

        ensambleValid = pd.DataFrame(trendResult)
        ensambleValid.set_index('Date', inplace=True) 

        print(ensambleValid)
        ensambleValid.to_csv(file_name)