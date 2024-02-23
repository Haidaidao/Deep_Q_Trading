import gym
from gym import spaces
#Numpy is the library to deal with matrices
import numpy
#Pandas is the library used to deal with the CSV dataset
import pandas as pd
#datetime is the library used to manipulate time and date
from datetime import datetime

#Callback is the library used to show metrics 
import callback
import global_config
from decimal import Decimal

MK = global_config.MK

class MACD:
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
            return 2
        # SIDEWAY
        else:
            return 0

    def trend(self):
        trendResult = [] 
        macd , signal = self.calculate_MACD()
        for i in range(0,len(self.Date)):
            trendResult.append(self.analyze_market_trend(macd[i], signal[i]))
        df = pd.DataFrame({'ensemble': trendResult}, index=pd.to_datetime(self.Date))
        df.index = pd.to_datetime(df.index)
        df.index = df.index.strftime('%m/%d/%Y')
        df.rename(columns={'trend': 'ensemble'}, inplace=True)
        return df
    
    def writeFile(self):
        
        ensambleValid=pd.DataFrame()
        ensambleValid.index.name='Date'
        trendResult = self.trend()
        for i in range(0,len(self.Date)):
            ensambleValid.at[trendResult.index[i],self.columnName]=trendResult['ensemble'][i]
        ensambleValid.to_csv("./Output/ensemble/"+"ensembleFolder"+"/walk"+self.name+str(self.iteration)+"ensemble_"+self.type+".csv") 
        

