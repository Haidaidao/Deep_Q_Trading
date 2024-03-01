import numpy as np
import pandas as pd

import global_config
import trendIdentifier
import trendStrengthIdentifier

MK = global_config.MK

identify_df_trends = trendIdentifier.use_trendet
trend_add_delta = trendStrengthIdentifier.two_point_slope

class TrendSlope:
    def __init__(self, iteration = None, minLimit=None, maxLimit=None, name = "Week", type = "test", columnName = "trend", frame = "Long"):
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
        self.frame = frame 

    def trendAddDelta(self, trendArr):
        for i in range(0,len(self.Date)):  
            if trendArr[i]!=0:
                if i-4>=0:
                    delta = trend_add_delta(self.Close, i - 4, i, False)
                    trendArr[i] = delta
                else:
                    if trendArr[i] != 0:
                        trendArr[i] = 0

        return trendArr

    def writeFile(self):
        ensambleValid=pd.DataFrame()
        ensambleValid.index.name='Date'
        self.spTimeserie.set_index('Date', inplace=True)
        self.spTimeserie.index = pd.to_datetime(self.spTimeserie.index, format='%m/%d/%Y')
        trendResult = identify_df_trends(df = self.spTimeserie, column='Close' , window_size=5)

        trendResult['trend'] = self.trendAddDelta(trendResult['trend'].tolist())

        for i in range(0,len(self.Date)):
            ensambleValid.at[trendResult.index[i],self.columnName]=trendResult['trend'][i]
        ensambleValid.to_csv("./Output/ensemble/"+"ensembleFolder"+"/walk"+self.name+str(self.iteration)+"ensemble_"+self.type+".csv")