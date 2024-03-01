import numpy as np
import pandas as pd

import global_config
import trendIdentifier
import trendStrengthIdentifier

MK = global_config.MK

identify_df_trends = trendIdentifier.use_TrendWA

class TrendBase:
    def __init__(self, ensembleFolderName, file_name, frame_type, iteration = None, minLimit=None, maxLimit=None, type = "test", columnName = "trend", ):
        self.spTimeserie = pd.read_csv('./datasets/'+ file_name + '.csv')[minLimit:maxLimit+1]
        self.minlimit = minLimit
        self.maxLimit = maxLimit
        self.Date = self.spTimeserie.loc[:, 'Datetime'].tolist()
        self.Open = self.spTimeserie.loc[:, 'Open'].tolist()
        self.High = self.spTimeserie.loc[:, 'High'].tolist()
        self.Low = self.spTimeserie.loc[:, 'Low'].tolist()
        self.Close = self.spTimeserie.loc[:, 'Close'].tolist()

        self.columnName = columnName
        self.frame_type = frame_type
        self.iteration = iteration
        self.type = type
        self.ensembleFolderName = ensembleFolderName


    def process(self):
        ensembleValid=pd.DataFrame()
        ensembleValid.index.name='Datetime'
        self.spTimeserie.set_index('Datetime', inplace=True)
        trendResult = identify_df_trends(df = self.spTimeserie, window_size=5)

        for i in range(0,len(self.Date)):
            ensembleValid.at[trendResult.index[i],self.columnName]=trendResult['trend'][i]
        ensembleValid.to_csv("./Output/ensemble/"+self.ensembleFolderName +"/walk"+self.frame_type+str(self.iteration)+"ensemble_"+self.type+".csv")