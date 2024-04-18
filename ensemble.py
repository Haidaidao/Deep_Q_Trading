# #Copyright (C) 2020 Salvatore Carta, Anselmo Ferreira, Alessandro Sebastian Podda, Diego Reforgiato Recupero, Antonio Sanna. All rights reserved.

import csv
from os import path
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score
from itertools import product

from sklearn.model_selection import train_test_split
import global_config

from sklearn.preprocessing import LabelEncoder
import global_config
import json
from Evaluation import Evaluation
import re

config = json.load(open('plotResultsConf.json', 'r'))

ensembleFolder = global_config.ensembleFolder
threshold = global_config.label_threshold

iteration = 'iteration' + str(config['epoch']-1)
# ================================================ Result New State
# def ResultNewState(numWalks,type,numDel):
#     wins_number = 0
#     loses_number = 0
#     profit_number = 0

#     columns = ["From","To", "Wins", "Closes", "Profit"]

#     values=[]

#     dax = pd.read_csv("./datasets/" + global_config.MK + "Day.csv", index_col='Date')

#     daxHour = pd.read_csv("./datasets/"+ global_config.MK +"Hour.csv", index_col='Date')
#     daxHour = daxHour.reset_index()
#     daxHour['Date'] = pd.to_datetime(daxHour['Date'] + ' ' + daxHour['Time'])
#     daxHour['Date'] = daxHour['Date'].dt.strftime('%m/%d/%Y %H:%M')
#     daxHour.set_index('Date', inplace=True)

#     for j in range(0,numWalks):
        
#         df_fn = path.join('./Output/ensemble', f'walkHour{str(j)}ensemble_{type}.csv')

#         df=pd.read_csv(df_fn, index_col='Date')

        
#         from_date=str(df.index[0])
#         to_date=str(df.index[len(df)-1])

#         for deleted in range(1,numDel):
#             del df['iteration'+str(deleted)]

            
#         df = pd.DataFrame(df[iteration])
#         df.rename(columns={iteration: 'ensemble'}, inplace=True)

#         for deleted in range(1,numDel):
#             del df['iteration'+str(deleted)]

#         df['close'] = df.index.map(daxHour['Close'])
#         df['open'] = df.index.map(daxHour['Open'])
#         df['high'] = df.index.map(daxHour['High'])
#         df['low'] = df.index.map(daxHour['Low'])

#         eva = Evaluation(df)
#         wins, loses, profit = eva.evaluate()
#         values.append([from_date, to_date,str(round(wins,2)),str(round(loses,2)),str(round(profit,2))])

#         wins_number+=wins
#         loses_number+=loses
#         profit_number+=profit


#     values.append([' ','Sum',str(round(wins_number,2)),str(round(loses_number,2)),str(round(profit_number,2))])
    
#     # print(values)
#     return values,columns

def ResultNewState(numWalks,type,numDel):
    dollSum=0
    rewSum=0
    posSum=0
    negSum=0
    covSum=0
    numSum=0

    columns = ["From","To", "Reward%", "#Wins", "#Losses", "Dollars", "Coverage", "Accuracy"]

    values=[]

    dax = pd.read_csv("./datasets/" + global_config.MK + "Hour.csv")
    dax['Date'] = pd.to_datetime(dax['Date'] + ' ' + dax['Time'])
    dax.set_index('Date', inplace=True)
    dax.index = dax.index.strftime('%m/%d/%Y %H:%M')
   

    for j in range(0,numWalks):
        
        df_fn = path.join('./Output/ensemble', f'walkHour{str(j)}ensemble_{type}.csv')

        df=pd.read_csv(df_fn, index_col='Date')

        
        from_date=str(df.index[0])
        to_date=str(df.index[len(df)-1])

        for deleted in range(1,numDel):
            del df['iteration'+str(deleted)]

            
        df = pd.DataFrame(df[iteration])
        df.rename(columns={iteration: 'ensemble'}, inplace=True)

        for deleted in range(1,numDel):
            del df['iteration'+str(deleted)]

        num=0
        rew=0
        pos=0
        neg=0
        doll=0
        cov=0
        for date, i in df.iterrows():
            num+=1

            if date in dax.index:

                if (i['ensemble']==1):
                    pos+= 1 if (dax.at[date,'Close']-dax.at[date,'Open'])/dax.at[date,'Open'] > 0 else 0
                    neg+= 0 if (dax.at[date,'Close']-dax.at[date,'Open'])/dax.at[date,'Open'] > 0 else 1
                    rew+=(dax.at[date,'Close']-dax.at[date,'Open'])/dax.at[date,'Open']
                    doll+=(dax.at[date,'Close']-dax.at[date,'Open'])*50
                    cov+=1
                elif (i['ensemble']==-1):
                    neg+= 0 if -(dax.at[date,'Close']-dax.at[date,'Open'])/dax.at[date,'Open'] > 0 else 1
                    pos+= 1 if -(dax.at[date,'Close']-dax.at[date,'Open'])/dax.at[date,'Open'] > 0 else 0
                    rew+=-(dax.at[date,'Close']-dax.at[date,'Open'])/dax.at[date,'Open']
                    cov+=1
                    doll+=-(dax.at[date,'Close']-dax.at[date,'Open'])*50
            
        values.append([from_date, to_date,str(round(rew,2)),str(round(pos,2)),str(round(neg,2)),str(round(doll,2)),str(round(cov/num,2)),(str(round(pos/cov,2)) if (cov>0) else "None")])

        dollSum+=doll
        rewSum+=rew
        posSum+=pos
        negSum+=neg
        covSum+=cov
        numSum+=num

    values.append([' ','Sum',str(round(rewSum,2)),str(round(posSum,2)),str(round(negSum,2)),str(round(dollSum,2)),str(round(covSum/numSum,2)),(str(round(posSum/covSum,2)) if (covSum>0) else "None")])
    
    # values.append("","",0,0,0,0,0,0)
    # print(values)
    return values,columns
