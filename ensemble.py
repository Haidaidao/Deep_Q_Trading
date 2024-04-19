# #Copyright (C) 2020 Salvatore Carta, Anselmo Ferreira, Alessandro Sebastian Podda, Diego Reforgiato Recupero, Antonio Sanna. All rights reserved.

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

from os import path

config = json.load(open('config.json', 'r'))

iteration = 'iteration' + str(config['epoch']-1)


ensembleFolder = global_config.ensemble_folder
threshold = global_config.label_threshold


# Get the results in the log file of the week that contains that day.
def getActionWeek(weeksFrame, date):
    date = datetime.strptime(date,"%m/%d/%Y")

    for i in range(0, len(weeksFrame)):
        week =  datetime.strptime(str(weeksFrame.index[i]),"%m/%d/%Y")
        if week>=date:
            return  weeksFrame['ensemble'][i]
    return 0

# def ensemble_y_true(df1, df2, df3):

#     df = pd.DataFrame(columns=['ensemble'])
#     df = df.set_index(pd.Index([], name='date'))

#     for k in range(0,len(df1)):
#         if(df1.index[k] in df2.index):
#             if df1['ensemble'][k] == 0:
#                 df.loc[df1.index[k]] = 0
#             else:
#                 if df1['ensemble'][k] == getActionWeek(df3, df2.index[k]) or df1['ensemble'][k] == df2.loc[df1.index[k],'ensemble']: 
#                     if  getActionWeek(df3, df2.index[k]) == df2.loc[df1.index[k],'ensemble']  and df2.loc[df1.index[k],'ensemble'] != df1['ensemble'][k] :
#                         df.loc[df1.index[k]] = 0
#                     elif df2.loc[df1.index[k],'ensemble'] == 0 and getActionWeek(df3, df2.index[k]) !=0:
#                         df.loc[df1.index[k]] = getActionWeek(df3, df2.index[k])
#                     elif df2.loc[df1.index[k],'ensemble'] != 0 and getActionWeek(df3, df2.index[k]) ==0:
#                         df.loc[df1.index[k]] = df2.loc[df1.index[k],'ensemble']
#                     elif getActionWeek(df3, df2.index[k]) != df2.loc[df1.index[k],'ensemble']:
#                         df.loc[df1.index[k]] = 0
#                     elif getActionWeek(df3, df2.index[k]) == df2.loc[df1.index[k],'ensemble']:
#                         df.loc[df1.index[k]] = df2.loc[df1.index[k],'ensemble']
#                     else:
#                         df.loc[df1.index[k]] = 0
#                 else: 
#                     df.loc[df1.index[k]] = 0

#     return df['ensemble'].tolist()

def ensemble_y_true(feature, stats, threshold):

    labels = []
    last_action = 0
    skip = 0
    
    for index, _ in feature.iterrows():
        if index not in stats.index:
            labels.append(last_action)
            skip += 1
            continue
        
        close = stats.loc[index, 'Close']
        open = stats.loc[index, 'Open']
        
        action = 0
        changes = (close - open) / open

        if changes >= threshold or (last_action == 1 and changes >= 0 and changes < threshold):
            last_action = 1
            action = 1
        elif changes < -threshold or (last_action == 2 and changes < 0 and changes >= -threshold):
            last_action = 2
            action = 2
        else:
            last_action = 0

        labels.append(action)

    # for index, row in stats.iterrows():
    #     if index not in feature1.index and index not in feature2.index and index not in feature3.index:
    #         continue
    #     action = 0
    #     changes = (row['Close'] - row['Open']) / row['Open']

    #     if changes >= threshold or (last_action == 1 and changes >= 0 and changes < threshold):
    #         last_action = 1
    #         action = 1
    #     elif changes < -threshold or (last_action == 2 and changes < 0 and changes >= -threshold):
    #         last_action = 2
    #         action = 2
    #     else:
    #         last_action = 0
            
    #     labels.append(action)

    # with open('test.txt', 'w') as f:
    #     f.write('\n'.join(map(str, labels)))
            
    return labels


# ================================================ Result New State
def ResultNewState(numWalks,type,numDel):
    dollSum=0
    rewSum=0
    posSum=0
    negSum=0
    covSum=0
    numSum=0

    columns = ["From","To", "Reward%", "#Wins", "#Losses", "Dollars", "Coverage", "Accuracy"]

    values=[]

    dax = pd.read_csv("./datasets/" + global_config.MK + "Day.csv", index_col='Date')

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
    # print(values)
    return values,columns