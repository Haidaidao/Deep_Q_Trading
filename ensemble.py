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

MK = global_config.MK
ensembleFolder = global_config.ensemble_folder
threshold = global_config.label_threshold

def full_ensemble(df):
    print("full")
    m1 = df.eq(1).all(axis=1)

    m2 = df.eq(-1).all(axis=1)

    local_df = df.copy()
    local_df['ensemble'] = np.select([m1, m2], [1, -1], 0)

    local_df = local_df.drop(local_df.columns.difference(['ensemble']), axis=1)

    return local_df

def perc_ensemble(df, thr = 0.7):
    print(thr)
    c1 = (df.eq(1).sum(1) / df.shape[1]).gt(thr)
    c2 = (df.eq(-1).sum(1) / df.shape[1]).gt(thr)
    return pd.DataFrame(np.select([c1, c2], [1, -1], 0), index=df.index, columns=['ensemble'])


# Get the results in the log file of the week that contains that day.
def getActionWeek(weeksFrame, date):
    date = datetime.strptime(date,"%m/%d/%Y")

    for i in range(0, len(weeksFrame)):
        week =  datetime.strptime(str(weeksFrame.index[i]),"%m/%d/%Y")
        if week == date:
            print(week)
            return  weeksFrame['ensemble'][i]
        elif week>date:
            print(week)
            return  weeksFrame['ensemble'][i-1]
    return 0

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
            
    return labels


# ================================================ Result New State
def BaseRule(numWalks,perc,type,numDel):
    dollSum=0
    rewSum=0
    posSum=0
    negSum=0
    covSum=0
    numSum=0

    columns = ["From","To", "Reward%", "#Wins", "#Losses", "Rotation" ,"Dollars", "Coverage", "Accuracy"]

    values=[]

    dax = pd.read_csv("./datasets/" + global_config.MK + "Day.csv", index_col='Date')

    for j in range(0,numWalks):

        df1=pd.read_csv(f"./Output/ensemble/walk"+"Hour"+str(j)+"ensemble_"+type+".csv",index_col='Date')
        df2=pd.read_csv(f"./Output/trend/{MK}Day"+".csv",index_col='Date')
        df3=pd.read_csv(f"./Output/trend/{MK}Week"+".csv",index_col='Date')

        from_date=str(df2.index[0])
        to_date=str(df2.index[len(df2)-1])

        for deleted in range(1,numDel):
            del df1['iteration'+str(deleted)]
            del df2['iteration'+str(deleted)]
            del df3['iteration'+str(deleted)]

        if perc==0:
            df1=full_ensemble(df1)
        else:
            df1=perc_ensemble(df1,perc)


            
        # df1 = pd.DataFrame(df1[iteration])
        # df1.rename(columns={iteration: 'ensemble'}, inplace=True)

        df2.index = pd.to_datetime(df2.index)
        df2.index = df2.index.strftime('%m/%d/%Y')
        df2.rename(columns={'trend': 'ensemble'}, inplace=True)

        df3.index = pd.to_datetime(df3.index)
        df3.index = df3.index.strftime('%m/%d/%Y')
        df3.rename(columns={'trend': 'ensemble'}, inplace=True)

        print(df3)
        print("================")

        for deleted in range(1,numDel):
            del df['iteration'+str(deleted)]

        df = pd.DataFrame(columns=['ensemble'])
        df = df.set_index(pd.Index([], name='date'))

        for k in range(0,len(df1)):
            if(df1.index[k] in df2.index):
                # print(df1.index[k])
                # print(getActionWeek(df2, df1.index[k]))
                # print(df2.loc[df1.index[k],'ensemble'])
                # print(getActionWeek(df3, df2.index[k]))
                # print("========================================")
                if df1['ensemble'][k] == 0:
                    df.loc[df1.index[k]] = 0
                else:
                    if df1['ensemble'][k] == getActionWeek(df3, df2.index[k]) or df1['ensemble'][k] == df2.loc[df1.index[k],'ensemble']: 
                        if  getActionWeek(df3, df2.index[k]) == df2.loc[df1.index[k],'ensemble']  and df2.loc[df1.index[k],'ensemble'] != df1['ensemble'][k] :
                            df.loc[df1.index[k]] = 0
                        elif df2.loc[df1.index[k],'ensemble'] == 0 and getActionWeek(df3, df2.index[k]) !=0:
                            df.loc[df1.index[k]] = getActionWeek(df3, df2.index[k])
                        elif df2.loc[df1.index[k],'ensemble'] != 0 and getActionWeek(df3, df2.index[k]) ==0:
                            df.loc[df1.index[k]] = df2.loc[df1.index[k],'ensemble']
                        elif getActionWeek(df3, df2.index[k]) != df2.loc[df1.index[k],'ensemble']:
                            df.loc[df1.index[k]] = 0
                        elif getActionWeek(df3, df2.index[k]) == df2.loc[df1.index[k],'ensemble']:
                            df.loc[df1.index[k]] = df2.loc[df1.index[k],'ensemble']
                        else:
                            df.loc[df1.index[k]] = 0
                    else: 
                        df.loc[df1.index[k]] = 0

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

        values.append([from_date, to_date,str(round(rew,2)),str(round(pos,2)),str(round(neg,2)),"",str(round(doll,2)),str(round(cov/num,2)),(str(round(pos/cov,2)) if (cov>0) else "None")])

        dollSum+=doll
        rewSum+=rew
        posSum+=pos
        negSum+=neg
        covSum+=cov
        numSum+=num


    values.append([' ','Sum',str(round(rewSum,2)),str(round(posSum,2)),str(round(negSum,2)),str(round(posSum/negSum,2)),str(round(dollSum,2)),str(round(covSum/numSum,2)),(str(round(posSum/covSum,2)) if (covSum>0) else "None")])
    # print(values)
    return values,columns