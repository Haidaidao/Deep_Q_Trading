# #Copyright (C) 2020 Salvatore Carta, Anselmo Ferreira, Alessandro Sebastian Podda, Diego Reforgiato Recupero, Antonio Sanna. All rights reserved.

import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import xgboost as xgb
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
    m1 = df.eq(1).all(axis=1)

    m2 = df.eq(-1).all(axis=1)

    local_df = df.copy()
    local_df['ensemble'] = np.select([m1, m2], [1, -1], 0)

    local_df = local_df.drop(local_df.columns.difference(['ensemble']), axis=1)

    return local_df

def perc_ensemble(df, thr = 0.7):
    c1 = (df.eq(1).sum(1) / df.shape[1]).gt(thr)
    c2 = (df.eq(-1).sum(1) / df.shape[1]).gt(thr)
    return pd.DataFrame(np.select([c1, c2], [1, -1], 0), index=df.index, columns=['ensemble'])


# Get the results in the log file of the week that contains that day.
def getAction(Frame, date, name="df2"):
    date = datetime.strptime(date,"%m/%d/%Y")
    for i in range(0, len(Frame)):
        result =  datetime.strptime(str(Frame.index[i]),"%m/%d/%Y")
        if result >= date:
            return  Frame['ensemble'][i]
        # elif result>date:
        #     return  Frame['ensemble'][i-1]
    
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
        elif changes < -threshold or (last_action == -1 and changes < 0 and changes >= -threshold):
            last_action = -1
            action = -1
        else:
            last_action = 0

        labels.append(action)
            
    return labels
# ================================================ XGBoots
def XGBoostEnsemble(numWalks,type,numDel):
    
    dollSum = 0
    rewSum = 0
    posSum = 0
    negSum = 0
    covSum = 0
    numSum = 0

    columns = ["From","To", "Reward%", "#Wins", "#Losses", "Dollars", "Coverage", "Accuracy"]

    values = []

    dax=pd.read_csv("./datasets/"+ global_config.MK +"Day.csv",index_col='Date')

    type_train = "train"

    xgb_model = xgb.XGBClassifier(n_estimators=100, random_state=42)

    for j in range(0, numWalks):

        # Train
        df1 = pd.read_csv(f"./Output/ensemble/{ensembleFolder}/walk" + "Hour" + str(j) + "ensemble_" + type_train+ ".csv",
                          index_col='Date')
        df2 = pd.read_csv(f"./Output/ensemble/{ensembleFolder}/walk" + "Day" + str(j) + "ensemble_" + type_train + ".csv",
                          index_col='Date')
        df3 = pd.read_csv(f"./Output/ensemble/{ensembleFolder}/walk" + "Week" + str(j) + "ensemble_" + type_train + ".csv",
                          index_col='Date')

        for deleted in range(1, numDel):
            del df1['iteration' + str(deleted)]
            del df2['iteration' + str(deleted)]
            del df3['iteration' + str(deleted)]

        df1 = pd.DataFrame(df1[iteration])
        df1.rename(columns={iteration: 'ensemble'}, inplace=True)

        df2.index = pd.to_datetime(df2.index)
        df2.index = df2.index.strftime('%m/%d/%Y')
        df2.rename(columns={'trend': 'ensemble'}, inplace=True)

        df3.index = pd.to_datetime(df3.index)
        df3.index = df3.index.strftime('%m/%d/%Y')
        df3.rename(columns={'trend': 'ensemble'}, inplace=True)

        df3_temp = pd.DataFrame(index=df2.index).assign(ensemble=0)
        for k in range(0,len(df3_temp)):
            df3_temp['ensemble'][k] = getAction(df3,df3_temp.index[k])

        list_combine_train = np.empty((0, 3))

        for k in range(0,len(df1)):
            list_combine_train = np.append(list_combine_train, [[df1['ensemble'][k], df2['ensemble'][k], df3_temp['ensemble'][k]]], axis=0)
      
        y_train = ensemble_y_true(df1, dax, threshold)

        le = LabelEncoder()
        y_train = le.fit_transform(y_train)
        # xgb_model = xgb.XGBClassifier(n_estimators=100, random_state=42)
        xgb_model.fit(list_combine_train, y_train)

        # Predict
        for deleted in range(1,numDel):
            del df['iteration'+str(deleted)]

        df = pd.DataFrame(columns=['ensemble'])
        df = df.set_index(pd.Index([], name='date'))

        df1_result = pd.read_csv(f"./Output/ensemble/{ensembleFolder}/walk" + "Hour" + str(j) + "ensemble_" + type + ".csv",
                          index_col='Date')
        df2_result = pd.read_csv(f"./Output/ensemble/{ensembleFolder}/walk" + "Day" + str(j) + "ensemble_" + type + ".csv",
                          index_col='Date')
        df3_result = pd.read_csv(f"./Output/ensemble/{ensembleFolder}/walk" + "Week" + str(j) + "ensemble_" + type + ".csv",
                          index_col='Date')

        from_date=str(df2_result.index[0])
        to_date=str(df2_result.index[len(df2_result)-1])

        for deleted in range(1, numDel):
            del df1_result['iteration' + str(deleted)]
            del df2_result['iteration' + str(deleted)]
            del df3_result['iteration' + str(deleted)]

        df1_result = pd.DataFrame(df1_result[iteration])
        df1_result.rename(columns={iteration: 'ensemble'}, inplace=True)

        df2_result.index = pd.to_datetime(df2_result.index)
        df2_result.index = df2_result.index.strftime('%m/%d/%Y')
        df2_result.rename(columns={'trend': 'ensemble'}, inplace=True)

        df3_result.index = pd.to_datetime(df3_result.index)
        df3_result.index = df3_result.index.strftime('%m/%d/%Y')
        df3_result.rename(columns={'trend': 'ensemble'}, inplace=True)

        df3_temp = pd.DataFrame(index=df2_result.index).assign(ensemble=0)
        for k in range(0,len(df3_temp)):
            df3_temp['ensemble'][k] = getAction(df3_result,df3_temp.index[k])

        for k in range(0,len(df1_result)):
            if(df1_result.index[k] in df2_result.index):
                new_data = np.array([[df1_result['ensemble'][k], df2_result['ensemble'][k], df3_temp['ensemble'][k]]])
                predicted_result = xgb_model.predict(new_data)
                df.loc[df1_result.index[k]] = predicted_result[0]

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
                elif (i['ensemble']==2):
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
# ================================================ Random Forest
def RandomForestEnsemble(numWalks,perc,type,numDel):
    dollSum = 0
    rewSum = 0
    posSum = 0
    negSum = 0
    covSum = 0
    numSum = 0

    columns = ["From","To", "Reward%", "#Wins", "#Losses", "Rotation" ,"Dollars", "Coverage", "Accuracy"]


    values = []

    dax=pd.read_csv("./datasets/"+ global_config.MK +"Day.csv",index_col='Date')

    type_train = "train"

    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)

    for j in range(0, numWalks):
        # Train
        df1 = pd.read_csv(f"./Output/ensemble/{ensembleFolder}/walk" + "Hour" + str(j) + "ensemble_" + type_train+ ".csv",
                          index_col='Date')
        df2=pd.read_csv(f"./Output/trend/{MK}Day"+".csv",index_col='Date')
        df3=pd.read_csv(f"./Output/trend/{MK}Week"+".csv",index_col='Date')

        for deleted in range(1, numDel):
            del df1['iteration' + str(deleted)]
            del df2['iteration' + str(deleted)]
            del df3['iteration' + str(deleted)]

        if perc==0:
            df1=full_ensemble(df1)
        else:
            df1=perc_ensemble(df1,perc)

        df2.index = pd.to_datetime(df2.index)
        df2.index = df2.index.strftime('%m/%d/%Y')
        df2.rename(columns={'trend': 'ensemble'}, inplace=True)

        df3.index = pd.to_datetime(df3.index)
        df3.index = df3.index.strftime('%m/%d/%Y')
        df3.rename(columns={'trend': 'ensemble'}, inplace=True)

        df3_temp = pd.DataFrame(index=df1.index).assign(ensemble=0)
        for k in range(0,len(df3_temp)):
            df3_temp['ensemble'][k] = getAction(df3,df3_temp.index[k],"df3")

        df2_temp = pd.DataFrame(index=df1.index).assign(ensemble=0)
        for k in range(0,len(df2_temp)):
            df2_temp['ensemble'][k] = getAction(df2,df2_temp.index[k],"df2")

        list_combine_train = np.empty((0, 3))

        for k in range(0,len(df1)):
            list_combine_train = np.append(list_combine_train, [[df1['ensemble'][k], df2_temp['ensemble'][k], df3_temp['ensemble'][k]]], axis=0)

        y_train = ensemble_y_true(df1, dax, threshold)

        # rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
        rf_model.fit(list_combine_train, y_train)

        # Predict
        for deleted in range(1,numDel):
            del df['iteration'+str(deleted)]

        df = pd.DataFrame(columns=['ensemble'])
        df = df.set_index(pd.Index([], name='date'))

        df1_result = pd.read_csv(f"./Output/ensemble/{ensembleFolder}/walk" + "Hour" + str(j) + "ensemble_" + type + ".csv",
                          index_col='Date')
        # df2_result = pd.read_csv(f"./Output/ensemble/{ensembleFolder}/walk" + "Day" + str(j) + "ensemble_" + type + ".csv",
        #                   index_col='Date')
        # df3_result = pd.read_csv(f"./Output/ensemble/{ensembleFolder}/walk" + "Week" + str(j) + "ensemble_" + type + ".csv",
        #                   index_col='Date')

        from_date=str(df2.index[0])
        to_date=str(df2.index[len(df2)-1])

        for deleted in range(1, numDel):
            del df1_result['iteration' + str(deleted)]
            # del df2_result['iteration' + str(deleted)]
            # del df3_result['iteration' + str(deleted)]

        if perc==0:
            df1_result=full_ensemble(df1_result)
        else:
            df1_result=perc_ensemble(df1_result,perc)

        df3_temp = pd.DataFrame(index=df1_result.index).assign(ensemble=0)
        for k in range(0,len(df3_temp)):
            df3_temp['ensemble'][k] = getAction(df3,df3_temp.index[k],"df3")

        df2_temp = pd.DataFrame(index=df1_result.index).assign(ensemble=0)
        for k in range(0,len(df2_temp)):
            df2_temp['ensemble'][k] = getAction(df2,df2_temp.index[k],"df2")

        for k in range(0,len(df1_result)):
            if(df1_result.index[k] in df2_temp.index):
                new_data = np.array([[df1_result['ensemble'][k], df2_temp['ensemble'][k], df3_temp['ensemble'][k]]])
                predicted_result = rf_model.predict(new_data)
                df.loc[df1_result.index[k]] = predicted_result[0]

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
                elif (i['ensemble']==2):

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
# ================================================ Random

def RandomForestEnsemble(numWalks,perc,type,numDel):
    dollSum = 0
    rewSum = 0
    posSum = 0
    negSum = 0
    covSum = 0
    numSum = 0

    columns = ["From","To", "Reward%", "#Wins", "#Losses", "Rotation" ,"Dollars", "Coverage", "Accuracy"]

    values = []

    dax=pd.read_csv("./datasets/"+ global_config.MK +"Day.csv",index_col='Date')

    type_train = "train"

    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)

    for j in range(0, numWalks):
        # Train
        df1 = pd.read_csv(f"./Output/ensemble/{ensembleFolder}/walk" + "Hour" + str(j) + "ensemble_" + type_train+ ".csv",
                          index_col='Date')
        df2=pd.read_csv(f"./Output/trend/{MK}Day"+".csv",index_col='Date')
        df3=pd.read_csv(f"./Output/trend/{MK}Week"+".csv",index_col='Date')

        for deleted in range(1, numDel):
            del df1['iteration' + str(deleted)]
            del df2['iteration' + str(deleted)]
            del df3['iteration' + str(deleted)]

        if perc==0:
            df1=full_ensemble(df1)
        else:
            df1=perc_ensemble(df1,perc)

        df2.index = pd.to_datetime(df2.index)
        df2.index = df2.index.strftime('%m/%d/%Y')
        df2.rename(columns={'trend': 'ensemble'}, inplace=True)

        df3.index = pd.to_datetime(df3.index)
        df3.index = df3.index.strftime('%m/%d/%Y')
        df3.rename(columns={'trend': 'ensemble'}, inplace=True)

        df3_temp = pd.DataFrame(index=df1.index).assign(ensemble=0)
        for k in range(0,len(df3_temp)):
            df3_temp['ensemble'][k] = getAction(df3,df3_temp.index[k],"df3")

        df2_temp = pd.DataFrame(index=df1.index).assign(ensemble=0)
        for k in range(0,len(df2_temp)):
            df2_temp['ensemble'][k] = getAction(df2,df2_temp.index[k],"df2")


        list_combine_train = np.empty((0, 3))

        for k in range(0,len(df1)):
            list_combine_train = np.append(list_combine_train, [[df1['ensemble'][k], df2_temp['ensemble'][k], df3_temp['ensemble'][k]]], axis=0)

        y_train = ensemble_y_true(df1, dax, threshold)

        # rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
        rf_model.fit(list_combine_train, y_train)

        # Predict
        for deleted in range(1,numDel):
            del df['iteration'+str(deleted)]

        df = pd.DataFrame(columns=['ensemble'])
        df = df.set_index(pd.Index([], name='date'))

        df1_result = pd.read_csv(f"./Output/ensemble/{ensembleFolder}/walk" + "Hour" + str(j) + "ensemble_" + type + ".csv",
                          index_col='Date')

        from_date=str(df1_result.index[0])
        to_date=str(df1_result.index[len(df1_result)-1])

        for deleted in range(1, numDel):
            del df1_result['iteration' + str(deleted)]

        if perc==0:
            df1_result=full_ensemble(df1_result)
        else:
            df1_result=perc_ensemble(df1_result,perc)

        df3_temp = pd.DataFrame(index=df1_result.index).assign(ensemble=0)
        for k in range(0,len(df3_temp)):
            df3_temp['ensemble'][k] = getAction(df3,df3_temp.index[k],"df3")

        df2_temp = pd.DataFrame(index=df1_result.index).assign(ensemble=0)
        for k in range(0,len(df2_temp)):
            df2_temp['ensemble'][k] = getAction(df2,df2_temp.index[k],"df2")

        for k in range(0,len(df1_result)):
            if(df1_result.index[k] in df2_temp.index):
                new_data = np.array([[df1_result['ensemble'][k], df2_temp['ensemble'][k], df3_temp['ensemble'][k]]])
                predicted_result = rf_model.predict(new_data)
                df.loc[df1_result.index[k]] = predicted_result[0]

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

    return values,columns

# ================================================ Base Rule
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

        from_date=str(df1.index[0])
        to_date=str(df1.index[len(df1)-1])

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


        for deleted in range(1,numDel):
            del df['iteration'+str(deleted)]

        df = pd.DataFrame(columns=['ensemble'])
        df = df.set_index(pd.Index([], name='date'))

        UP = 1; DOWN = -1; SIDEWAY = 0
        LONG = 1; SHORT = -1; HOLD = 0

        rules = {
            (LONG, UP, UP): LONG,
            (SHORT, UP, UP): LONG,
            (HOLD, UP, UP): LONG,
            

            (LONG, DOWN, DOWN): SHORT, 
            (SHORT, DOWN, DOWN): SHORT, 
            (HOLD, DOWN, DOWN): SHORT, 

            (LONG, SIDEWAY, SIDEWAY): LONG, 
            (SHORT, SIDEWAY, SIDEWAY): SHORT, 
            (HOLD, SIDEWAY, SIDEWAY): HOLD,

            (LONG, UP, SIDEWAY): LONG,
            (LONG, SIDEWAY, UP): LONG, 

            (SHORT, DOWN, SIDEWAY): SHORT,
            (SHORT, SIDEWAY, DOWN): SHORT, 

            (HOLD, UP, SIDEWAY): HOLD,
            (HOLD, SIDEWAY, UP): HOLD, 
            (HOLD, DOWN, SIDEWAY): HOLD,
            (HOLD, SIDEWAY, DOWN): HOLD, 

            (LONG, DOWN, UP): HOLD, 
            (SHORT, DOWN, UP): HOLD, 
            (HOLD, DOWN, UP): HOLD, 

            (LONG, UP, DOWN): HOLD, 
            (SHORT, UP, DOWN): HOLD, 
            (HOLD, UP, DOWN): HOLD, 
        }

        for k in range(0,len(df1)):
            if(df1.index[k] in df2.index):
                key = (df1['ensemble'][k], int(df2.loc[df1.index[k],'ensemble']), int(getAction(df3, df2.index[k])))
                df.loc[df1.index[k]] = rules.get(key, 0)
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

    return values,columns

# ================================================ Rule author

def EnsembleAuthor(numWalks,perc,type,numDel):
    dollSum=0
    rewSum=0
    posSum=0
    negSum=0
    covSum=0
    numSum=0

    values=[]
    #output=open("daxValidDel9th60.csv","w+")
    #output.write("Iteration,Reward%,#Wins,#Losses,Euro,Coverage,Accuracy\n")
    columns = ["Sum","Reward%", "#Wins", "#Losses", "Rotation" ,"Dollars", "Coverage", "Accuracy"]

    dax = pd.read_csv("./datasets/" + global_config.MK + "Day.csv", index_col='Date')
    for j in range(0,numWalks):

        df=pd.read_csv(f"./Output/ensemble/walk"+"Hour"+str(j)+"ensemble_"+type+".csv",index_col='Date')

        for deleted in range(1,numDel):
            del df['iteration'+str(deleted)]
        
        if perc==0:
            df=full_ensemble(df)
        else:
            df=perc_ensemble(df,perc)

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
        
        values.append(["",str(round(rew,2)),str(round(pos,2)),str(round(neg,2)),"",str(round(doll,2)),str(round(cov/num,2)),(str(round(pos/cov,2)) if (cov>0) else "None")])

        
        dollSum+=doll
        rewSum+=rew
        posSum+=pos
        negSum+=neg
        covSum+=cov
        numSum+=num


    values.append(["Sum",str(round(rewSum,2)),str(round(posSum,2)),str(round(negSum,2)),str(round(posSum/negSum,2)),str(round(dollSum,2)),str(round(covSum/numSum,2)),(str(round(posSum/covSum,2)) if (covSum>0) else "None")])
    return values,columns

