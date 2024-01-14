# #Copyright (C) 2020 Salvatore Carta, Anselmo Ferreira, Alessandro Sebastian Podda, Diego Reforgiato Recupero, Antonio Sanna. All rights reserved.

import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from itertools import product
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import global_config

def full_ensemble(df):
    m1 = df.eq(1).all(axis=1)

    m2 = df.eq(2).all(axis=1)

    local_df = df.copy()
    local_df['ensemble'] = np.select([m1, m2], [1, 2], 0)

    local_df = local_df.drop(local_df.columns.difference(['ensemble']), axis=1)

    return local_df

def perc_ensemble(df, thr = 0.7):
    c1 = (df.eq(1).sum(1) / df.shape[1]).gt(thr)
    c2 = (df.eq(2).sum(1) / df.shape[1]).gt(thr)
    return pd.DataFrame(np.select([c1, c2], [1, 2], 0), index=df.index, columns=['ensemble'])


def getActionWeek(weeksFrame, date):
    date = datetime.strptime(date,"%m/%d/%Y")

    for i in range(0, len(weeksFrame)):
        week =  datetime.strptime(str(weeksFrame.index[i]),"%m/%d/%Y")
        if week>=date:
            return  weeksFrame['ensemble'][i]
    return 0

def bin_predictions(predicted_value):
    if predicted_value <= 0.3:
        return 2
    elif predicted_value >= 0.7:
        return 0
    else:
        return 1
    
# Hàm tính toán trung bình có trọng số
def weighted_average(preds, weights):
    return np.dot(preds.T, weights)

def ensemble_y_true(df1, df2, df3):

    df = pd.DataFrame(columns=['ensemble'])
    df = df.set_index(pd.Index([], name='date'))

    for k in range(0,len(df1)):
        if(df1.index[k] in df2.index):
            if df1['ensemble'][k] == df2.loc[df1.index[k],'ensemble'] and getActionWeek(df3, df2.index[k]) == df2.loc[df1.index[k],'ensemble']:
                df.loc[df1.index[k]] = df1['ensemble'][k]
            elif getActionWeek(df3, df2.index[k]) == df2.loc[df1.index[k],'ensemble'] and df1['ensemble'][k] != df2.loc[df1.index[k],'ensemble']:
                df.loc[df1.index[k]] = df2.loc[df1.index[k],'ensemble']
            elif getActionWeek(df3, df2.index[k]) != df2.loc[df1.index[k],'ensemble']:
                df.loc[df1.index[k]] = 0
            else:
                df.loc[df1.index[k]] = 0
    return df['ensemble'].tolist()

def ensemble(numWalks,perc,type,numDel):
    dollSum = 0
    rewSum = 0
    posSum = 0
    negSum = 0
    covSum = 0
    numSum = 0

    columns = ["From","To", "Reward%", "#Wins", "#Losses", "Dollars", "Coverage", "Accuracy"]

    values = []

    dax = pd.read_csv("./datasets/" + global_config.MK + "Day.csv", index_col='Date')

    type_train = "train"

    for j in range(0, numWalks):
        # Train
        df1 = pd.read_csv("./Output/ensemble/ensembleFolder/walk" + "Hour" + str(j) + "ensemble_" + type_train+ ".csv",
                          index_col='Date')
        df2 = pd.read_csv("./Output/ensemble/ensembleFolder/walk" + "Day" + str(j) + "ensemble_" + type_train + ".csv",
                          index_col='Date')
        df3 = pd.read_csv("./Output/ensemble/ensembleFolder/walk" + "Week" + str(j) + "ensemble_" + type_train + ".csv",
                          index_col='Date')
        
        for deleted in range(1, numDel):
            del df1['iteration' + str(deleted)]
            del df2['iteration' + str(deleted)]
            del df3['iteration' + str(deleted)]

        if perc == 0:
            df1 = full_ensemble(df1)
        else:
            df1 = perc_ensemble(df1, perc)
        print(df1)
        df2.index = pd.to_datetime(df2.index)
        df2.index = df2.index.strftime('%m/%d/%Y')
        df2.rename(columns={'trend': 'ensemble'}, inplace=True)

        df3.index = pd.to_datetime(df3.index)
        df3.index = df3.index.strftime('%m/%d/%Y')
        df3.rename(columns={'trend': 'ensemble'}, inplace=True)

        df3_temp = pd.DataFrame(index=df2.index).assign(ensemble=0)
        for k in range(0,len(df3_temp)):
            df3_temp['ensemble'][k] = getActionWeek(df3,df3_temp.index[k])

        list_combine_train = np.empty((0, 3))

        for k in range(0,len(df1)):
            list_combine_train = np.append(list_combine_train, [[df1['ensemble'][k], df2['ensemble'][k], df3_temp['ensemble'][k]]], axis=0)

        y_train = ensemble_y_true(df1, df2, df3)
       
        rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
        rf_model.fit(list_combine_train, y_train)

        # Predict
        for deleted in range(1,numDel):
            del df['iteration'+str(deleted)]

        df = pd.DataFrame(columns=['ensemble'])
        df = df.set_index(pd.Index([], name='date'))

        df1_result = pd.read_csv("./Output/ensemble/ensembleFolder/walk" + "Hour" + str(j) + "ensemble_" + type + ".csv",
                          index_col='Date')
        df2_result = pd.read_csv("./Output/ensemble/ensembleFolder/walk" + "Day" + str(j) + "ensemble_" + type + ".csv",
                          index_col='Date')
        df3_result = pd.read_csv("./Output/ensemble/ensembleFolder/walk" + "Week" + str(j) + "ensemble_" + type + ".csv",
                          index_col='Date')
        
        from_date=str(df2_result.index[0])
        to_date=str(df2_result.index[len(df2_result)-1])
        
        for deleted in range(1, numDel):
            del df1_result['iteration' + str(deleted)]
            del df2_result['iteration' + str(deleted)]
            del df3_result['iteration' + str(deleted)]

        if perc == 0:
            df1_result = full_ensemble(df1_result)
        else:
            df1_result = perc_ensemble(df1_result, perc)

        df2_result.index = pd.to_datetime(df2_result.index)
        df2_result.index = df2_result.index.strftime('%m/%d/%Y')
        df2_result.rename(columns={'trend': 'ensemble'}, inplace=True)

        df3_result.index = pd.to_datetime(df3_result.index)
        df3_result.index = df3_result.index.strftime('%m/%d/%Y')
        df3_result.rename(columns={'trend': 'ensemble'}, inplace=True)

        df3_temp = pd.DataFrame(index=df2_result.index).assign(ensemble=0)
        for k in range(0,len(df3_temp)):
            df3_temp['ensemble'][k] = getActionWeek(df3_result,df3_temp.index[k])

        for k in range(0,len(df1_result)):
            if(df1_result.index[k] in df2_result.index):
                new_data = np.array([[df1_result['ensemble'][k], df2_result['ensemble'][k], df3_temp['ensemble'][k]]])
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
# ================================================ Sử Dụng Phương Pháp Trung Bình Có Trọng Số

# def ensemble(numWalks,perc,type,numDel):
#     dollSum = 0
#     rewSum = 0
#     posSum = 0
#     negSum = 0
#     covSum = 0
#     numSum = 0

#     columns = ["From","To", "Reward%", "#Wins", "#Losses", "Dollars", "Coverage", "Accuracy"]

#     values = []

#     dax = pd.read_csv("./datasets/daxDay.csv", index_col='Date')

#     from_date=""
#     to_date=""

#     for j in range(0, numWalks):
#         df1 = pd.read_csv("./Output/ensemble/ensembleFolder/walk" + "Hour" + str(j) + "ensemble_" + type + ".csv",
#                           index_col='Date')
#         df2 = pd.read_csv("./Output/ensemble/ensembleFolder/walk" + "Day" + str(j) + "ensemble_" + type + ".csv",
#                           index_col='Date')
#         df3 = pd.read_csv("./Output/ensemble/ensembleFolder/walk" + "Week" + str(j) + "ensemble_" + type + ".csv",
#                           index_col='Date')
        
#         from_date=str(df2.index[0])
#         to_date=str(df2.index[len(df2)-1])

#         for deleted in range(1, numDel):
#             del df1['iteration' + str(deleted)]
#             del df2['iteration' + str(deleted)]
#             del df3['iteration' + str(deleted)]

#         if perc == 0:
#             df1 = full_ensemble(df1)
#         else:
#             df1 = perc_ensemble(df1, perc)

#         df2.index = pd.to_datetime(df2.index)
#         df2.index = df2.index.strftime('%m/%d/%Y')
#         df2.rename(columns={'trend': 'ensemble'}, inplace=True)

#         df3.index = pd.to_datetime(df3.index)
#         df3.index = df3.index.strftime('%m/%d/%Y')
#         df3.rename(columns={'trend': 'ensemble'}, inplace=True)

#         df3_temp = pd.DataFrame(index=df2.index).assign(ensemble=0)

#         for k in range(0,len(df3_temp)):
#             df3_temp['ensemble'][k] = getActionWeek(df3,df3_temp.index[k])
        
#         y_true = ensemble_y_true(df1, df2, df3)
        
#         # Tạo lưới trọng số
#         # Tạo ra tất cả các tổ hợp có thể của trọng số với các giá trị từ 0 đến 1 
#         # (với bước nhảy 0.1), với điều kiện tổng trọng số bằng 1
#         weights = list(product(np.arange(0, 1.1, 0.1), repeat=3))
#         weights = [w for w in weights if np.sum(w) == 1]

#         # Tìm trọng số tốt nhất
#         best_score = 0
#         best_weights = None

#         for w in weights:
#             combined_pred = weighted_average(np.array([df1['ensemble'].to_list(), df2['ensemble'].to_list(), df3_temp['ensemble'].to_list()]), w)
#             score = accuracy_score(y_true, np.round(combined_pred))
#             if score > best_score:
#                 best_score = score
#                 best_weights = w

#         weight_1 = best_weights[0]
#         weight_2 = best_weights[1]
#         weight_3 = best_weights[2]
        
#         for deleted in range(1,numDel):
#             del df['iteration'+str(deleted)]

#         df = pd.DataFrame(columns=['ensemble'])
#         df = df.set_index(pd.Index([], name='date'))

#         for k in range(0,len(df1)):
#             if(df1.index[k] in df2.index):
#                 df.loc[df1.index[k]] = bin_predictions((df1['ensemble'][k] * weight_1 + df2.loc[df1.index[k],'ensemble'] * weight_2 + getActionWeek(df3, df2.index[k]) * weight_3))

#         num=0
#         rew=0
#         pos=0
#         neg=0
#         doll=0
#         cov=0
#         for date, i in df.iterrows():
#             num+=1

#             if date in dax.index:
#                 if (i['ensemble']==1):
#                     pos+= 1 if (dax.at[date,'Close']-dax.at[date,'Open'])/dax.at[date,'Open'] > 0 else 0

#                     neg+= 0 if (dax.at[date,'Close']-dax.at[date,'Open'])/dax.at[date,'Open'] > 0 else 1
#                     rew+=(dax.at[date,'Close']-dax.at[date,'Open'])/dax.at[date,'Open']
#                     doll+=(dax.at[date,'Close']-dax.at[date,'Open'])*50
#                     cov+=1
#                 elif (i['ensemble']==2):

#                     neg+= 0 if -(dax.at[date,'Close']-dax.at[date,'Open'])/dax.at[date,'Open'] > 0 else 1
#                     pos+= 1 if -(dax.at[date,'Close']-dax.at[date,'Open'])/dax.at[date,'Open'] > 0 else 0
#                     rew+=-(dax.at[date,'Close']-dax.at[date,'Open'])/dax.at[date,'Open']
#                     cov+=1
#                     doll+=-(dax.at[date,'Close']-dax.at[date,'Open'])*50

#         values.append([from_date, to_date,str(round(rew,2)),str(round(pos,2)),str(round(neg,2)),str(round(doll,2)),str(round(cov/num,2)),(str(round(pos/cov,2)) if (cov>0) else "None")])

#         dollSum+=doll
#         rewSum+=rew
#         posSum+=pos
#         negSum+=neg
#         covSum+=cov
#         numSum+=num


#     values.append([' ','Sum',str(round(rewSum,2)),str(round(posSum,2)),str(round(negSum,2)),str(round(dollSum,2)),str(round(covSum/numSum,2)),(str(round(posSum/covSum,2)) if (covSum>0) else "None")])
#     # print(values)
#     return values,columns