# #Copyright (C) 2020 Salvatore Carta, Anselmo Ferreira, Alessandro Sebastian Podda, Diego Reforgiato Recupero, Antonio Sanna. All rights reserved.

import csv
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
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
import global_config
import json
from Evaluation import Evaluation
import re

config = json.load(open('plotResultsConf.json', 'r'))

ensembleFolder = global_config.ensembleFolder
threshold = global_config.label_threshold

iteration = 'iteration' + str(config['epoch']-1)


# Get the results in the log file of the week that contains that day.
# def getAction(Frame, date):
    
#     date = datetime.strptime(date,"%m/%d/%Y")

#     for i in range(0, len(Frame)):
#         date_point =  datetime.strptime(Frame.index[i],"%m/%d/%Y")
#         if date_point>date:
#             print(date)
#             print(datetime.strptime(Frame.index[i-1],"%m/%d/%Y"))
#             print(Frame['ensemble'][i-1])
#             print("=======================")
#             return  Frame['ensemble'][i-1]
    
#     return 0

def getAction(Frame, date):
    # # Chuyển đổi chuỗi ngày thành đối tượng datetime
    # target_date = datetime.strptime(date, "%m/%d/%Y")
    # # print(target_date)
    # # Chắc chắn rằng chỉ mục của Frame là đối tượng datetime
    # Frame.index = pd.to_datetime(Frame.index)

    # Tìm chỉ số của ngày gần nhất không lớn hơn target_date
    target_date = date.replace(hour=0, minute=0, second=0) 
    index = None
    for i in range(len(Frame)):
        date_point = Frame.index[i]
        if date_point > target_date:
            break
        index = i

    if index is not None:
        # In thông tin và trả về giá trị ensemble
        prev_date = Frame.index[index]
        ensemble_value = Frame['ensemble'][index]
        # print(prev_date)
        # print(ensemble_value)
        # print("=======================")
        return ensemble_value
    else:
        print("Không tìm thấy ngày phù hợp trong Frame")
        return 0

def ensemble_y_true(feature, stats, threshold):

    labels = []
    last_action = 0
    skip = 0
    
    for index, _ in feature.iterrows():
        # date = re.search(r'\d{2}/\d{2}/\d{4}', index).group()
        date = index
        if date not in stats.index:
            labels.append(last_action)
            skip += 1
            continue
        
        close = stats.loc[date, 'Close']
        open = stats.loc[date, 'Open']
        
        action = 0
        changes = (close - open) / open

        if changes >= threshold or (last_action > 0 and changes >= 0 and changes < threshold):
            last_action = 1
            action = 1
        elif changes < -threshold or (last_action < 0 and changes < 0 and changes >= -threshold):
            last_action = -1
            action = -1
        else:
            last_action = 0

        labels.append(action)
            
    return labels




# ================================================ XGBoots
def XGBoostEnsemble(numWalks,type,numDel):
    
    wins_number = 0
    loses_number = 0
    profit_number = 0

    columns = ["From","To", "Wins", "Closes", "Profit"]

    values = []

    dax=pd.read_csv("./datasets/"+ global_config.MK +"Day.csv",index_col='Date')
   
    daxHour = pd.read_csv("./datasets/"+ global_config.MK +"Hour.csv", index_col='Date')
    daxHour = daxHour.reset_index()
    daxHour['Date'] = pd.to_datetime(daxHour['Date'] + ' ' + daxHour['Time'])
    daxHour['Date'] = daxHour['Date'].dt.strftime('%m/%d/%Y %H:%M')
    daxHour.set_index('Date', inplace=True)


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

        df2_new = df1.copy()
        df3_new = df1.copy()

        for i in range(len(df2_new)):
            date = re.search(r'\d{2}/\d{2}/\d{4}', df2_new.index[i]).group()
            df2_new['ensemble'][i] = getAction(df2,date)

        df3.index = pd.to_datetime(df3.index)
        df3.index = df3.index.strftime('%m/%d/%Y')
        df3.rename(columns={'trend': 'ensemble'}, inplace=True)

        for i in range(len(df3_new)):            
            date = re.search(r'\d{2}/\d{2}/\d{4}', df3_new.index[i]).group()
            df3_new['ensemble'][i] = getAction(df3,date)


        list_combine_train = np.empty((0, 3))

        for k in range(0,len(df1)):
            list_combine_train = np.append(list_combine_train, [[df1['ensemble'][k], df2_new['ensemble'][k], df3_new['ensemble'][k]]], axis=0)
      
        y_train = ensemble_y_true(df1, daxHour, threshold)
        # with open('y_train.csv', 'w', newline='') as file:
        #     writer = csv.writer(file)
        #     for value in y_train:
        #         writer.writerow([value])  # Ghi từng số trong list vào một hàng
        # # print(y_train)
        # # print("=====================")
        for i in range(len(y_train)):
            if y_train[i] == -1:
                y_train[i] = 2
        
        for i in range(len(list_combine_train)):
            for k in range(len(list_combine_train[i])):
                if list_combine_train[i][k]==-1:
                    list_combine_train[i][k]=2
       

        le = LabelEncoder()
        y_train = le.fit_transform(y_train)
        xgb_model.fit(list_combine_train, y_train)

        # # Predict
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

        from_date=str(df1_result.index[0])
        to_date=str(df1_result.index[len(df1_result)-1])

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

        df2_new_result = df1.copy()
        df3_new_result = df1.copy()

        for i in range(len(df2_new_result)):
            date = re.search(r'\d{2}/\d{2}/\d{4}', df2_new_result.index[i]).group()
            df2_new_result['ensemble'][i] = getAction(df2_result,date)

        for i in range(len(df3_new)):            
            date = re.search(r'\d{2}/\d{2}/\d{4}', df3_new_result.index[i]).group()
            df3_new_result['ensemble'][i] = getAction(df3_result,date)

        for k in range(0,len(df1_result)):
            date = re.search(r'\d{2}/\d{2}/\d{4}', df1_result.index[k]).group()
            if(date in df2_result.index):
                new_data = np.array([[df1_result['ensemble'][k], df2_new_result['ensemble'][k], df3_new_result['ensemble'][k]]])  
                for m in range(len(new_data)):
                    for n in range(len(new_data[m])):
                        if new_data[m][n]==-1:
                            new_data[m][n]=2
                predicted_result = xgb_model.predict(new_data)
                df.loc[df1_result.index[k]] = predicted_result[0]
        
        df['close'] = df.index.map(daxHour['Close'])
        df['open'] = df.index.map(daxHour['Open'])
        df['high'] = df.index.map(daxHour['High'])
        df['low'] = df.index.map(daxHour['Low'])
        # df.to_csv('df.csv', index=False)

        eva = Evaluation(df)
        wins, loses, profit = eva.evaluate()
        values.append([from_date, to_date,str(round(wins,2)),str(round(loses,2)),str(round(profit,2))])

        wins_number+=wins
        loses_number+=loses
        profit_number+=profit

    values.append([' ','Sum',str(round(wins_number,2)),str(round(loses_number,2)),str(round(profit_number,2))])
    
    # values.append([' ','Sum',0,0,0])
    return values,columns
# ================================================ Random Forest
def RandomForestEnsemble(numWalks,type,numDel):
    
    wins_number = 0
    loses_number = 0
    profit_number = 0

    columns = ["From","To", "Wins", "Closes", "Profit"]

    values = []

    dax=pd.read_csv("./datasets/"+ global_config.MK +"Day.csv",index_col='Date')

    daxHour = pd.read_csv("./datasets/"+ global_config.MK +"Hour.csv", index_col='Date')
    daxHour = daxHour.reset_index()
    daxHour['Date'] = pd.to_datetime(daxHour['Date'] + ' ' + daxHour['Time'])
    daxHour['Date'] = daxHour['Date'].dt.strftime('%m/%d/%Y %H:%M')
    daxHour.set_index('Date', inplace=True)

    type_train = "train"

    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)

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
        df1.index = pd.to_datetime(df1.index, format='%m/%d/%Y %H:%M')
        df1.rename(columns={iteration: 'ensemble'}, inplace=True)

        df2.index = pd.to_datetime(df2.index, format='%m/%d/%Y %H:%M')
        # df2.index = df2.index.strftime('%m/%d/%Y')
        df2.rename(columns={'trend': 'ensemble'}, inplace=True)

        df2_new = df1.copy()
        df3_new = df1.copy()

        for i in range(len(df2_new)):
            # date = re.search(r'\d{2}/\d{2}/\d{4}', df2_new.index[i]).group()
            date = df2_new.index[i]
            df2_new['ensemble'][i] = getAction(df2,date)



        df3.index = pd.to_datetime(df3.index, format='%m/%d/%Y %H:%M')
        # df3.index = df3.index.strftime('%m/%d/%Y')
        df3.rename(columns={'trend': 'ensemble'}, inplace=True)

        for i in range(len(df3_new)):            
            # date = re.search(r'\d{2}/\d{2}/\d{4}', df3_new.index[i]).group()
            date = df3_new.index[i]
            df3_new['ensemble'][i] = getAction(df3,date)


        list_combine_train = np.empty((0, 3))

        for k in range(0,len(df1)):
            list_combine_train = np.append(list_combine_train, [[df1['ensemble'][k], df2_new['ensemble'][k], df3_new['ensemble'][k]]], axis=0)
      
        y_train = ensemble_y_true(df1, daxHour, threshold)
     

        # rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
        rf_model.fit(list_combine_train, y_train)

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

        from_date=str(df1_result.index[0])
        to_date=str(df1_result.index[len(df1_result)-1])

        for deleted in range(1, numDel):
            del df1_result['iteration' + str(deleted)]
            del df2_result['iteration' + str(deleted)]
            del df3_result['iteration' + str(deleted)]

        df1_result = pd.DataFrame(df1_result[iteration])
        df1_result.index = pd.to_datetime(df1_result.index, format='%m/%d/%Y %H:%M')
        df1_result.rename(columns={iteration: 'ensemble'}, inplace=True)

        df2_result.index = pd.to_datetime(df2_result.index, format='%m/%d/%Y %H:%M')
        # df2_result.index = df2_result.index.strftime('%m/%d/%Y')
        df2_result.rename(columns={'trend': 'ensemble'}, inplace=True)

        df3_result.index = pd.to_datetime(df3_result.index, format='%m/%d/%Y %H:%M')
        # df3_result.index = df3_result.index.strftime('%m/%d/%Y')
        df3_result.rename(columns={'trend': 'ensemble'}, inplace=True)

        df2_new_result = df1.copy()
        df3_new_result = df1.copy()

        for i in range(len(df2_new_result)):
            # date = re.search(r'\d{2}/\d{2}/\d{4}', df2_new_result.index[i]).group()
            date = df2_new_result.index[i]
            df2_new_result['ensemble'][i] = getAction(df2_result,date)


        for i in range(len(df3_new)):            
            # date = re.search(r'\d{2}/\d{2}/\d{4}', df3_new_result.index[i]).group()
            date = df3_new_result.index[i]
            df3_new_result['ensemble'][i] = getAction(df3_result,date)

        for k in range(0,len(df1_result)):
            # date = re.search(r'\d{2}/\d{2}/\d{4}', df1_result.index[k]).group()
            date = df1_result.index[k]
            if(date in df2_result.index):
                new_data = np.array([[df1_result['ensemble'][k], df2_new_result['ensemble'][k], df3_new_result['ensemble'][k]]])
                predicted_result = rf_model.predict(new_data)
                df.loc[df1_result.index[k]] = predicted_result[0]
 
        df['close'] = df.index.map(daxHour['Close'])
        df['open'] = df.index.map(daxHour['Open'])
        df['high'] = df.index.map(daxHour['High'])
        df['low'] = df.index.map(daxHour['Low'])
        # print(df)
        eva = Evaluation(df)
        wins, loses, profit = eva.evaluate()
        values.append([from_date, to_date,str(round(wins,2)),str(round(loses,2)),str(round(profit,2))])

        wins_number+=wins
        loses_number+=loses
        profit_number+=profit

    values.append([' ','Sum',str(round(wins_number,2)),str(round(loses_number,2)),str(round(profit_number,2))])
    
    # values.append([' ','Sum',0,0,0])
    return values,columns
# ================================================ Base-rule

def normalized(a):
    if a > 0:
        return 1
    elif a < 0:
        return -1
    return 0

def SimpleEnsemble(numWalks,type,numDel):
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

        df1=pd.read_csv(f"./Output/ensemble/{ensembleFolder}/walk"+"Hour"+str(j)+"ensemble_"+type+".csv",index_col='Date')
        df2=pd.read_csv(f"./Output/ensemble/{ensembleFolder}/walk"+"Day"+str(j)+"ensemble_"+type+".csv",index_col='Date')
        df3=pd.read_csv(f"./Output/ensemble/{ensembleFolder}/walk"+"Week"+str(j)+"ensemble_"+type+".csv",index_col='Date')

        from_date=str(df2.index[0])
        to_date=str(df2.index[len(df2)-1])

        for deleted in range(1,numDel):
            del df1['iteration'+str(deleted)]
            del df2['iteration'+str(deleted)]
            del df3['iteration'+str(deleted)]

            
        df1 = pd.DataFrame(df1[iteration])
        df1.rename(columns={iteration: 'ensemble'}, inplace=True)

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

        for k in range(0,len(df1)):
            if(df1.index[k] in df2.index):
                if df1['ensemble'][k] == 0:
                    df.loc[df1.index[k]] = 0
                else:
                    Sh = normalized(df1['ensemble'][k])
                    Sw = normalized(getAction(df3, df2.index[k]))
                    Sd = normalized(df2.loc[df1.index[k],'ensemble'])
                    
                    if Sh == 1:
                        if Sw + Sd >= 1:
                            df.loc[df1.index[k]] = 1
                    elif Sh == -1: 
                        if Sw + Sd <= -1:
                            df.loc[df1.index[k]] = -1
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

