# #Copyright (C) 2020 Salvatore Carta, Anselmo Ferreira, Alessandro Sebastian Podda, Diego Reforgiato Recupero, Antonio Sanna. All rights reserved.

import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.metrics import accuracy_score
from itertools import product
from Evaluation import Evaluation
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

def Evaluate(data, name, stop_loss_pct=0.02, take_profit_pct=0.04, capital=10000):
        first_index = data.index[0]
        last_index = data.index[-1]

        ensemble_data = {pd.Timestamp(date): str(ensemble) for date, ensemble in data['ensemble'].items()}
        df = pd.read_csv("./datasets/" + name + "Hour.csv", index_col='Date')[first_index:last_index]
       
        grouped_open = df.groupby(df.index).apply(lambda x: np.array(x[['Open']]).flatten())
        open_data = {pd.Timestamp(date): values for date, values in grouped_open.items()}
   
        grouped_close = df.groupby(df.index).apply(lambda x: np.array(x[['Close']]).flatten())
        close_data = {pd.Timestamp(date): values for date, values in grouped_close.items()} 

        win_trades = 0
        lose_trades = 0
        for date, prices in close_data.items():

            signal = ensemble_data[date]
            if signal == 0:
                continue  # Không giao dịch nếu tín hiệu là SIDEWAY

            entry_price = open_data[date][0]  # Giá mở cửa của ngày
            stop_loss_price = entry_price * (1 - stop_loss_pct) if signal == 1 else entry_price * (1 + stop_loss_pct)
            take_profit_price = entry_price * (1 + take_profit_pct) if signal == 1 else entry_price * (1 - take_profit_pct)
            
            for price in prices:
                if (signal == 1 and (price <= stop_loss_price or price >= take_profit_price)) or \
                (signal == -1 and (price >= stop_loss_price or price <= take_profit_price)):
                    # Xuất hiện điều kiện StopLoss hoặc Take Profit
                    profit_or_loss = (price - entry_price) * (1 if signal == 1 else -1)
                    capital += profit_or_loss
                    if profit_or_loss > 0:
                        win_trades += 1
                    else:
                        lose_trades += 1
                    break
            else:
                # Nếu không có StopLoss hoặc Take Profit, tính toán lãi/lỗ tại giá đóng cửa
                profit_or_loss = (close_data[date][-1] - entry_price) * (1 if signal == 1 else -1)
                capital += profit_or_loss
                if profit_or_loss > 0:
                    win_trades += 1
                else:
                    lose_trades += 1
           
            # print(f"Ngày: {date}, Tín hiệu: {signal}, Vốn hiện tại: {capital:.2f}")
        
        return capital, win_trades, lose_trades


# ================================================ XGBoots
def XGBoostEnsemble(numWalks,perc,type,numDel):
    
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

    xgb_model = xgb.XGBClassifier(n_estimators=100, random_state=42)

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

        print(df1)
        print("===========================")
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
        list_combine_train[list_combine_train == -1] = 2

        y_train = ensemble_y_true(df1, dax, threshold)
        y_train[y_train == -1] = 2

        le = LabelEncoder()
        y_train = le.fit_transform(y_train)
        xgb_model.fit(list_combine_train, y_train)

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
                if df1_result['ensemble'][k] == -1:
                    df1_result['ensemble'][k] = 2
                if df2_temp['ensemble'][k] == -1:
                    df2_temp['ensemble'][k] = 2
                if df3_temp['ensemble'][k] == -1:
                    df3_temp['ensemble'][k] = 2
                new_data = np.array([[df1_result['ensemble'][k], df2_temp['ensemble'][k], df3_temp['ensemble'][k]]])
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

        values.append([from_date, to_date,str(round(rew,2)),str(round(pos,2)),str(round(neg,2)),"",str(round(doll,2)),str(round(cov/num,2)),(str(round(pos/cov,2)) if (cov>0) else "None")])
        dollSum+=doll
        rewSum+=rew
        posSum+=pos
        negSum+=neg
        covSum+=cov
        numSum+=num


    values.append([' ','Sum',str(round(rewSum,2)),str(round(posSum,2)),str(round(negSum,2)),str(round(posSum/negSum,2)),str(round(dollSum,2)),str(round(covSum/numSum,2)),(str(round(posSum/covSum,2)) if (covSum>0) else "None")])

    return values,columns
# ================================================ Random Forest
def RandomForestEnsemble(numWalks,perc,type,numDel):
    Capital = 10000 
    Capital_original = Capital

    Wins = 0
    Losses = 0

    columns = ["From","To", "Capital", "#Wins", "#Losses", "Ratio", "Difference"]


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

        rf_model.fit(list_combine_train, y_train)

        # Predict
        for deleted in range(1,numDel):
            del df['iteration'+str(deleted)]

        df = pd.DataFrame(columns=['ensemble'])
        df = df.set_index(pd.Index([], name='date'))

        df1_result = pd.read_csv(f"./Output/ensemble/{ensembleFolder}/walk" + "Hour" + str(j) + "ensemble_" + type + ".csv",
                          index_col='Date')

        from_date=str(df2.index[0])
        to_date=str(df2.index[len(df2)-1])

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

        df['open'] = df.index.map(dax['Open'])
        df['high'] = df.index.map(dax['High'])
        df['low'] = df.index.map(dax['Low'])
        df['close'] = df.index.map(dax['Close'])
        print(Capital)
        print("==================")
        Capital, win_trades, lose_trades =Evaluation(df, MK).evaluate(capital=Capital)
        values.append([from_date, to_date,str(round(Capital,2)),str(round(win_trades,2)),str(round(lose_trades,2)), "", str(round(Capital-Capital_original,2))])
        Capital_final = Capital
        Wins+=win_trades
        Losses+=lose_trades

    values.append([' ', "Finall",str(round(Capital_final,2)),str(round(Wins,2)),str(round(Losses,2)), str(round(Wins/Losses,2) if (Losses>0) else "None"), " "])

    return values,columns

# ================================================ Base Rule


def BaseRule(numWalks,perc,type,numDel):

    Capital = 10000 
    Capital_original = Capital

    Wins = 0
    Losses = 0

    columns = ["From","To", "Capital", "#Wins", "#Losses", "Ratio", "Difference"]

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
        

        df['open'] = df.index.map(dax['Open'])
        df['high'] = df.index.map(dax['High'])
        df['low'] = df.index.map(dax['Low'])
        df['close'] = df.index.map(dax['Close'])
       
        # eva = Evaluation()
        Capital, win_trades, lose_trades = Evaluate(data=df,name=MK,capital=Capital)

        values.append([from_date, to_date,str(round(Capital,2)),str(round(win_trades,2)),str(round(lose_trades,2)), "", str(round(Capital-Capital_original,2))])
        Capital_final = Capital
        Wins+=win_trades
        Losses+=lose_trades

    values.append([' ', "Finall",str(round(Capital_final,2)),str(round(Wins,2)),str(round(Losses,2)), str(round(Wins/Losses,2) if (Losses>0) else "None"), " "])

    return values,columns


# ================================================ Rule author

def EnsembleAuthor(numWalks,perc,type,numDel):
    Capital = 10000 
    Capital_original = Capital

    Wins = 0
    Losses = 0
    values=[]
    columns = ["From","To", "Capital", "#Wins", "#Losses", "Ratio", "Difference"]
    dax = pd.read_csv("./datasets/" + global_config.MK + "Day.csv", index_col='Date')
    for j in range(0,numWalks):


        df=pd.read_csv(f"./Output/ensemble/walk"+"Hour"+str(j)+"ensemble_"+type+".csv",index_col='Date')

        from_date=str(df.index[0])
        to_date=str(df.index[len(df)-1])

        for deleted in range(1,numDel):
            del df['iteration'+str(deleted)]
        
        if perc==0:
            df=full_ensemble(df)
        else:
            df=perc_ensemble(df,perc)

        df['open'] = df.index.map(dax['Open'])
        df['high'] = df.index.map(dax['High'])
        df['low'] = df.index.map(dax['Low'])
        df['close'] = df.index.map(dax['Close'])
        eva = Evaluation()
        Capital, win_trades, lose_trades = eva.evaluate(data=df,name=MK,capital=Capital)
        
        values.append([from_date, to_date,str(round(Capital,2)),str(round(win_trades,2)),str(round(lose_trades,2)), "", str(round(Capital-Capital_original,2))])
        Capital_final = Capital
        Wins+=win_trades
        Losses+=lose_trades

    values.append([' ', "Finall",str(round(Capital_final,2)),str(round(Wins,2)),str(round(Losses,2)), str(round(Wins/Losses,2) if (Losses>0) else "None"), " "])

    return values,columns

