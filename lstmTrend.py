import numpy as np
import pandas as pd
#datetime is the library used to manipulate time and date
from datetime import datetime

from decimal import Decimal

import string
import yfinance as yf
from hmmlearn import hmm
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

import global_config

MK = global_config.MK

class LSTMTrend:
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

    def create_trend_labels(self, close_prices):
        
        trends = [0]
        for i in range(1, len(close_prices)):
            if close_prices[i] > close_prices[i - 1]: trends.append(1)  # UP
            elif close_prices[i] < close_prices[i - 1]: trends.append(2)  # DOWN
            else: trends.append(0)  # HOLD
        return np.array(trends)

    def writeFile(self):
        # Preprocess the data
        df = self.spTimeserie[['Close']]
        print(df)
        print("====================")
        data = df[['Close']].values.reshape(-1, 1)
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(data)
        print(data_scaled)
        # Train GMM-HMM
        data_scaled = np.nan_to_num(data_scaled, nan=np.nan, posinf=None, neginf=None)
        # Kiểm tra và loại bỏ hàng chứa NaN
        data_scaled = data_scaled[~np.isnan(data_scaled).any(axis=1)]
        num_states = 4
        num_components = 2
        gmm_hmm = hmm.GMMHMM(n_components=num_states, n_mix=num_components, covariance_type="full", random_state=42)
        gmm_hmm.fit(data)

        # Use GMM-HMM probabilities to construct dataset
        market_trends = self.create_trend_labels(self.Close)
        print(market_trends)
        print("---------------")
        gmm_hmm_probs = gmm_hmm.predict_proba(data_scaled)
        X = np.array([gmm_hmm_probs[i].flatten() for i in range(len(gmm_hmm_probs))])
        y = market_trends

        # Split dataset into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train LSTM model
        model = Sequential()
        model.add(LSTM(units=50, activation='relu', input_shape=(X_train.shape[1], 1)))
        model.add(Dense(units=3, activation='softmax'))  # 3 units for UP, DOWN, HOLD
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

        # Reshape input for LSTM
        X_train_lstm = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
        X_test_lstm = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

        # Fit the LSTM model
        model.fit(X_train_lstm, y_train, epochs=10, batch_size=32, validation_data=(X_test_lstm, y_test))
        
        # Evaluate the model
        y_pred = np.argmax(model.predict(X_test_lstm), axis=1)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Accuracy: {accuracy}")

        # Predict market trends for entire dataset
        X_all = X.reshape((X.shape[0], X.shape[1], 1))
        y_pred_all = np.argmax(model.predict(X_all), axis=1)
        df[self.columnName] = y_pred_all

        ensambleValid=pd.DataFrame(index=self.Date)
        ensambleValid.index.name='Date'
        ensambleValid['trend'] = 0
        for i in range(len(ensambleValid)):
            ensambleValid['trend'].iat[i] = df['trend'].iat[i]

        ensambleValid.to_csv("./Output/ensemble/"+"ensembleFolder"+"/walk"+self.name+str(self.iteration)+"ensemble_"+self.type+".csv")