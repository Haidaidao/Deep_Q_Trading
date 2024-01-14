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
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.utils import to_categorical

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

    def create_sequences(self, data, sequence_length):
        sequences = []
        labels = []
        for i in range(len(data)):
            if i < sequence_length - 1:
                # Đối với ngày đầu tiên, sử dụng dữ liệu từ chính ngày đó
                sequences.append(data.iloc[i:i+1, :-1].values.repeat(sequence_length, axis=0))
            else:
                sequences.append(data.iloc[i-sequence_length+1:i+1, :-1].values)
            labels.append(data.iloc[i, -1])
        return np.array(sequences), np.array(labels)
    
    def writeFile(self):
        df = self.spTimeserie
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)
        df['Close_change'] = df['Close'].pct_change() * 100
        df['Trend'] = df.apply(lambda row: 1 if row['Close_change'] > 0.5 else 2 if row['Close_change'] < -0.5 else 0, axis=1)
        df.dropna(inplace=True)

        scaler = MinMaxScaler()
        scaled_columns = scaler.fit_transform(df[['Open', 'High', 'Low', 'Close']])
        scaled_df = pd.DataFrame(scaled_columns, index=df.index, columns=['Scaled_Open', 'Scaled_High', 'Scaled_Low', 'Scaled_Close'])
        scaled_df['Trend'] = df['Trend'].values

        sequence_length = 1
        X, y = self.create_sequences(scaled_df, sequence_length)
        y_onehot = to_categorical(y, num_classes=3)

        # Xây dựng mô hình LSTM
        model = Sequential([
            LSTM(50, activation='relu', input_shape=(X.shape[1], X.shape[2])),
            Dropout(0.2),
            Dense(3, activation='softmax')
        ])

        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        # Huấn luyện mô hình
        model.fit(X, y_onehot, epochs=100)

        # Dự đoán xu hướng cho mỗi ngày
        predicted_trends = model.predict(X)
        predicted_trends = np.argmax(predicted_trends, axis=1)
        predicted_trends = np.insert(predicted_trends, 0, 1) 

        X_hmm = predicted_trends.reshape(-1, 1)

        # Khởi tạo mô hình GMMHMM
        n_components = 3  # Số lượng trạng thái ẩn (có thể điều chỉnh tùy theo dữ liệu)
        param=set(predicted_trends.ravel())
        model_hmm=hmm.GaussianHMM(n_components=n_components, covariance_type="full", n_iter=100,params="s")

        # Huấn luyện mô hình GMMHMM
        model_hmm.fit(X_hmm)
        if np.isnan(X_hmm).any() or np.isinf(X_hmm).any():
            print("Dữ liệu đầu vào chứa NaN hoặc inf.")
        # Dự đoán trạng thái ẩn
        hidden_states = model_hmm.predict(X_hmm)

        ensambleValid=pd.DataFrame(index=self.Date)
        ensambleValid.index.name='Date'
        ensambleValid['trend'] = 0
        for i in range(len(ensambleValid)):
            ensambleValid['trend'].iat[i] = hidden_states[i]

        ensambleValid.to_csv("./Output/ensemble/"+"ensembleFolder"+"/walk"+self.name+str(self.iteration)+"ensemble_"+self.type+".csv")
