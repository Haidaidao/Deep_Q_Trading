import numpy as np
from scipy import stats
import pandas as pd
import global_config
import matplotlib.pyplot as plt
import json

# config = json.load(open('plotResultsConf.json', 'r'))

class Evaluation:
    # def __init__(self):
    #     data.to_csv('outputDF_'+ type+str(j)+str(perc)+'_.csv', index=False)

    #     self.name = name
    #     self.data = data
    #     first_index = data.index[0]
    #     # Lấy giá trị index cuối
    #     last_index = data.index[-1]
      
        # self.df = pd.read_csv("./datasets/" + name + "Hour.csv", index_col='Date')[first_index:last_index]
       
        # grouped_open = self.df.groupby(self.df.index).apply(lambda x: np.array(x[['Open']]).flatten())
        # self.open_data = {pd.Timestamp(date): values for date, values in grouped_open.items()}
   
        # grouped_close = self.df.groupby(self.df.index).apply(lambda x: np.array(x[['Close']]).flatten())
        # self.close_data = {pd.Timestamp(date): values for date, values in grouped_close.items()}
       
        # self.ensemble_data = {pd.Timestamp(date): str(ensemble) for date, ensemble in data['ensemble'].items()}
  
        # data['date'] = data.index  
        # self.date_list = data['date'].tolist()

    def evaluate(self, data, name, stop_loss_pct=0.02, take_profit_pct=0.04, capital=10000):
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

    