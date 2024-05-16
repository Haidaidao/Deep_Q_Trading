import numpy as np
from scipy import stats
import pandas as pd
import global_config
import matplotlib.pyplot as plt
import json

# config = json.load(open('plotResultsConf.json', 'r'))

class Evaluation:
    def __init__(self, data, name):
        self.name = name
        self.data = data
        first_index = data.index[0]
        # Lấy giá trị index cuối
        last_index = data.index[-1]
        self.df = pd.read_csv("./datasets/" + name + "Hour.csv", index_col='Date')[first_index:last_index]
        grouped_open = self.df.groupby(self.df.index).apply(lambda x: np.array(x[['Open']]).flatten())
        self.open_data = {pd.Timestamp(date): values for date, values in grouped_open.items()}
        grouped_close = self.df.groupby(self.df.index).apply(lambda x: np.array(x[['Close']]).flatten())
        self.close_data = {pd.Timestamp(date): values for date, values in grouped_close.items()}
        self.ensemble_data = {pd.Timestamp(date): str(ensemble) for date, ensemble in data['ensemble'].items()}

        data['date'] = data.index  
        self.date_list = data['date'].tolist()

        # print(self.open_data)

    # def evaluate(self):
    #     wins = 0
    #     loses = 0
    #     profit = 0

    #     for i in range(len(self.data)-1):
    #         signal = self.data.iloc[i]['ensemble']
    #         entry_price = self.data.iloc[i]['close'] 
    #         next_price_low = self.data.iloc[i+1]['low'] 
    #         next_price_high = self.data.iloc[i+1]['high'] 

    #         if signal == 1:
    #             if next_price_high >= entry_price * 1.01:  # Takeprofit
    #                 wins += 1
    #                 profit += entry_price * 0.1
    #             elif next_price_low <= entry_price * 0.995:  # Stoploss
    #                 loses += 1
    #                 profit -= entry_price * 0.05
            
    #         elif signal == 2:
    #             if next_price_low <= entry_price * 0.99:  # Takeprofit
    #                 wins += 1
    #                 profit += entry_price * 0.1
    #             elif next_price_high >= entry_price * 1.005:  # Stoploss
    #                 loses += 1
    #                 profit -= entry_price * 0.05
    #     return wins, loses, profit

    def evaluate(self,stop_loss_pct=0.02, take_profit_pct=0.04, capital=10000):
        win_trades = 0
        lose_trades = 0
        close_data = self.close_data
        for date, prices in self.close_data.items():
            signal = self.ensemble_data[date]
            if signal == 0:
                continue  # Không giao dịch nếu tín hiệu là SIDEWAY

            entry_price = prices[0]  # Giá mở cửa của ngày
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
                profit_or_loss = (prices[-1] - entry_price) * (1 if signal == 1 else -1)
                capital += profit_or_loss
                if profit_or_loss > 0:
                    win_trades += 1
                else:
                    lose_trades += 1
            
            # print(f"Ngày: {date}, Tín hiệu: {signal}, Vốn hiện tại: {capital:.2f}")
        
        return capital, win_trades, lose_trades

    