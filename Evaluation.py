import numpy as np
from scipy import stats
import pandas as pd
import global_config
import matplotlib.pyplot as plt
import json

config = json.load(open('plotResultsConf.json', 'r'))

class Evaluation:
    def __init__(self, data):
        self.data = data

    # def evaluate(self):
    #     wins = 0
    #     loses = 0
    #     profit = 0

    #     for i in range(len(self.data)-1):
    #         signal = self.data.iloc[i]['ensemble']
    #         entry_price = self.data.iloc[i]['close'] 
    #         entry_price_open = self.data.iloc[i]['open'] 
    #         next_price_low = self.data.iloc[i+1]['low'] 
    #         next_price_high = self.data.iloc[i+1]['high'] 

    #         if signal == 1:
    #             if next_price_high >= entry_price * 1.01:  # Takeprofit
    #                 wins += 1
    #                 profit += entry_price - entry_price_open
    #             elif next_price_low <= entry_price * 0.995:  # Stoploss
    #                 loses += 1
    #                 profit -= entry_price - entry_price_open
            
    #         elif signal == 2 or signal == -1:
    #             if next_price_low <= entry_price * 0.99:  # Takeprofit
    #                 wins += 1
    #                 profit += entry_price - entry_price_open
    #             elif next_price_high >= entry_price * 1.005:  # Stoploss
    #                 loses += 1
    #                 profit -= entry_price - entry_price_open
    #     return wins, loses, profit
        

    def evaluate(self):
        wins = 0
        loses = 0
        profit = 0

        for i in range(len(self.data)-1):
            signal = self.data.iloc[i]['ensemble']
            entry_price = self.data.iloc[i]['close'] 
            entry_price_open = self.data.iloc[i]['open'] 
            next_price_low = self.data.iloc[i+1]['low'] 
            next_price_high = self.data.iloc[i+1]['high'] 

            if signal == 1:
                if next_price_high >= entry_price * 1.01:  # Takeprofit
                    wins += 1
                    profit += entry_price*0.1
                elif next_price_low <= entry_price * 0.995:  # Stoploss
                    loses += 1
                    profit -= entry_price*0.05
            
            elif signal == 2 or signal == -1:
                if next_price_low <= entry_price * 0.99:  # Takeprofit
                    wins += 1
                    profit += entry_price*0.1
                elif next_price_high >= entry_price * 1.005:  # Stoploss
                    loses += 1
                    profit -= entry_price*0.05
        return wins, loses, profit


    