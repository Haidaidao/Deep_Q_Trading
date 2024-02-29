# Copyright 2019-2020 Alvaro Bartolome
# See LICENSE for details.

from investpy import get_stock_historical_data

import numpy as np
import pandas as pd

from statistics import mean
import numpy
from unidecode import unidecode

import gym
from gym import spaces

#datetime is the library used to manipulate time and date
from datetime import datetime

from decimal import Decimal
from sklearn.linear_model import LinearRegression
import string
import global_config

MK = global_config.MK

def identify_df_trends(df, column, window_size=5, identify='both'):
    """
    This function receives as input a pandas.DataFrame from which data is going to be analysed in order to
    detect/identify trends over a certain date range. A trend is considered so based on the window_size, which
    specifies the number of consecutive days which lead the algorithm to identify the market behaviour as a trend. So
    on, this function will identify both up and down trends and will remove the ones that overlap, keeping just the
    longer trend and discarding the nested trend.

    Args:
        df (:obj:`pandas.DataFrame`): dataframe containing the data to be analysed.
        column (:obj:`str`): name of the column from where trends are going to be identified.
        window_size (:obj:`window`, optional): number of days from where market behaviour is considered a trend.
        identify (:obj:`str`, optional):
            which trends does the user wants to be identified, it can either be 'both', 'up' or 'down'.

    Returns:
        :obj:`pandas.DataFrame`:
            The function returns a :obj:`pandas.DataFrame` which contains the retrieved historical data from Investing
            using `investpy`, with a new column which identifies every trend found on the market between two dates
            identifying when did the trend started and when did it end. So the additional column contains labeled date
            ranges, representing both bullish (up) and bearish (down) trends.
    Raises:
        ValueError: raised if any of the introduced arguments errored.
    """

    if df is None:
        raise ValueError("df argument is mandatory and needs to be a `pandas.DataFrame`.")

    if not isinstance(df, pd.DataFrame):
        raise ValueError("df argument is mandatory and needs to be a `pandas.DataFrame`.")

    if column is None:
        raise ValueError("column parameter is mandatory and must be a valid column name.")

    if column and not isinstance(column, str):
        raise ValueError("column argument needs to be a `str`.")

    if isinstance(df, pd.DataFrame):
        if column not in df.columns:
            raise ValueError("introduced column does not match any column from the specified `pandas.DataFrame`.")
        else:
            if df[column].dtype not in ['int64', 'float64']:
                raise ValueError("supported values are just `int` or `float`, and the specified column of the "
                                 "introduced `pandas.DataFrame` is " + str(df[column].dtype))

    if not isinstance(window_size, int):
        raise ValueError('window_size must be an `int`')

    if isinstance(window_size, int) and window_size < 3:
        raise ValueError('window_size must be an `int` equal or higher than 3!')

    if not isinstance(identify, str):
        raise ValueError('identify should be a `str` contained in [both, up, down]!')

    if isinstance(identify, str) and identify not in ['both', 'up', 'down']:
        raise ValueError('identify should be a `str` contained in [both, up, down]!')

    df_result = pd.DataFrame(index=df.index, columns=['trend'])
    objs = list()

    up_trend = {
        'name': 'Up Trend',
        'element': np.negative(df[column])
    }

    down_trend = {
        'name': 'Down Trend',
        'element': df[column]
    }

    if identify == 'both':
        objs.append(up_trend)
        objs.append(down_trend)
    elif identify == 'up':
        objs.append(up_trend)
    elif identify == 'down':
        objs.append(down_trend)

    results = dict()


    for obj in objs:
        limit = None
        values = list()

        trends = list()

        for index, value in enumerate(obj['element'], 0):
            if limit and limit > value:
                values.append(value)
                limit = mean(values)
            elif limit and limit < value:
                if len(values) > window_size:
                    min_value = min(values)

                    for counter, item in enumerate(values, 0):
                        if item == min_value:
                            break

                    to_trend = from_trend + counter

                    trend = {
                        'from': df.index.tolist()[from_trend],
                        'to': df.index.tolist()[to_trend],
                    }
                    trends.append(trend)

                limit = None
                values = list()
            else:
                from_trend = index
                values.append(value)
                limit = mean(values)

        results[obj['name']] = trends

    if identify == 'both':
        up_trends = list()

        for up in results['Up Trend']:
            flag = True
            for down in results['Down Trend']:
                if down['from'] < up['from'] < down['to'] or down['from'] < up['to'] < down['to']:
                    if ((datetime.strptime(up['to'], '%m/%d/%Y') - (datetime.strptime(up['from'], '%m/%d/%Y')))) > ((datetime.strptime(down['to'], '%m/%d/%Y') - (datetime.strptime(down['from'], '%m/%d/%Y')))):
                        flag = True
                    else:
                        flag = False
                else:
                    flag = True

            if flag is True:
                up_trends.append(up)

        labels = [letter for letter in string.ascii_uppercase[:len(up_trends)]]

        for up_trend, label in zip(up_trends, labels):
            for index, row in df[up_trend['from']:up_trend['to']].iterrows():
                df_result.loc[index, 'trend'] = 1

        down_trends = list()

        for down in results['Down Trend']:
            flag = True

            for up in results['Up Trend']:
                if up['from'] < down['from'] < up['to'] or up['from'] < down['to'] < up['to']:
                    if ((datetime.strptime(up['to'], '%m/%d/%Y') - (datetime.strptime(up['from'], '%m/%d/%Y')))) < ((datetime.strptime(down['to'], '%m/%d/%Y') - (datetime.strptime(down['from'], '%m/%d/%Y')))):
                        flag = True
                    else:
                        flag = False
                else:
                    flag = True

            if flag is True:
                down_trends.append(down)

        labels = [letter for letter in string.ascii_uppercase[:len(down_trends)]]

        for down_trend, label in zip(down_trends, labels):
            for index, row in df[down_trend['from']:down_trend['to']].iterrows():
                df_result.loc[index, 'trend'] = 2

        for index , value in df.iterrows():
          if df_result.loc[index, 'trend'] !=1 and df_result.loc[index, 'trend'] !=2:
            df_result.loc[index, 'trend'] = 0

        return df_result
    

class TrendSlope:
    def __init__(self, iteration = None, minLimit=None, maxLimit=None, name = "Week", type = "test", columnName = "trend", frame = "Long"):
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
        self.frame = frame 

    def findDelta(self, begin, end):
        X = numpy.arange(1, 6).reshape(-1, 1) 
        y = self.Close[begin:end]
        model_term = LinearRegression().fit(X, y)
        slope_term = model_term.coef_[0]
        return slope_term

    def trendAddDelta(self, trendArr):

        for i in range(0,len(self.Date)):  
            if i-5>=0:
                delta = self.findDelta(i-5,i)
                trendArr[i] = delta
            else:
                if trendArr[i] == 2:
                    trendArr[i] = -1

        return trendArr


    def writeFile(self):
        ensambleValid=pd.DataFrame()
        ensambleValid.index.name='Date'
        self.spTimeserie.set_index('Date', inplace=True)
        trendResult = identify_df_trends(df = self.spTimeserie, column = 'Close',  window_size=5, identify='both')

        trendResult['trend'] = self.trendAddDelta(trendResult['trend'].tolist())

        for i in range(0,len(self.Date)):
            ensambleValid.at[trendResult.index[i],self.columnName]=trendResult['trend'][i]
        ensambleValid.to_csv("./Output/ensemble/"+"ensembleFolder"+"/walk"+self.name+str(self.iteration)+"ensemble_"+self.type+".csv")