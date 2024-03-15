#Copyright (C) 2020 Salvatore Carta, Anselmo Ferreira, Alessandro Sebastian Podda, Diego Reforgiato Recupero, Antonio Sanna. All rights reserved.

#Library used to manipulate the CSV Dataset
#organize the dataset for the Enviroment
import pandas

#Library used to manipulate dates
from datetime import datetime, timedelta
import global_config
#This is the prefix of the files that will be opened. It is related to the s&p500 stock market datasets
MK = global_config.MK

# def getDate_Index(Frame, date):
#     datesFrame = pandas.read_csv('./datasets/'+MK+"Hour"+'.csv')
#     Frame['Date'] = pandas.to_datetime(Frame['Date'], format='%m/%d/%Y')
#     specific_date = datesFrame.loc[date,'Date']

#     specific_date = pandas.to_datetime(specific_date, format='%m/%d/%Y')
#     next_date = Frame[Frame['Date'] >= specific_date].iloc[0]['Date']

#     date_to_find = pandas.to_datetime(next_date)
#     index = Frame.index[Frame['Date'] == date_to_find].tolist()

#     return index[0]

# def getTrendsWeek(Frame, date):
#     # result = []
#     date = datetime.strptime(date, "%m/%d/%Y")  # Chuyển đổi ngày đầu vào sang datetime
#     print(Frame)
#     # Đảm bảo rằng index của Frame là datetime để so sánh
#     # Frame.index = pandas.to_datetime(Frame.index)
    
#     print("*********")
#     for i in range(len(Frame)):
        
#         # Giả định rằng Frame.index đã là ngày tháng
#         if Frame.index[i] >= date:
#             print("----")
#             print(Frame.index[i])
#             print("----")
#             return Frame.index[i]

def getTrendsWeek(Frame, date):
    date = datetime.strptime(date, "%m/%d/%Y")  # Convert input date to datetime

    Frame['Date'] = pandas.to_datetime(Frame['Date'])

    # Tìm và in giá trị 'Date' đầu tiên thỏa mãn điều kiện
    for i in range(len(Frame)):
        if Frame['Date'].iloc[i] >= date:
            return Frame['Date'].iloc[i].strftime("%Y-%m-%d")




class MergedDataStructure():

    def __init__(self,filename="sp500Week.csv"):

        #Read the CSV
        self.timeserie = pandas.read_csv(filename)
        # print(self.timeserie)
        #Transform each column into a list
        Date = self.timeserie.loc[:, 'Date'].tolist()
        print(Date)
        Trend = self.timeserie.loc[:, 'trend'].tolist()

        #Create empty list and dictionary
        self.list=[]
        self.dict={}

        #The limit is the number of dates
        limit = len(Date)

        #Just converting pandas data to a list
        #lets pick up the csv data and put them in the list (self.list) 
        for i in range(0,limit-1):
            self.list.append({'Date' : Date[i],'Trend' : Trend[i]})
            
            #Fill the gaps with days that do not exist 
            dateList = [datetime.strptime(Date[i], "%m/%d/%Y") - timedelta(days=x) for x in range(0, (datetime.strptime(Date[i+1], "%m/%d/%Y") - datetime.strptime(Date[i], "%m/%d/%Y")).days)]
            
            for date in dateList:
                dateString=date.strftime("%m/%d/%Y")
                #Contains dates and indexes for the list self.list
                self.dict[dateString]=i
        # print(self.list)

    def get(self, date, delta = 5, name = "Day"):
        result = []
        if name == "Week":
            date = getTrendsWeek(self.timeserie, date)
            date = datetime.strptime(date, "%Y-%m-%d")
            date = date.strftime("%m/%d/%Y")
        dateString=str(date)
        # print(dateString)
        # # print(self.list)
        # print("&&&&")
        print(name)
        start = self.dict[dateString] + 1
        if self.dict[dateString]-(delta) + 1 < 0: 
            for i in range (0, delta - start):
                result.append(0)
            print(self.list[:self.dict[dateString] + 1])
            result.extend([item['Trend'] for item in self.list[0:self.dict[dateString]+1]])
        else:
            print(self.list[self.dict[dateString]-(delta) + 1:self.dict[dateString]+1])
            result.extend([item['Trend'] for item in self.list[self.dict[dateString]-(delta) + 1:self.dict[dateString]+1]])


        return result
      
