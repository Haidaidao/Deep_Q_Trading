#Copyright (C) 2020 Salvatore Carta, Anselmo Ferreira, Alessandro Sebastian Podda, Diego Reforgiato Recupero, Antonio Sanna. All rights reserved.

#Library used to manipulate the CSV Dataset
#organize the dataset for the Enviroment
import pandas

#Library used to manipulate dates
import datetime
import global_config
#This is the prefix of the files that will be opened. It is related to the s&p500 stock market datasets
MK = global_config.MK

def getDate_Index(Frame, date):
    datesFrame = pandas.read_csv('./datasets/'+MK+"Hour"+'.csv')
    Frame['Date'] = pandas.to_datetime(Frame['Date'], format='%m/%d/%Y')
    specific_date = datesFrame.loc[date,'Date']

    specific_date = pandas.to_datetime(specific_date, format='%m/%d/%Y')
    next_date = Frame[Frame['Date'] >= specific_date].iloc[0]['Date']

    date_to_find = pandas.to_datetime(next_date)
    index = Frame.index[Frame['Date'] == date_to_find].tolist()

    return index[0]

class MergedDataStructure():

    def __init__(self,filename="sp500Week.csv"):

        #Read the CSV
        self.timeserie = pandas.read_csv(filename)
        
        #Transform each column into a list
        Date = self.timeserie.loc[:, 'Date'].tolist()
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
            dateList = [datetime.datetime.strptime(Date[i+1], "%m/%d/%Y") - datetime.timedelta(days=x) for x in range(0, ( datetime.datetime.strptime(Date[i+1], "%m/%d/%Y")- datetime.datetime.strptime(Date[i], "%m/%d/%Y") ).days )]
            
            for date in dateList:
                dateString=date.strftime("%m/%d/%Y")
                #Contains dates and indexes for the list self.list
                self.dict[dateString]=i

    def get(self, date, name = "Day"):
        if name == "Day":
            #Converts the date to string
            dateString=str(date)
            #given the date, you get an interval of past days or weeks
            return self.list[self.dict[dateString]]
      
