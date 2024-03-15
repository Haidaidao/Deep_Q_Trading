import pandas
import datetime

class TrendReader:
    def __init__(self, filename):
         #Read the CSV
        self.timeseries = pandas.read_csv(filename)

        #Transform each column into a list
        Date = self.timeseries.loc[:, 'Date'].tolist()
        
        Trend = self.timeseries.loc[:, 'trend'].tolist()

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

    def get(self, date, window_size=5):
        result = []
        start = None
        while(start is None):
            try:
                dateString=str(date)

                start = self.dict[dateString] + 1
            except Exception:
                date = date - datetime.timedelta(days=1)
                
        if self.dict[dateString]-(window_size) + 1 < 0: 
            for i in range (0, window_size - start):
                result.append(0) 
            # print(self.list[:self.dict[dateString] + 1])
            result.extend([item['Trend'] for item in self.list[0:self.dict[dateString]+1]])
        else:
            # print(self.list[self.dict[dateString]-(delta) + 1:self.dict[dateString]+1])
            result.extend([item['Trend'] for item in self.list[self.dict[dateString]-(window_size) + 1:self.dict[dateString]+1]])

        return result