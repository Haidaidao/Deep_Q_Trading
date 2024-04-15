import pandas
import datetime
import numpy

class TrendReader:
    def __init__(self, filename, fill_interval, date_track_filename=None):
         #Read the CSV
        self.filename = filename
        self.timeseries = pandas.read_csv(filename)

        #Transform each column into a list
        Date = self.timeseries.loc[:, 'Date'].tolist()
        
        Trend = self.timeseries.loc[:, 'trend'].tolist()

        #Create empty list and dictionary
        self.list=[]
        self.dict={}

        if date_track_filename != None: # date tracker
            self.date_track_filename = date_track_filename
            open(self.date_track_filename, 'w').close()

        #The limit is the number of dates
        limit = len(Date)

        self.first_date = datetime.datetime.strptime(Date[0], "%m/%d/%Y")

        self.fill_interval = fill_interval
 
        #Just converting pandas data to a list
        #lets pick up the csv data and put them in the list (self.list) 
        for i in range(0,limit-1):
            self.list.append({'Date' : Date[i],'Trend' : Trend[i]})
            
            #Fill the gaps with days that do not exist 
            # dateList = [datetime.datetime.strptime(Date[i+1], "%m/%d/%Y") - datetime.timedelta(days=x) for x in range(0, ( datetime.datetime.strptime(Date[i+1], "%m/%d/%Y")- datetime.datetime.strptime(Date[i], "%m/%d/%Y") ).days )]
            
            dateList = self.fill_gaps_with_interval(datetime.datetime.strptime(Date[i], "%m/%d/%Y"), datetime.datetime.strptime(Date[i+1], "%m/%d/%Y"), self.fill_interval)
            
            for date in dateList:
                dateString=date.strftime("%m/%d/%Y")
                #Contains dates and indexes for the list self.list
                self.dict[dateString]=i

    # Function to fill gaps with a specified interval
    def fill_gaps_with_interval(self, start_date, end_date, interval):
        date_list = []
        current_date = start_date
        while current_date < end_date:
            date_list.append(current_date)
            current_date += datetime.timedelta(days=interval)
        return date_list

    # def get(self, date, window_size=1):
    #     result = []

    #     og_date_str = date.strftime("%m/%d/%Y")

    #     # sync the request date with the closest possible date before the required date
    #     # (only get data that is already known, not the data in the future)
    #     start = None
    #     while start is None:
    #         try:
    #             if date <= self.first_date:
    #                 return [numpy.zeros(5).tolist()]  # Assuming the zeroed list should be wrapped in a list to match the structure

    #             dateString = date.strftime("%m/%d/%Y")
    #             start = self.dict[dateString] + 1
    #         except Exception:
    #             date = date - datetime.timedelta(days=1)

    #     if self.dict[dateString] - (window_size) + 1 < 0:
    #         result = [0] * (window_size - start)  # Or create a list of zeroed lists or dicts as per your data structure
    #         result.extend(self.list[0:self.dict[dateString] + 1])
    #     else:
    #         result.extend(self.list[self.dict[dateString] - (window_size) + 1:self.dict[dateString] + 1])

    #     date_list = [item['Date'] for item in result]  # Assuming each item in the result has a 'Date' key

    #     if self.date_track_filename:
    #         with open(self.date_track_filename, 'a') as file:
    #             file.writelines(f"{og_date_str} - {date_list}\n")
    #     print(result)
    #     print("--------------------")
    #     return result


    def get(self, date, window_size=1):
        result = []

        og_date_str = date.strftime("%m/%d/%Y")

        # sync the request date with closest possible date before the required date (only get data that already know, not the data in the future)
        start = None
        while(start is None):
            try:
                if date <= self.first_date:
                    return numpy.zeros(window_size).tolist()

                dateString= date.strftime("%m/%d/%Y")
                start = self.dict[dateString] + 1
            except Exception:
                date = date - datetime.timedelta(days=1)
                
        if self.dict[dateString]-(window_size) + 1 < 0: 
            for i in range (0, window_size - start):
                result.append(0) 
            #print(self.list[:self.dict[dateString] + 1])
            date_list = [ dateString ]
            result.extend([item['Trend'] for item in self.list[0:self.dict[dateString]+1]])
        else:
            #print(self.list[self.dict[dateString]-(window_size) + 1:self.dict[dateString]+1])
            date_list = [item['Date'] for item in self.list[self.dict[dateString]-(window_size) + 1:self.dict[dateString]+1]]
            result.extend([item['Trend'] for item in self.list[self.dict[dateString]-(window_size) + 1:self.dict[dateString]+1]])

        if self.date_track_filename != None:
            open(self.date_track_filename, 'a').writelines(f"{og_date_str} - {date_list}\n")
        # print(result)
        return result