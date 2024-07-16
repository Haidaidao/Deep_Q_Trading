import pandas
import datetime
import numpy

class DailyWeeklyData:
    def __init__(self, filename, fill_interval, date_track_filename=None):
         #Read the CSV
        self.filename = filename
        self.timeseries = pandas.read_csv(filename)

        #Transform each column into a list
        Date = self.timeseries.loc[:, 'Date'].tolist()
        
        # Trend = self.timeseries.loc[:, 'trend'].tolist()
        Close = self.timeseries.loc[:, 'close'].tolist()
        Open = self.timeseries.loc[:, 'open'].tolist()

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
            self.list.append({'Date' : Date[i], 'Close': Close[i], 'Open': Open[i]})
            
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


    def get(self, date, window_size=1, name = "Day"):
        result = []

        sum = 0

        if name == "Week":
            sum=1

        og_date_str = date.strftime("%m/%d/%Y")

        # sync the request date with closest possible date before the required date (only get data that already know, not the data in the future)
        start = None
        while(start is None):
            try:
                if date <= self.first_date:
                    for i in range(window_size):
                        result.append({'Close': 1, 'Open': 1}) 
                    return result;

                dateString= date.strftime("%m/%d/%Y")
                start = self.dict[dateString] + 1
            except Exception:
                date = date - datetime.timedelta(days=1)
                
        if self.dict[dateString]-(window_size) <  0: 
            for i in range (window_size - start + 1):
                result.append({'Close': 1, 'Open': 1}) 
            #print(self.list[:self.dict[dateString] + sum])
            date_list = [ dateString ]
            result.extend([{'Close': item['Close'], 'Open': item['Open']}  for item in self.list[0 + sum:self.dict[dateString] + sum]])
        else:
            #print(self.list[self.dict[dateString]-(window_size)+sum:self.dict[dateString]+sum])
            date_list = [item['Date'] for item in self.list[self.dict[dateString]-(window_size) + sum:self.dict[dateString]+sum]]
            result.extend([{'Close': item['Close'], 'Open': item['Open']}  for item in self.list[self.dict[dateString]-(window_size) + sum:self.dict[dateString]+sum]])

        if self.date_track_filename != None:
            open(self.date_track_filename, 'a').writelines(f"{og_date_str} - {date_list}\n")
        # print(result)
        return result

    # def get(self, date, window_size=1, name="Day"):
    #     result = []

    #     sum = 0
    #     if name == "Week":
    #         sum = 1

    #     og_date_str = date.strftime("%m/%d/%Y")

    #     # Đồng bộ hóa ngày yêu cầu với ngày gần nhất trước ngày đó (chỉ lấy dữ liệu đã biết, không lấy dữ liệu trong tương lai)
    #     start = None
    #     while start is None:
    #         try:
    #             if date <= self.first_date:
    #                 # Trả về danh sách các từ điển với giá trị 0 cho 'Close' và 'Open', và ngày là ngày hiện tại
    #                 return [{'Date': date.strftime("%m/%d/%Y"), 'Close': 0, 'Open': 0} for _ in range(window_size)]

    #             dateString = date.strftime("%m/%d/%Y")
    #             start = self.dict[dateString] + 1
    #         except Exception:
    #             date = date - datetime.timedelta(days=1)

    #     # Xử lý dữ liệu cho mỗi khoảng thời gian
    #     if self.dict[dateString] - (window_size - 1) < 0:
    #         # Nếu yêu cầu lấy dữ liệu nằm trước danh sách
    #         print(self.list[self.dict[dateString]-(window_size)+sum:self.dict[dateString]+sum])
    #         for i in range(window_size - start):
    #             result.append({'Date': (date - datetime.timedelta(days=i)).strftime("%m/%d/%Y"), 'Close': 0, 'Open': 0})
    #         # Lấy dữ liệu từ danh sách và thêm vào kết quả
    #         result.extend([{'Date': item['Date'], 'Close': item['Close'], 'Open': item['Open']} for item in self.list[0:self.dict[dateString] + sum]])
    #     else:
    #         # Lấy đúng khoảng dữ liệu cần thiết
    #         result.extend([{'Date': item['Date'], 'Close': item['Close'], 'Open': item['Open']} for item in self.list[self.dict[dateString] - (window_size - 1) + sum: self.dict[dateString] + sum]])

    #     # Nếu có theo dõi ngày, ghi vào tệp
    #     if self.date_track_filename:
    #         date_list = [item['Date'] for item in result]
    #         with open(self.date_track_filename, 'a') as file:
    #             file.writelines(f"{og_date_str} - {date_list}\n")

    #     return result

