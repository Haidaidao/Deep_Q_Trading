import yfinance as yf
import datetime
import pandas as pd

# Define the stock symbol and time range
stock_symbol = "BTX-USD"  
start_date = datetime.datetime(2022, 2, 5)
end_date = datetime.datetime(2024, 1, 26)

end_date = datetime.datetime.now().date()

start_hourly = end_date - datetime.timedelta(days=720) # roughly 2 years from the time running
start_daily_weekly = end_date - datetime.timedelta(days=1095) # roughly 3 years from the time running

# Download hourly data
dataHourly = yf.download(stock_symbol, start=start_hourly, end=end_date, interval='1h')
dataDaily = yf.download(stock_symbol, start=start_daily_weekly, end=end_date, interval='1d')
dataWeekly= yf.download(stock_symbol, start=start_daily_weekly, end=end_date, interval='1wk')

dataHourly['Date'] = dataHourly.index.date
dataHourly['Time'] = dataHourly.index.strftime('%H:%M')
dataHourly['Date'] = dataHourly['Date'].apply(lambda x: x.strftime('%m/%d/%Y'))

dataDaily['Time'] = '00:00'
dataDaily.index = pd.to_datetime(dataDaily.index).strftime('%m/%d/%Y')

dataWeekly['Time'] = '00:00'
dataWeekly.index =  pd.to_datetime(dataWeekly.index).strftime('%m/%d/%Y')

dataHourly = dataHourly[['Date', 'Time', 'Open', 'High', 'Low', 'Close', 'Volume']]
dataDaily = dataDaily[['Time', 'Open', 'High', 'Low', 'Close', 'Volume']]
dataWeekly = dataWeekly[['Time', 'Open', 'High', 'Low', 'Close', 'Volume']]

# Display the data
dataHourly.to_csv(f"{stock_symbol}Hour.csv", index=False)
dataDaily.to_csv(f"{stock_symbol}Day.csv")
dataWeekly.to_csv(f"{stock_symbol}Week.csv")