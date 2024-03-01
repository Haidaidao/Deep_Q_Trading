from trendet import identify_df_trends
import pandas as pd

def use_TrendWA(df, column = 'Close', window_size = 5):
    if window_size < 2: raise Exception("Window size must be larger than 2")

    df_result = pd.DataFrame(index=df.index, columns=['trend'])

    trends = []  # Danh sách để lưu trữ xu hướng của mỗi cửa sổ

    prices = df['Close'].to_list()
    
    # Duyệt qua mỗi cửa sổ con trong tập dữ liệu
    for i in range(len(prices) - window_size + 1):
        # Lấy cửa sổ con hiện tại
        window = prices[i:i+window_size]
        
        
        # Kiểm tra xu hướng tăng
        if all(window[j] < window[j+1] for j in range(window_size - 1)):
            trends.append(1)
        # Kiểm tra xu hướng giảm
        elif all(window[j] > window[j+1] for j in range(window_size - 1)):
            trends.append(-1)
        # Nếu không phải tăng hoặc giảm, xu hướng là Sideway
        else:
            trends.append(0)

    count = 0 
    for i in range(window_size - 1):
        trends = [0] + trends
    for index, row in df_result.iterrows():
        if count >= window_size -1: 
            df_result.at[index, 'trend'] = trends[count]
        else:
            df_result.at[index, 'trend'] = 0
        count = count + 1

    return df_result

def use_trendet(df, column = 'Close', window_size = 5):
    df_result = pd.DataFrame(index=df.index, columns=['trend'])

    trendResult = identify_df_trends(df=df, column=column, window_size=window_size)
    try:
        upLabels = trendResult['Up Trend'].dropna().unique().tolist()
    except:
        upLabels = None 
    try:
        downLabels = trendResult['Down Trend'].dropna().unique().tolist()
    except:
        downLabels = None   

    for index, row in trendResult.iterrows():
        trend = 0   
        if upLabels is not None and row['Up Trend'] in upLabels:
            trend = 1
        elif downLabels is not None and row['Down Trend'] in downLabels:
            trend = -1
        df_result.at[index,'trend']= trend

    return df_result