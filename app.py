from flask import Flask, render_template, request
import twstock
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.dates import date2num
from mplfinance.original_flavor import candlestick_ochl
import plotly.graph_objs as go
import requests
from bs4 import BeautifulSoup
import numpy as np



# 獲取股票區間
def get_stock_data(stock_number):
    stock = twstock.Stock(stock_number)
    return stock.fetch_from(2019,1)

# 股票月開盤收盤
def process_stock_data(stock_data):
    grouped_data = {}
    for row in stock_data:
        date = row[0]
        year_month = (date.year, date.month)
        if year_month not in grouped_data:
            grouped_data[year_month] = {'開盤價': row[3], '最高價': row[4], '最低價': row[5], '收盤價': row[6]}
        else:
            grouped_data[year_month]['最高價'] = max(grouped_data[year_month]['最高價'], row[4])
            grouped_data[year_month]['最低價'] = min(grouped_data[year_month]['最低價'], row[5])
            grouped_data[year_month]['收盤價'] = row[6]
    
    df = pd.DataFrame(grouped_data).T.reset_index()
    df.columns = ['年份', '月份', '開盤價', '最高價', '最低價', '收盤價']
    df['年月'] = pd.to_datetime(df['年份'].astype(str) + '-' + df['月份'].astype(str), format='%Y-%m')
    df = df.sort_values(by='年月', ascending=False)
    # 计算 KD 值并添加到 DataFrame 中
    KD_values = calculate_stochastic_oscillator(df)
    df = pd.merge(df, KD_values, how='inner')
    return df

#計算均價
def avergeprice(data):
    ma_3 = data['收盤價'].rolling(window=3).mean()
    ma_6 = data['收盤價'].rolling(window=6).mean()
    ma_12 = data['收盤價'].rolling(window=12).mean()
    ma_20_plus = data['收盤價'].rolling(window=20).mean()
    print(1)
    ma_3_first = ma_3.iloc[2].round(2)
    ma_6_first = ma_6.iloc[5].round(2)
    ma_12_first = ma_12.iloc[11].round(2)
    ma_20_plus_first = ma_20_plus.iloc[19].round(2)




    return ma_3_first, ma_6_first, ma_12_first, ma_20_plus_first
#計算KD
def calculate_stochastic_oscillator(data, m=3):
    # 計算K值
    data['n_low'] = data['最低價']
    data['n_high'] = data['最高價']
    data['K'] = ((data['收盤價'] - data['n_low']) / (data['n_high'] - data['n_low'])) * 100
    data['K'] = data['K'].round(2)
    
    # 計算K線的m日移動平均值，即D線
    data['D'] = data['K'].rolling(window=m, min_periods=1).mean()
    data['D'] = data['D'].round(2)
    
    return data[['年月', 'K', 'D']].dropna()




# 找股票名稱
def get_company_short_name(stock_number):
    url = f"https://openapi.twse.com.tw/v1/opendata/t187ap03_L"
    response = requests.get(url)
    print(response)
    if response.status_code == 200:
        data = response.json()
        for company in data:
            if company["公司代號"] == str(stock_number):
                return company["公司簡稱"]
    else:
        print("Failed to fetch data from API")
        return None

#移動平均線
def add_moving_averages(fig, data):
    # 计算移动平均线
    ma_5 = data['收盤價'].rolling(window=5).mean()
    ma_10 = data['收盤價'].rolling(window=10).mean()
    ma_20 = data['收盤價'].rolling(window=20).mean()
    ma_50 = data['收盤價'].rolling(window=50).mean()
    ma_200 = data['收盤價'].rolling(window=200).mean()

    # 添加移动平均线到图表
    fig.add_trace(go.Scatter(x=data['年月'], y=ma_5, mode='lines', name='5日移動平均線'))
    fig.add_trace(go.Scatter(x=data['年月'], y=ma_10, mode='lines', name='10日移動平均線'))
    fig.add_trace(go.Scatter(x=data['年月'], y=ma_20, mode='lines', name='20日移動平均線'))
    fig.add_trace(go.Scatter(x=data['年月'], y=ma_50, mode='lines', name='50日移動平均線'))
    fig.add_trace(go.Scatter(x=data['年月'], y=ma_200, mode='lines', name='200日移動平均線'))
# 畫K線圖
# 畫K線圖
def generate_k_line_plot(data):
    fig = go.Figure(data=[go.Candlestick(x=data['年月'],
                                         open=data['開盤價'],
                                         high=data['最高價'],
                                         low=data['最低價'],
                                         close=data['收盤價'],
                                         increasing=dict(line=dict(color='red')),
                                         decreasing=dict(line=dict(color='green')))])
    
    # 添加移動平均線
    add_moving_averages(fig, data)
    
    # 添加K值和D值到圖表
    fig.add_trace(go.Scatter(x=data['年月'], y=data['K'], mode='lines', name='K值'))
    fig.add_trace(go.Scatter(x=data['年月'], y=data['D'], mode='lines', name='D值'))

    # 設置圖形佈局
    fig.update_layout(xaxis_rangeslider_visible=False)

    return fig.to_html(full_html=False)

def get_balance_sheet(stock_number):
    # 構建查詢股票營收的 URL
    url = f"https://goodinfo.tw/tw/StockAssetsStatus.asp?STOCK_ID={stock_number}"
    
    # 發送 GET 請求獲取網頁內容
    response = requests.get(url)
    
    # 如果請求成功
    if response.status_code == 200:
        # 使用 BeautifulSoup 解析 HTML 內容
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # 找到包含營收數據的表格
        table = soup.find('div', id='divDetail')
        
        # 創建一個空的列表來存儲有效的數據
        data = []
        
        # 遍歷表格的每一行
        for row in table.find_all('tr'):
            # 獲取每一行的所有列
            cols = row.find_all('td')
            
            # 將每一列的文本內容去除空白字符後存入列表
            cols = [col.text.strip() for col in cols]
            
            if len(cols) == 22:
                    data.append(cols)
        
        # 定義 DataFrame 的列名
        columns = ["年度", "股本", "財報評分", "年度股價_去年收盤", "年度股價_今年收盤", 
                   "年度股價_漲跌(元)", 
                   "年度股價_漲跌(%)", "資產比例_現金","資產比例_應收帳款",
                   "資產比例_存貨", "資產比例_流動資產", "資產比例_基金投資", "資產比例_固定資產",
                   "資產比例_無形資產","資產比例_其他資產", "負債比例_應付帳款", "負債比例_流動負債",
                   "負債比例_長期負債","負債比例_其他負債","負債比例_負債總額","股東權益(%)","BPS(元)"
                   ]
        # 將數據列表轉換為 DataFrame
        df = pd.DataFrame(data, columns=columns)
        print(df)
        return df

# 算出月營收
# 算出季營收
def get_monthly_quartly_revenue(stock_number):
    # 構建查詢股票營收的 URL
    url = f"https://goodinfo.tw/tw/ShowSaleMonChart.asp?STOCK_ID={stock_number}"
    
    # 發送 GET 請求獲取網頁內容
    response = requests.get(url)
    
    # 如果請求成功
    if response.status_code == 200:
        # 使用 BeautifulSoup 解析 HTML 內容
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # 找到包含營收數據的表格
        table = soup.find('div', id='divSaleMonChartDetail')
        
        # 創建一個空的列表來存儲有效的數據
        data = []
        
        # 遍歷表格的每一行
        for row in table.find_all('tr'):
            # 獲取每一行的所有列
            cols = row.find_all('td')
            
            # 將每一列的文本內容去除空白字符後存入列表
            cols = [col.text.strip() for col in cols]
            
            if len(cols) == 17:
                # 將月份拆分為年份和月份
                    year_month = cols[0].split('/')
                    year = year_month[0]
                    month = year_month[1]
                    # 添加年份和月份到列表中
                    data.append([year, month] + cols[1:])
        
        # 定義 DataFrame 的列名
        columns = ["年份", "月份", "開盤", "收盤", "最高", "最低", 
                   "漲跌", "漲跌(%)", "單月營收","單月月增",
                   "單月年增", "累計營收", "累計年增", "合併營業收入_單月營收",
                   "合併營業收入_月增","合併營業收入_年增", "累計營收", "累計年增"]
        
        # 將數據列表轉換為 DataFrame
        df = pd.DataFrame(data, columns=columns)
        
        df["去年同期月營收"] = df.apply(lambda x: df[(df["年份"] == str(int(x["年份"]) - 1)) & (df["月份"] == x["月份"])]["單月營收"].iloc[0] if len(df[(df["年份"] == str(int(x["年份"]) - 1)) & (df["月份"] == x["月份"])]) > 0 else None, axis=1)
        # 將字符串數據轉換為浮點數
        df["去年同期月營收"] = df["去年同期月營收"].str.replace(",", "").astype(float)
        df["單月營收"] = df["單月營收"].str.replace(",", "").astype(float)
        # 將錯誤值變為 0    
        df["YOY"] = np.where(pd.notnull(df["去年同期月營收"]) & pd.notnull(df["單月營收"]), ((df["單月營收"] - df["去年同期月營收"]) / df["去年同期月營收"]) * 100, 0)
        df["YOY"] = df["YOY"].round(2)
         # 將月營收添加到季度營收 DataFrame 中
        # 將月份轉換為季度
        df["季度"] = df["月份"].astype(int) // 4 + 1
        
        
        quarterly_revenue_df = pd.DataFrame(columns=["年份", "季度"])
        quarterly_revenue_df = pd.concat([quarterly_revenue_df, df.groupby(["年份", "季度"])["單月營收"].sum().reset_index()], ignore_index=True)
        # 排序
        quarterly_revenue_df = quarterly_revenue_df.sort_values(by=['年份', "季度"], ascending=[False,False])
        quarterly_revenue_df["單月營收"]=quarterly_revenue_df["單月營收"].round(2)
        quarterly_revenue_df["去年同季營收"]=quarterly_revenue_df.apply(lambda x: df[(df["年份"] == str(int(x["年份"]) - 1)) & (df["季度"] == x["季度"])]["單月營收"].sum() if len(df[(df["年份"] == str(int(x["年份"]) - 1)) & (df["季度"] == x["季度"])]) > 0 else None, axis=1)
        quarterly_revenue_df["去年同季營收"]=quarterly_revenue_df["去年同季營收"].round(2)
        # 將錯誤值變為 0    
        quarterly_revenue_df["YOY"] = np.where(pd.notnull(quarterly_revenue_df["去年同季營收"]) & pd.notnull(quarterly_revenue_df["單月營收"]), ((quarterly_revenue_df["單月營收"] - quarterly_revenue_df["去年同季營收"]) / quarterly_revenue_df["去年同季營收"]) * 100, 0)
        quarterly_revenue_df["YOY"] = quarterly_revenue_df["YOY"].round(2)
        print(quarterly_revenue_df)
        
        return df,quarterly_revenue_df
    else:
        # 如果請求失敗，則打印錯誤信息並返回None
        print(f"Failed to fetch data from {url}. Status code: {response.status_code}")
        return None


# 抓獲利情況   
def get_profit(stock_number):
    # 構建查詢股票營收的 URL
    url = f"https://goodinfo.tw/tw/StockDetail.asp?STOCK_ID={stock_number}"
    
    # 發送 GET 請求獲取網頁內容
    response = requests.get(url)
    
    # 如果請求成功
    if response.status_code == 200:
        # 使用 BeautifulSoup 解析 HTML 內容
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # 找到包含營收數據的表格
        table = soup.find('div', id='FINANCE_INCOME_M')
        
        # 創建一個空的列表來存儲有效的數據
        data = []   
        
        # 遍歷表格的每一行
        for row in table.find_all('tr'):
            # 獲取每一行的所有列
            cols = row.find_all('td')
            
            # 將每一列的文本內容去除空白字符後存入列表
            cols = [col.text.strip() for col in cols]
            data.append(cols)

  
        # 定義 DataFrame 的列名
        columns = ["年季", "營收(億)", "稅後淨利(億)", "毛利(%)", "營業利益(%)", "稅後淨利(%)", 
                   "ROE(%)", "EPS(元)"]
        data=data[5:]
        
        # 將數據列表轉換為 DataFrame
        df = pd.DataFrame(data, columns=columns)
        return df
    else:
        # 如果請求失敗，則打印錯誤信息並返回None
        print(f"Failed to fetch data from {url}. Status code: {response.status_code}")
        return None

# 回饋到前端
def render_template_data(stock_data, k_line_plot, financial_data, quarterly_revenue, profit_data, balance_sheet, company_short_name, avgprice):
    return render_template('index.html', stock_data=stock_data, k_line_plot=k_line_plot,
                           financial_data=financial_data, quarterly_revenue=quarterly_revenue, 
                           profit_data=profit_data, balance_sheet=balance_sheet, company_short_name=company_short_name, avgprice=avgprice)

app = Flask(__name__)
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        stock_number = request.form['stock_number']
        
        # 股票代號公司簡稱
        company_short_name = get_company_short_name(stock_number)
        # 獲取原始股票數據
        stock_data = get_stock_data(stock_number)
        # 處理月開盤
        processed_stock_data = process_stock_data(stock_data)
        #均價
        avgprice=avergeprice(processed_stock_data)
        # K圖
        k_line_plot = generate_k_line_plot(processed_stock_data[['年月','開盤價', '最高價', '最低價', '收盤價',"K","D"]])
        # 財務
        financial_data, quarterly_revenue = get_monthly_quartly_revenue(stock_number)
        # 淨利
        profit_data = get_profit(stock_number)
        # 資產負債
        balance_sheet = get_balance_sheet(stock_number)
        
        return render_template_data(processed_stock_data, k_line_plot, financial_data, quarterly_revenue, profit_data, balance_sheet, company_short_name, avgprice)
    
    # 空的時候顯示
    return render_template_data(None, None, None, None, None, None, None,None)






if __name__ == '__main__':
    app.run(debug=True)
