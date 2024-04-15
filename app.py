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

# 畫K線圖
def generate_k_line_plot(data):
    fig = go.Figure(data=[go.Candlestick(x=data['年月'],
                                         open=data['開盤價'],
                                         high=data['最高價'],
                                         low=data['最低價'],
                                         close=data['收盤價'],
                                         increasing=dict(line=dict(color='red')),
                                         decreasing=dict(line=dict(color='green')))])
    fig.update_layout(xaxis_rangeslider_visible=False)
    k_line_plot = fig.to_html(full_html=False)
    return k_line_plot

# 算出月營收
# 算出季營收
# 創建一個空的 DataFrame 來存儲季度營收

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
        df["去年同期月營收"] = df["去年同期月營收"].astype(float)
        df["單月營收"] = df["單月營收"].astype(float)
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

app = Flask(__name__)
@app.route('/', methods=['GET', 'POST'])



def index():
    if request.method == 'POST':
        stock_number = request.form['stock_number']
        
        # 獲取股票數據
        stock = twstock.Stock(stock_number)
        target_price = stock.fetch_from(2019, 10)
        
        #print(target_price)
        # 對數據進行預處理和分組
        grouped_data = {}
        for row in target_price:
            date = row[0]
            year_month = (date.year, date.month)
            if year_month not in grouped_data:
                grouped_data[year_month] = {'開盤價': row[3], '最高價': row[4], '最低價': row[5], '收盤價': row[6]}
            else:
                grouped_data[year_month]['最高價'] = max(grouped_data[year_month]['最高價'], row[4])
                grouped_data[year_month]['最低價'] = min(grouped_data[year_month]['最低價'], row[5])
                grouped_data[year_month]['收盤價'] = row[6]
        
        # 轉換為 DataFrame，並添加年月
        df = pd.DataFrame(grouped_data).T.reset_index()
        df.columns = ['年份', '月份', '開盤價', '最高價', '最低價', '收盤價']
        df['年月'] = pd.to_datetime(df['年份'].astype(str) + '-' + df['月份'].astype(str), format='%Y-%m')
        
        # 獲取股票月營收數據及季營收
        financial_data, quarterly_revenue = get_monthly_quartly_revenue(stock_number)
        # 獲取股票淨利
        profit_data = get_profit(stock_number)
        
        # 生成K線圖
        k_line_plot = generate_k_line_plot(df[['年月','開盤價', '最高價', '最低價', '收盤價']])
        
        # 將股票數據及K財務都丟回前端
        return render_template('index.html', stock_data=df, k_line_plot=k_line_plot,
                               financial_data=financial_data, quarterly_revenue=quarterly_revenue, profit_data=profit_data)
    
    # 用戶未提交前都為none
    return render_template('index.html', stock_data=None, k_line_plot=None,
                           financial_data=None, quarterly_revenue=None,profit_data=None)

if __name__ == '__main__':
    app.run(debug=True)
