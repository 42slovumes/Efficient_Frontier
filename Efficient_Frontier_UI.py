import gradio as gr
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

stockCodeArray_split = []

year = []
month = []
day = []

for i in range(2000, 2023):
    year.append(str(i))
for i in range(1, 13):
    if(i < 10):
        month.append("0"+str(i))
    else:
        month.append(str(i))
for i in range(1, 32):
    if(i < 10):
        day.append("0"+str(i))
    else:
        day.append(str(i))

df = pd.read_csv('stock_codes.csv')
first_column_values = []
for index, row in df.iterrows():
    first_column_values.append(row[0])
stockCodeArray = [str(value) for value in first_column_values]
# stockCodeArray2 = [str(value).split()[0] for value in first_column_values]

def Efficient_Frontier(stock_code, start_year_downInput, start_month_downInput, start_day_downInput, end_year_downInput,
                       end_month_downInput, end_day_downInput, key, text_1, text_2, text_3, text_4, text_5, text_6):
    def port_ret(weights):
        return np.sum(avg_daily_return * weights) * 252


    def port_vol(weights):
        return np.sqrt(np.dot(weights.T, np.dot(daily_returns.cov() * 252, weights)))
    prets = []
    pvols = []
    startday = start_year_downInput + "-" + \
        start_month_downInput + "-" + start_day_downInput
    endday = end_year_downInput + "-" + end_month_downInput + "-" + end_day_downInput
    if not stock_code:
        return "請選擇股票代碼"
    if len(startday) != 10:
        return "請選擇起始時間"
    if len(endday) != 10:
        return "請選擇結束時間"
    stockCodeArray_split = [str(value).split()[0] for value in stock_code]


    wt = [text_1, text_2, text_3, text_4, text_5, text_6]
    tickers = stockCodeArray_split
    if len(stockCodeArray_split) < len(wt):
        wt = wt[:len(stockCodeArray_split)]
    print(wt)
    


    start_date = startday
    end_date = endday
    data = yf.download(tickers, start=start_date, end=end_date)
    data.to_csv('portfolio.csv')
    df = pd.read_csv('portfolio.csv', index_col=0)

    dfarray = []
    for i, index in enumerate(stockCodeArray_split):
        if i == 0:
            dfarray.append('Adj Close')
        else:
            dfarray.append('Adj Close.'+str(i))
    print(dfarray)
    df1 = df[dfarray]
    df1 = df1.drop(df1.index[0])
    df1 = df1.dropna()
    df1.columns = stockCodeArray_split
    for i, ticker in enumerate(tickers):
        df1[ticker] = df1[ticker].astype(float)
    print(df1)

    daily_returns = df1.pct_change()
    avg_daily_return = daily_returns.mean()
    avg_annual_return = avg_daily_return * 252
    daily_volatility = daily_returns.std()
    annual_volatility = daily_volatility * (252 ** 0.5)
    daily_returns.cov() * 252
    for p in range(2500):
        weights = np.random.random(len(stockCodeArray_split))
        weights /= np.sum(weights)
        prets.append(np.sum(avg_daily_return * weights) * 252)
        pvols.append(
            np.sqrt(np.dot(weights.T, np.dot(daily_returns.cov() * 252, weights))))
    prets = np.array(prets)
    pvols = np.array(pvols)

    if key == "plot":
        plt.figure(figsize=(10, 6))
        plt.scatter(pvols, prets, c=prets / pvols,
                    marker='o', cmap='coolwarm')
        plt.xlabel('expected volatility')
        plt.ylabel('expected return')
        plt.colorbar(label='Sharpe ratio')
        plt.show()
        return "這是未加權重的圖"
    elif key == "plotWithX":
        if round(sum(wt), 2) != 1:


            return "請確認權重是否正確"
        print(round(sum(wt), 2))
        plt.figure(figsize=(10, 6))
        plt.scatter(pvols, prets, c=prets / pvols,
                    marker='o', cmap='coolwarm')
        plt.scatter(port_vol(np.array(wt)), port_ret(np.array(wt)), c='green',
                    marker='x', cmap='coolwarm')
        plt.xlabel('expected volatility')
        plt.ylabel('expected return')
        plt.colorbar(label='Sharpe ratio')
        plt.show()
        return "這是加權重的圖"

with gr.Blocks(css="#testpls { width : 100% ; height : 67px ;  } #testpls2 {margin-top : 10px ; width : 100% ; height : 67px } ") as demo:
    with gr.Box():
        with gr.Row():
            gr.HighlightedText(value="第七週 效率前緣 作業範例",
                               label="", interactive=False)
            key1 = gr.HighlightedText(
                value="plot", label="", interactive=False, visible=False)
            key2 = gr.HighlightedText(
                value="plotWithX", label="", interactive=False, visible=False)
    with gr.Tab("效率前緣圖_再圖上加上你的權重配置的點"):
        with gr.Row():
            with gr.Column():
                stock_code = gr.Dropdown(
                    stockCodeArray, label="選擇代號，最多六個(有點卡 等他一下)", multiselect=True, max_choices=6)
        with gr.Box():
            with gr.Row():
                gr.Markdown(
                    value="請選擇您的權重(要依照下拉選單選擇的順序，權重加起來要等於1) 選擇的權重範例 : 0.1 ", label="")
            with gr.Row():
                text_1 = gr.Slider(0, 1, step=0.1, label="代號1權重", value=0)
                text_2 = gr.Slider(0, 1, step=0.1, label="代號2權重", value=0)
                text_3 = gr.Slider(0, 1, step=0.1, label="代號3權重", value=0)
                text_4 = gr.Slider(0, 1, step=0.1, label="代號4權重", value=0)
                text_5 = gr.Slider(0, 1, step=0.1, label="代號5權重", value=0)
                text_6 = gr.Slider(0, 1, step=0.1, label="代號6權重", value=0)

        with gr.Row():
            with gr.Accordion("選擇起始時間"):
                with gr.Row():
                    start_year_downOptions = year
                    start_year_downInput = gr.inputs.Dropdown(
                        start_year_downOptions, label="年")
                    start_month_downOptions = month
                    start_month_downInput = gr.inputs.Dropdown(
                        start_month_downOptions, label="月")
                    start_day_downOptions = day
                    start_day_downInput = gr.inputs.Dropdown(
                        start_day_downOptions, label="日")
            with gr.Accordion("選擇結束時間"):
                with gr.Row():
                    end_year_downOptions = year
                    end_year_downInput = gr.inputs.Dropdown(
                        end_year_downOptions, label="年")
                    end_month_downOptions = month
                    end_month_downInput = gr.inputs.Dropdown(
                        end_month_downOptions, label="月")
                    end_day_downOptions = day
                    end_day_downInput = gr.inputs.Dropdown(
                        end_day_downOptions, label="日")

        with gr.Box():
            with gr.Row():
                gr.Markdown(
                    value="選擇完股票代號，起始時間以及結束時間後即可按下按鈕1，選擇完權重後才能按下按鈕2", label="")
            with gr.Row():
                with gr.Column():
                    btn1 = gr.Button(value="按鈕1 : 效率前緣圖")
                with gr.Column():
                    btn2 = gr.Button(value="按鈕2 : 加上你的權重")
        with gr.Box():
            with gr.Row():
                with gr.Column():
                    text_output = gr.Textbox(
                        lines=5, label="輸出", placeholder="系統輸出區域")
    btn1.click(Efficient_Frontier, inputs=[
        stock_code, start_year_downInput, start_month_downInput, start_day_downInput, end_year_downInput, end_month_downInput, end_day_downInput, key1, text_1, text_2, text_3, text_4, text_5, text_6], outputs=[text_output])
    btn2.click(Efficient_Frontier, inputs=[stock_code, start_year_downInput, start_month_downInput, start_day_downInput, end_year_downInput,
               end_month_downInput, end_day_downInput, key2, text_1, text_2, text_3, text_4, text_5, text_6], outputs=[text_output])

demo.launch()
