import streamlit as st
import pandas as pd
import yfinance as yf
from finta import TA
import datetime as dt
import numpy as np
import mplfinance as mpf
from plotly import express as px
##

##

st.set_page_config(page_title="Daily Intraday Analysis", 
                   page_icon="🤖",
                   layout="wide")

st.title("Daily Intraday Analysis")
# Load data -- data functions
@st.cache_data
def get_sp500_components():
    df = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")
    df = df[0]
    tickers = df["Symbol"].to_list()
    extended_symbols = ['RIVN', 'AVGO', 'SPY', 'QQQ', 'TSLA', 'MA', 'BITO','BTC-USD', 'NIO', 'LCID', 'GME', 'SPCE', 'NKLA']
    extended_companies = ['Rivian Automotive', 'Broadcom Inc', 'SPDR S&P 500 ETF', 'Invesco QQQ Trust', 'Tesla', 'Mastercard', 'BITO','BTCUSD', 'NIO', 'LUCID', 'GME', 'virgin galactic', 'Nikola']
    # Combine tickers with extended symbols
    tickers.extend(extended_symbols)
    tickers_companies_dict = dict(zip(df["Symbol"], df["Security"]), **dict(zip(extended_symbols, extended_companies)))
    return tickers, tickers_companies_dict

@st.cache_data()
def load_data(ticker,interval):

     if interval == '1m':
        stock_df = yf.download(tickers = [ticker], interval= '1m')
     if interval == '5m':
        stock_df = yf.download(tickers = [ticker], period = '60d', interval= '5m')
     if interval == '15m':
        stock_df = yf.download(tickers = [ticker], period = '60d', interval= '15m')
     if interval == '1h':
        stock_df = yf.download(tickers = [ticker], period = '500d',interval= '1h')
     if interval == '1d':
        stock_df = yf.download(tickers = ticker, period = '10y', interval= '1d')


     return stock_df

@st.cache_data
def get_industry_data(stocks, period):
    #formatted_end_date = datetime.strptime(end_date, "%Y-%m-%d").strftime("%Y%m%d")
    #stocks = ['XLK', 'XLB', 'XLI', 'XLE', 'XLV', 'XLY', 'XLF', 'XLU', 'XLP']
    data = yf.download(stocks, period = period, interval = '1d').dropna()
    data_close = round(data['Adj Close'], 2)
    df_relative = round(data_close / data_close.iloc[0] * 100, 1)
    # Function to calculate trend for different periods
    def calculate_trend(stock_data, period):
        if len(stock_data) >= period:
            trend_diff = stock_data.iloc[-1] - stock_data.iloc[-period]
            return 'Rising' if trend_diff > 0 else 'Falling'
        else:
            return 'Not Enough Data'

    # Create a trends DataFrame with multiple trend periods
    trends = []
    for stock in stocks:
        trends.append({
            'Sector': stock,
            'Trend5': calculate_trend(df_relative[stock], 5),
            'Trend10': calculate_trend(df_relative[stock], 10),
            'Trend20': calculate_trend(df_relative[stock], 20),
            'Trend50': calculate_trend(df_relative[stock], 50),
            'Trend100': calculate_trend(df_relative[stock], 100),
            'Trend150': calculate_trend(df_relative[stock], 150)
        })
    
    trend_df = pd.DataFrame(trends)
            
    return df_relative, data_close, trend_df      
    
def highlight_trends(val):
    """
    Highlight the cell red if 'Falling' and green if 'Rising'.
    """
    color = 'green' if val == 'Rising' else 'red' if val == 'Falling' else ''
    return f'background-color: {color}'
# Analyze the trend for each quartile subset
def analyze_trend(subset):
        if subset['Close'].iloc[-1] > subset['Close'].iloc[0]:
            return 'Bullish'
        elif subset['Close'].iloc[-1] < subset['Close'].iloc[0]:
            return 'Bearish'
        else:
            return 'Neutral'

def find_trend(df):
     # Sort the dataframe by the "Date" column
    # = df.sort_values(by = 'Datetimne')

    # Calculate quartiles
    quartiles = df['Close'].quantile([0.05, 0.10, 0.25, 0.5, 0.75, 1])
    #print(quartiles)

    # Split the dataframe into quartile subsets
    q1 = df[df['Close'] <= quartiles[0.05]]
    q2 = df[(df['Close'] > quartiles[0.05]) & (df['Close'] <= quartiles[0.10])]
    q3 = df[(df['Close'] > quartiles[0.10]) & (df['Close'] <= quartiles[0.25])]
    q4 = df[(df['Close'] > quartiles[0.25]) & (df['Close'] <= quartiles[0.5])]
    q5 = df[(df['Close'] > quartiles[0.5]) & (df['Close'] <= quartiles[0.75])]
    q6 = df[(df['Close'] > quartiles[0.75]) & (df['Close'] <= quartiles[1])]

    #print(q1,q2,q3,q4,q5,q6)

    
    trend_q1 = analyze_trend(q1)
    trend_q2 = analyze_trend(q2)
    trend_q3 = analyze_trend(q3)
    trend_q4 = analyze_trend(q4)
    trend_q5 = analyze_trend(q5)
    trend_q6 = analyze_trend(q6)
    
    return trend_q1, trend_q2, trend_q3, trend_q4, trend_q5, trend_q6

#Generating Colors For Histogram
def gen_macd_color(df):
    macd_color = []
    macd_color.clear()
    for i in range (0,len(df["MACDh_12_26_9"])):
        if df["MACDh_12_26_9"][i] >= 0 and df["MACDh_12_26_9"][i-1] < df["MACDh_12_26_9"][i]:
            macd_color.append('#26A69A')
            #print(i,'green')
        elif df["MACDh_12_26_9"][i] >= 0 and df["MACDh_12_26_9"][i-1] > df["MACDh_12_26_9"][i]:
            macd_color.append('#B2DFDB')
            #print(i,'faint green')
        elif df["MACDh_12_26_9"][i] < 0 and df["MACDh_12_26_9"][i-1] > df["MACDh_12_26_9"][i] :
            #print(i,'red')
            macd_color.append('#FF5252')
        elif df["MACDh_12_26_9"][i] < 0 and df["MACDh_12_26_9"][i-1] < df["MACDh_12_26_9"][i] :
            #print(i,'faint red')
            macd_color.append('#FFCDD2')
        else:
            macd_color.append('#000000')
            #print(i,'no')
    return macd_color

def color(goldencrossover):
                UP = []
                DOWN = []
                for i in range(len(goldencrossover)):
                    if goldencrossover[maflag2][i] < goldencrossover[maflag1][i]:
                        UP.append(float(goldencrossover[maflag2][i]))
                        DOWN.append(np.nan)
                    elif goldencrossover[maflag2][i] > goldencrossover[maflag1][i]:
                        DOWN.append(float(goldencrossover[maflag2][i]))
                        UP.append(np.nan)
                    else:
                        UP.append(np.nan)
                        DOWN.append(np.nan)
                goldencrossover['up'] = UP
                goldencrossover['down'] = DOWN
                return goldencrossover
        
def golden_cal(df):
                goldenSignal = []
                deathSignal = []
                position = False
                for i in range(len(df)):
                    if df[maflag1][i] > df[maflag2][i]:
                        if position == False :
                            goldenSignal.append((df[maflag2][i]-df[maflag2][i]*0.01))
                            deathSignal.append(np.nan)
                            position = True
                        else:
                            goldenSignal.append(np.nan)
                            deathSignal.append(np.nan)
                    elif df[maflag1][i] < df[maflag2][i]:
                        if position == True:
                            goldenSignal.append(np.nan)
                            deathSignal.append((df[maflag2][i]+df[maflag2][i]*0.01))
                            position = False
                        else:
                            goldenSignal.append(np.nan)
                            deathSignal.append(np.nan)
                    else:
                        goldenSignal.append(np.nan)
                        deathSignal.append(np.nan)
                df['GoldenCrossOver'] = goldenSignal
                df['DeathCrossOver'] = deathSignal

     

with st.sidebar:
    stock_df = pd.DataFrame()
    with st.sidebar.form("Submit",clear_on_submit=False, border= True):
        default_val = "********************"
        available_tickers, tickers_companies_dict = get_sp500_components()
        selected_ticker = st.selectbox("Select Ticker", available_tickers, format_func=tickers_companies_dict.get,placeholder='Choose a value')
        interval = st.selectbox("Select Data Interval", ['1m', '5m', '15m', '1h', '1d'] )
        print(interval)
        #if submitted_form:
        stock_df = load_data(selected_ticker, interval)
       #st.sidebar.dataframe(stock_df[['Open','High','Low','Close']].tail(5),hide_index=False)
            #st.sidebar.caption('_Last_ :blue[5] Candles :stars:')
            #st.caption('A caption with _italics_ :blue[colors] and emojis :sunglasses:')
        
        #st.sidebar.line_chart(stock_df['Close'],height=500)
        #st.sidebar.caption('Volume:blue[BarChart]:stars:')
        #st.sidebar.bar_chart(stock_df['Volume'].tail(200),height=100)

        #print(f"selected_ticker is {selected_ticker}")
        #stock_df = load_data(selected_ticker, interval)
        #print(stock_df.columns)
        submitted_form = st.form_submit_button("Submit")

    st.sidebar.caption(f'{selected_ticker} ClosingPrice:blue[LineChart]:stars:')
    fig = px.line(stock_df, x=stock_df.index, y=stock_df.Close, template= 'simple_white')
    st.sidebar.plotly_chart(fig,use_container_width=True)
        

    chart_styles = [
            'default', 'binance', 'blueskies', 'brasil', 
            'charles', 'checkers', 'classic', 'yahoo',
            'mike', 'nightclouds', 'sas', 'starsandstripes'
        ]
        
    chart_style = st.selectbox('Chart style', options=chart_styles, index=chart_styles.index('yahoo'))
        
    chart_types = [
            'candle', 'ohlc', 'line', 'renko', 'pnf'
        ]

col = st.columns((10, 3), gap='small')
with col[0]:
    #st.write("Column 0")
    #st.dataframe(stock_df)
    if interval == '1d':
            ohlc = stock_df.sort_values(by=['Date'],ascending=True)
    else:
            ohlc = stock_df.sort_values(by=['Datetime'],ascending=True)
            
    ta_df = pd.DataFrame()
    ta_df['open'] = round(ohlc['Open'],2)
    ta_df['close'] = round(ohlc['Close'],2)
    ta_df['volume'] = round(ohlc['Volume'],2)
    ta_df['BB_Upper'] = round(TA.MOBO(ohlc),2)['BB_UPPER']
    #ta_df['BB_Middle'] = round(TA.MOBO(ohlc),2)['BB_MIDDLE']
    ta_df['BB_Lower'] = round(TA.MOBO(ohlc),2)['BB_LOWER']
    #Calculation For Simple Moving Average For Length 5 as Short SMA
    ta_df['SMA5'] = round(ohlc['Close'].rolling(window=5).mean(), 2)
    ta_df['SMA_9'] = round(TA.SMA(ohlc,9),2)
    ta_df['SMA_20'] = round(TA.SMA(ohlc,20),2)
    #Calculation For Simple Moving Average For Length 34 as Long SMA
    ta_df['SMA34'] = round(ohlc['Close'].rolling(window=34).mean(), 2)
    ta_df['SMA_50'] = round(TA.SMA(ohlc,50),2)
    ta_df['SMA_100'] = round(TA.SMA(ohlc,100),2)
    ta_df['MACD'] =  round(TA.MACD(ohlc),2)['MACD']
    ## calculating owesome oscillator where OA =  SMA (MEDIAN PRICE, 5)-SMA (MEDIAN PRICE, 34) , Median Price = (HIGH+LOW)/2
    ## 
    # Awesome_Oscillator Column Assign To df
    ta_df['AO'] = ta_df['SMA5'] - ta_df['SMA34']
    ##
    ta_df['RSI'] =  round(TA.RSI(ohlc),2)
    #ta_df['IFT_RSI'] =  round(TA.IFT_RSI(ohlc),2)
    ta_df['SAR'] = round(TA.SAR(ohlc),2)
    #SAR stands for “stop and reverse,” which is the actual indicator used in the system.
    #    SAR trails price as the trend extends over time. The indicator is below prices when prices are rising and above prices when prices are falling.
    #    In this regard, the indicator stops and reverses when the price trend reverses and breaks above or below the indicator.
    
    ta_df['ROC'] = round(TA.ROC(ohlc),2)
    ta_df['high'] = round(ohlc['High'],2)
    ta_df['low'] = round(ohlc['Low'],2)
    #ta_df.to_csv('./data/ta_df.csv')
    ta_df = ta_df.sort_index(ascending=False)
    ta_df = ta_df.reset_index()
    st.dataframe(ta_df, hide_index=True)

    ###################
    ###### Next Plot the charts
    ###--------MACD Chart---------------------------------------------------------------------------------------------###
    st.write('Note:- Default Chart Style is nightclouds- you may select your style from the sidebar by Scrolling Down !!')
    st.write(f'{selected_ticker} - MACD <12_26_9> [Signal->Orange,MACD->Blue]')
    df_macd = stock_df.tail(250).copy()
            #Get the 26-day EMA of the closing price
    k = df_macd['Close'].ewm(span=12, adjust=False, min_periods=12).mean()
            #Get the 12-day EMA of the closing price
    d = df_macd['Close'].ewm(span=26, adjust=False, min_periods=26).mean()
            #Subtract the 26-day EMA from the 12-Day EMA to get the MACD
    macd = k - d
            #Get the 9-Day EMA of the MACD for the Trigger line
    macd_s = macd.ewm(span=9, adjust=False, min_periods=9).mean()
            #Calculate the difference between the MACD - Trigger for the Convergence/Divergence value
    macd_h = macd - macd_s
            #Add all of our new values for the MACD to the dataframe
    df_macd['MACD_12_26_9'] = df_macd.index.map(macd)
    df_macd['MACDh_12_26_9'] = df_macd.index.map(macd_h)
    df_macd['MACDs_12_26_9'] = df_macd.index.map(macd_s)
            ###-----------------------------------------------------------------------------------------------------###
    macd = df_macd[['MACD_12_26_9']]
    histogram = df_macd[['MACDh_12_26_9']]
    signal = df_macd[['MACDs_12_26_9']]
            ###-----------------------------------------------------------------------------------------------------###
    macd_color = gen_macd_color(df_macd)
            ###-----------------------------------------------------------------------------------------------------###
    apds1 = [
            mpf.make_addplot(macd,color='#2962FF', width = 1.0, panel=1),
            mpf.make_addplot(signal,color='#FF6D00', width = 2.0, panel=1),
            mpf.make_addplot(histogram,type='bar',width=0.7,panel=1, color=macd_color,alpha=1,secondary_y=True),
                ]
            ###-----------------------------------------------------------------------------------------------------###
    fig1, ax = mpf.plot(
                df_macd,
                #title=f'{selected_ticker} MACD',
                volume=False,
                type='candle', 
                style=chart_style,
                addplot=apds1,
                volume_panel=2,
                panel_ratios=(2, 1.5),
                figsize=(10,5),
                tight_layout=True,
                returnfig=True
                )
    st.pyplot(fig1)
    ###################

    df_dc = stock_df.tail(200).copy()
    st.write("DONCHAIN CHANNEL - Volatility Indicator to identify price trends & optimal entry & exit in ranging markets.")
    period = 10
    df_dc['Upper'] = df_dc['High'].rolling(period).max()
    df_dc['Lower'] = df_dc['Low'].rolling(period).min()
    df_dc['Middle'] = (df_dc['Upper'] + df_dc['Lower']) / 2

    # Data Extracted And New Variable Applied
    DCU = df_dc[['Upper']]
    DCM = df_dc[['Middle']]
    DCL = df_dc[['Lower']]

    apds2 = [
                            mpf.make_addplot(DCU,color='#2962FF',panel=0,),
                            mpf.make_addplot(DCM,color='#FF6D00',panel=0,),
                            mpf.make_addplot(DCL,color='#2962FF',panel=0,),
                        ]
                    
    fig2, ax = mpf.plot(
                            df_dc,
                            volume=False,
                            type="candle",
                            fill_between=dict(y1=df_dc['Upper'].values,y2=df_dc['Lower'].values,alpha=0.1,color='#2962FF'),
                            style=chart_style,
                            addplot=apds2,
                            figsize=(10,3),
                            tight_layout=True,
                            returnfig=True
                            
                        ) 

    st.pyplot(fig2)

    ###--------Multi SMA Chart---------------------------------------------------------------------------------------------###
        #columns_to_copy = ['open', 'high', 'low', 'close', 'volume']
        #rename_dict = {'close': 'Close', 'open': 'Open', 'high': 'High', 'low': 'Low', 'volume': 'Volume'}
        #df_macd.index = pd.to_datetime(df_macd.index)  # Ensure index is datetime
        #df_ma = df_macd[columns_to_copy]
    st.write(f'{selected_ticker}- Multi Period SMA [ 9->Blue, 20->Green ,50->Red, 100-> Purple]')
    df_ma = ta_df.head(170).copy()
    df_ma = df_ma.sort_index(ascending=False)
    if interval == '1d':
            df_ma = df_ma.set_index('Date')
           # df_ma.index = pd.to_datetime(df_ma.index)  # Ensure index is datetime
    else:
            df_ma = df_ma.set_index('Datetime')  # Ensure index is datetime
    
    #st.dataframe(df_ma)

    #df_ma['SMA_200'] = df_ma['close'].rolling(window=200).mean()
    #df_ma.rename(columns=rename_dict, inplace=True)
    #df_ma = df_ma.dropna()
    s9 = df_ma['SMA_9']
    s20 = df_ma['SMA_20']
    s50 = df_ma['SMA_50']
    s100 = df_ma['SMA_100']
    #s200 = df_ma['SMA_200']

        # Generate the color lists for each SMA using the function
        #sma_colors = gen_sma_colors(df_ma)

        # Create addplot objects for each SMA with the corresponding color list
    apds5 = [
            mpf.make_addplot(df_ma['SMA_9'], color='#2053c7', width = 0.8, panel=0),
            mpf.make_addplot(df_ma['SMA_20'], color='#235a56', width = 1.0, panel=0),
            mpf.make_addplot(df_ma['SMA_50'], color='#c6213d', width = 1.4, panel=0),
            mpf.make_addplot(df_ma['SMA_100'], color='#733bac', width = 1.8, panel=0)
            #mpf.make_addplot(df_ma['SMA_200'], color='#9b924c', width = 0.8, panel=0)
            #mpf.make_addplot(df_ma['SMA_500'], color=sma_colors['sma500'], panel=0)
        ]
        
    fig5, ax = mpf.plot(
                        df_ma,
                        #title=f'{selected_ticker} Multi Period SMA with Color Coding',
                        volume=True,
                        type='candle', 
                        style=chart_style,
                        addplot=apds5,
                        volume_panel=1,
                        figsize=(10,4),
                        tight_layout=True,
                        panel_ratios=(10, 3),
                        returnfig=True
                        )
    st.pyplot(fig5)

    ###-----------   Golden cross---------------------------------------------------------------------------###
            ### A golden cross is a chart pattern in which a relatively short-term moving average crosses above a long-term moving average ###
            ### 
    st.subheader("SMA Cross Overs.Choose short/long periods to proceed [G -> BUY, R-> SELL]")
    #st.write(selected_ticker)
    frames = ['<select>',9,21,50,100,200]
    default_long_period = frames.index(21)
    default_short_period = frames.index(9)
    maflag1 = ""
    df = stock_df.tail(200).copy()
    with st.form("select Periods",clear_on_submit=False, border= True):
            short_period = st.selectbox("Select Short Period", frames, index = default_short_period )
            long_period = st.selectbox("Select Long Period", frames, index = default_long_period)
            if int(long_period) < int(short_period):
                        st.write("You have not selected periods in the right oreder. Try again !")
                        st.stop()
                    
            if short_period == 9: maflag1 = "SMA9"
            elif short_period == 21: maflag1 = "SMA21"
            elif short_period == 50: maflag1 = "SMA50"
            elif short_period == 100: maflag1 = "SMA100"
            else : maflag1 = "SMA200"

            maflag2=""
            if long_period == 21: maflag2 = "SMA21"
            elif long_period == 50: maflag2 = "SMA50"
            elif long_period == 100: maflag2 = "SMA100"
            else : maflag2 = "SMA200"

            print(long_period, short_period)
                            
            if (short_period == 9 and long_period == 21):
                                df[maflag1] = df['Close'].rolling(window=9).mean()
                                df[maflag2] = df['Close'].rolling(window=21).mean()
            elif (short_period == 9 and long_period == 50):
                                df[maflag1] = df['Close'].rolling(window=9).mean()
                                df[maflag2] = df['Close'].rolling(window=50).mean()
            elif (short_period == 9 and long_period == 100):
                                df[maflag1] = df['Close'].rolling(window=9).mean()
                                df[maflag2] = df['Close'].rolling(window=100).mean()
            elif (short_period == 9 and long_period == 200):
                                df[maflag1] = df['Close'].rolling(window=9).mean()
                                df[maflag2] = df['Close'].rolling(window=200).mean()
            elif (short_period == 21 and long_period == 50):
                                df[maflag1] = df['Close'].rolling(window=21).mean()
                                df[maflag2] = df['Close'].rolling(window=50).mean()
            elif (short_period == 21 and long_period == 100):
                                df[maflag1] = df['Close'].rolling(window=21).mean()
                                df[maflag2] = df['Close'].rolling(window=100).mean()
            elif (short_period == 21 and long_period == 200):
                                df[maflag1] = df['Close'].rolling(window=21).mean()
                                df[maflag2] = df['Close'].rolling(window=200).mean()        
            elif (short_period == 50 and long_period == 100):
                                df[maflag1] = df['Close'].rolling(window=50).mean()
                                df[maflag2] = df['Close'].rolling(window=100).mean()
            elif (short_period == 50 and long_period == 200):
                                df[maflag1] = df['Close'].rolling(window=50).mean()
                                df[maflag2] = df['Close'].rolling(window=200).mean()        
            else:
                                st.write("Invalid combinations of periods found ! Try Again")
                                st.stop()   

            golden_cal(df)
                            #Fuction Color Applied And Df Generated 
                            #Fuction Color Applied And Df Generated 
            goldencrossover = color(df)
                            # Data Extracted And New Variable Applied
            up_sma100 = goldencrossover[['up']]
            down_sma100 = goldencrossover[['down']]
            up_sma21 = goldencrossover[[maflag1]]
            dco = goldencrossover[['GoldenCrossOver']]
            gco = goldencrossover[['DeathCrossOver']]
                            #st.dataframe(df)
            ic = [
                                    #Golden Crossover
                                    mpf.make_addplot(up_sma100,color = 'green',panel=0,width=1.5),
                                    mpf.make_addplot(down_sma100,color = 'red',panel=0,width=1.5),
                                    mpf.make_addplot(up_sma21,color = 'blue',panel=0,linestyle='dashdot',width=1.2),
                                    mpf.make_addplot(gco,type='scatter',markersize=100,marker='v',color='red',panel=0),
                                    mpf.make_addplot(dco,type='scatter',markersize=100,marker='^',color='green',panel=0),
                                ]
                            
            fig3, ax = mpf.plot(
                                    df,
                                    volume=False,
                                    type="candle", 
                                    style= chart_style,
                                    addplot=ic,
                                    figsize=(10,4),
                                    tight_layout=True,
                                    returnfig=True
                                )
            
            if st.form_submit_button('Submit'):
                    filtered_df = df[df['GoldenCrossOver'].notna() | df['DeathCrossOver'].notna()]
                    st.dataframe(filtered_df)
                    st.pyplot(fig3)

            


   
with col[1]:
     #st.write('*****Data Inferences*****',)
     temp_df = stock_df.tail(300).copy()
     trend_q1, trend_q2, trend_q3, trend_q4, trend_q5, trend_q6= find_trend(temp_df)
     #st.metric("5% quartile Trend is ", value=trend_q1, delta = trend_q1 )
     #st.metric("10% quartile Trend is ", value=trend_q2, delta = trend_q2 )
     #st.metric("25% quartile Trend is ", value=trend_q3, delta = trend_q3 )
     #st.metric("50% quartile Trend is ", value=trend_q4, delta = trend_q4 )
     st.metric("75% quartile Trend is ", value=trend_q5, delta = trend_q5 )
     #st.metric("100% quartile Trend is ", value=trend_q6, delta = trend_q6 )
     st.write('Price & Volume EMA Plot. blue->11ema, purple->21ema')
#st.dataframe(stock_df[['Open','High','Low','Close','Volume']].tail(5),hide_index=False)
    #temp_df = stock_df.copy()
     temp_df = temp_df.drop(columns=['Adj Close'], axis=1)
     temp_df.index = pd.to_datetime(temp_df.index)  # Ensure index is datetime
     #temp_df = temp_df.rename(columns={'Volume':'Close'})
     temp_df['9ema'] =  round((temp_df['Volume'].ewm(span=11, adjust=False, min_periods=11).mean() / 100) , 2)
     temp_df['21ema'] =  round((temp_df['Volume'].ewm(span=21, adjust=False, min_periods=21).mean() / 100), 2)
     #temp_df['50ema'] =  round((temp_df['Volume'].ewm(span=50, adjust=False, min_periods=50).mean()) / 100, 2)
     #temp_df['vol_100ema'] =  round((temp_df['Volume'].ewm(span=100, adjust=False, min_periods=100).mean() / 1000000), 2)
     

     #s21 = temp_df['vol_21ema']
     #s50 = temp_df['vol_50ema']

     apds5 = [
             mpf.make_addplot(temp_df['9ema'], color='blue', width = 1.5, panel=0),
             mpf.make_addplot(temp_df['21ema'], color='purple', width = 2.5, panel=0)
             #mpf.make_addplot(temp_df['50ema'], color='purple', width = 3.0, panel=0)
             
            ]
     
     fig5, ax = mpf.plot(
                        temp_df,
                        #title=f'{selected_ticker} Multi Period SMA with Color Coding',
                        volume=False,
                        type='candle', 
                        style=chart_style,
                        addplot=apds5,
                        volume_panel=1,
                        figsize=(20,12),
                        tight_layout=True,
                        #panel_ratios=(4, 1),
                        returnfig=True
                        )
     st.pyplot(fig5)
     temp_df = temp_df.tail(6).drop(columns=['Open','High','Low'])
     st.write('Last 6 volume rows')
     st.dataframe(temp_df,hide_index=True)

     
     st.header('Breadth Analysis of Market Components!')
     #### Relative Price Data for Indexes  SPY, QQQ, IWM, DIA, TLT, OILK, RINF, VIXY
     index_etfs = ['SPY', 'DIA', 'QQQ', 'TLT', 'GLD', 'SLV', 'OILK', 'RINF', 'VIXY']
     index_df, index_cp, index_trend_df = get_industry_data(index_etfs, '2y')
     # Suppress the time part and display just the date portion
     index_df.index = index_df.index.strftime('%Y-%m-%d')
     index_df = index_df.sort_index().tail(6)
     #print(relative_df.info())
     st.write('Index ETFs- Trends')
     # Apply highlighting to the trend DataFrame
     styled_index_trend_df = index_trend_df.style.map(highlight_trends, subset=['Trend5', 'Trend10', 'Trend20', 'Trend50','Trend100','Trend150'])
     st.dataframe(styled_index_trend_df, hide_index = True)
     st.write('Index ETFs- ALL Data')
     st.dataframe(index_cp)

     #### Relative Price Data for Industry ETF's 
     industry_etfs = ['XLK', 'XLB', 'XLI', 'XLE', 'XLV', 'XLY', 'XLF', 'XLU', 'XLP']
     relative_df, industry_cp, ind_trend_df = get_industry_data(industry_etfs, 'max')
     # Suppress the time part and display just the date portion
     relative_df.index = relative_df.index.strftime('%Y-%m-%d')
     relative_df = relative_df.sort_index().tail(6)
     #print(relative_df.info())
     st.write('Industry- Trends')
     # Apply highlighting to the trend DataFrame
     styled_ind_trend_df = ind_trend_df.style.map(highlight_trends, subset=['Trend5', 'Trend10', 'Trend20', 'Trend50','Trend100','Trend150'])
     st.dataframe(styled_ind_trend_df, hide_index = True)
     st.write('Relative Price Movement of Industry ETFs starting from the Start Date in the sidebar ! Displaying latest 6 entries.')
     st.dataframe(relative_df)
     
     #### Relative Price Data for Crypto
     crypto_list = ['BTC-USD', 'MSTR', 'BITX', 'BITO', 'ARKB', 'MARA', 'YBTC']
     crypto_df, crypto_cp, dummy = get_industry_data(crypto_list, 'max')  
     crypto_cp['BTC_BITO'] = round(((crypto_cp['BTC-USD']) / 1000) / crypto_cp['BITO'], 2)
        # Suppress the time part and display just the date portion
     crypto_df.index = crypto_df.index.strftime('%Y-%m-%d')
     crypto_df = crypto_df.sort_index().tail(6)
        #print(relative_df.info())
     st.write('Relative Price Movement of Crypto Tickers ! Displaying latest 6 entries.')
     st.dataframe(crypto_df)
     
     crypto_cp.index = crypto_cp.index.strftime('%Y-%m-%d')
     crypto_cp = crypto_cp.sort_index()
     st.write('Crypto Prices')
     st.dataframe(crypto_cp.tail(6))

     with st.expander('About', expanded=True):
            st.write('''
                - Data: [yfinance](https://finance.yahoo.com/)
                - :orange[**Summary**]: Intarday analysis data & tools
                - :orange[**Developer**]: mayank.dangwal2019@gmail.com
                - :orange[**Future**]: Machine learning specific features
                - :blue[**Version**]: [git](https://github.com/dangwam/stockdashboardapp.git)
                    
                ''')
