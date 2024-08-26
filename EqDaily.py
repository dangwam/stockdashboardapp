import streamlit as st
import altair as alt  # Import altair for themes
import pandas as pd
import yfinance as yf
from finta import TA
import numpy as np
import mplfinance as mpf
import yahooquery as yq
from pandas_datareader import data as pdr
from plotly import express as px
import plotly.graph_objs as go
import datetime
from matplotlib import pyplot as plt
from plotly import graph_objects as go
#yf.pdr_override()

#############################################
#####API_KEY 
#############################################
# Streamlit App Configuration (Best Practice)
hide = """
            <style>
            ul.streamlit-expander {
                overflow: scroll;
            } 
           
            </style>
            """

title_format = """
                    <style> 
                        div.block-container{
                            padding-top:0rem;
                            padding-bottom:0rem;
                            }
                    </style>
               """


#######################
# Page configuration
st.set_page_config(
    page_title="Stock Dashboard",
    page_icon=":chart_with_upwards_trend:",
    layout="wide",
    initial_sidebar_state="expanded")

#alt.themes.enable("dark")

#######################
# CSS styling
st.markdown("""
<style>

[data-testid="block-container"] {
    padding-left: 2rem;
    padding-right: 2rem;
    padding-top: 1rem;
    padding-bottom: 0rem;
    margin-bottom: -7rem;
}

[data-testid="stVerticalBlock"] {
    padding-left: 0rem;
    padding-right: 0rem;
}

[data-testid="stMetric"] {
    background-color: #393939;
    text-align: center;
    padding: 15px 0;
}

[data-testid="stMetricLabel"] {
  display: flex;
  justify-content: center;
  align-items: center;
}

[data-testid="stMetricDeltaIcon-Up"] {
    position: relative;
    left: 38%;
    -webkit-transform: translateX(-50%);
    -ms-transform: translateX(-50%);
    transform: translateX(-50%);
}

[data-testid="stMetricDeltaIcon-Down"] {
    position: relative;
    left: 38%;
    -webkit-transform: translateX(-50%);
    -ms-transform: translateX(-50%);
    transform: translateX(-50%);
}

</style>
""", unsafe_allow_html=True)


#######################
print("------------------------------------------------------------Starting-----------------------------------------------------------------------------")
st.title("Equity Analysis App :chart_with_upwards_trend:")
st.markdown(title_format, unsafe_allow_html=True)
#st.write("---------------------------------------------------------------------------------------------------------------------------------------------------------")
# Load data -- data functions
@st.cache_data
def get_sp500_components():
    df = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")
    df = df[0]
    tickers = df["Symbol"].to_list()
    extended_symbols = ['RIVN', 'AVGO', 'SPY', 'QQQ', 'TSLA', 'MA', 'PLTR', 'SOFI', 'PFE', 'WBA']
    extended_companies = ['Rivian Automotive', 'Broadcom Inc', 'SPDR S&P 500 ETF', 'Invesco QQQ Trust', 'Tesla', 'Mastercard', 'Palantir', 'Sofi', 'Pfizer', 'WallGreens']
    # Combine tickers with extended symbols
    tickers.extend(extended_symbols)
    ##tickers_companies_dict = dict(zip(df["Symbol"], df["Security"]))
    tickers_companies_dict = dict(zip(df["Symbol"], df["Security"]), **dict(zip(extended_symbols, extended_companies)))
    return tickers, tickers_companies_dict

@st.cache_data
def load_data(symbol, start, end):
    #start_date = datetime.date(year=min(2010,year_now-10), month=1, day=1) # Take first day and month of 10 years ago / or take first day of 2010
    #time_start = datetime.date(year=2014, month=1, day=1)
    #end_date = datetime.date(year=year_now, month=month_now, day=1) # Take first day of month of today's date
    
    return yf.download(symbol, start, end)

@st.cache_data
def get_company_info(ticker):
  symbol = ticker
  tk = yq.Ticker(symbol)
  sp_dict = tk.asset_profile
  #overall_risk = sp_dict[symbol]['overallRisk']
  #num_employees = sp_dict[symbol]['fullTimeEmployees']
  longBusinessSummary = sp_dict[symbol]['longBusinessSummary']
  #industry = sp_dict[symbol]['industry']
  #sector = sp_dict[symbol]['sector']
  #website = sp_dict[symbol]['website']
  return longBusinessSummary,pd.DataFrame(tk.fund_ownership)


@st.cache_data
def convert_df_to_csv(df):
    return df.to_csv().encode("utf-8")


@st.cache_data
def get_company_info(ticker):
  symbol = ticker
  tk = yq.Ticker(symbol)
  sp_dict = tk.asset_profile
  #overall_risk = sp_dict[symbol]['overallRisk']
  #num_employees = sp_dict[symbol]['fullTimeEmployees']
  longBusinessSummary = sp_dict[symbol]['longBusinessSummary']
  #industry = sp_dict[symbol]['industry']
  #sector = sp_dict[symbol]['sector']
  #website = sp_dict[symbol]['website']
  return longBusinessSummary,pd.DataFrame(tk.fund_ownership)


@st.cache_data
def yquery_technical_insights(symbol):
    tk = yq.Ticker(symbol)
    technical_insights = tk.technical_insights
    return technical_insights[symbol]


@st.cache_data
def get_historical_data(symbol, start_date, end_date):

    #df = pdr.get_data_yahoo(symbol, start=start_date, end=end_date)
    df = yf.download(symbol, interval= '1d')
    df = df.rename(columns = {'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close', 'Adj Close': 'adj close', 'Volume': 'volume'})
    for i in df.columns:
        df[i] = df[i].astype(float)
    df.index = pd.to_datetime(df.index)
    if start_date:
        df = df[df.index >= pd.to_datetime(start_date)]
    return df
    
# Convert number to text 
def format_number(num):
    if num > 1000000:
        if not num % 1000000:
            return f'{num // 1000000} M'
        return f'{round(num / 1000000, 1)} M'
    return f'{num // 1000} K'


def get_stock_type(symbol):
    stock = yf.Ticker(symbol)
    stock_type = stock.info.get('quoteType')
    return stock_type

@st.cache_data
def get_stock_info(symbol, stock_type):
    try:
        # Download the stock data
        stock = yf.Ticker(symbol)
        if stock_type == 'EQUITY':
            
            forwardPE = stock.info.get('forwardPE')
            
            priceToSalesTrailing12Months = stock.info.get('priceToSalesTrailing12Months')
            
            enterpriseValue = stock.info.get('enterpriseValue')
            
            profitMargins =  stock.info.get('profitMargins')

            dividend_yield = stock.info.get('dividendYield')

            if forwardPE is None: forwardPE = 0
            if priceToSalesTrailing12Months is None: priceToSalesTrailing12Months = 0
            if enterpriseValue is None: enterpriseValue = 0
            if profitMargins is None: profitMargins = 0
            if dividend_yield is None : dividend_yield = 0
            
            return forwardPE,priceToSalesTrailing12Months,enterpriseValue,profitMargins,dividend_yield
        elif stock_type == 'ETF':
            trailingPE = stock.info.get('trailingPE')
            navPrice = stock.info.get('navPrice')
            threeYearAverageReturn = stock.info.get('threeYearAverageReturn')
            fiveYearAverageReturn = stock.info.get('fiveYearAverageReturn')
            fiftyDayAverage = stock.info.get('fiftyDayAverage')

            if trailingPE is None: trailingPE = 0
            if navPrice is None: navPrice = 0
            if threeYearAverageReturn is None: threeYearAverageReturn = 0
            if fiveYearAverageReturn is None: fiveYearAverageReturn = 0
            if fiftyDayAverage is None: fiftyDayAverage = 0

            return trailingPE,navPrice,threeYearAverageReturn,fiveYearAverageReturn,fiftyDayAverage
        
    except Exception as e:
        return f"Stock Information is not available at this time !!"


def fetch_financials(symbol):
    try:
        stock_obj = yf.Ticker(selected_ticker)
        quarterly_financials = stock_obj.quarterly_financials.T
        # Get the quarterly cash flow data for the last 8 quarters
        quarterly_cashflow = stock_obj.quarterly_cashflow.T
        # Get the quarterly balance sheet data for the last 8 quarters
        quarterly_balancesheet = stock_obj.quarterly_balance_sheet.T
        # columns of interest
        income_cols = ["EBIT", "Total Expenses", "Basic EPS", "Diluted Average Shares"]
        cash_cols = ["Free Cash Flow", "Capital Expenditure", "Changes In Cash"]
        balance_sheet_cols = ["Common Stock Equity", "Total Debt","Net Tangible Assets", "Total Capitalization"]

       # Filter data based on columns of interest and format numbers
        if set(income_cols).issubset(quarterly_financials.columns):
            quarterly_financials = quarterly_financials[income_cols].map(format_number)
        else:
            print("Some columns in 'income_cols' are not found in the quarterly financials data.")
        
        if set(cash_cols).issubset(quarterly_cashflow.columns):
            quarterly_cashflow = quarterly_cashflow[cash_cols].map(format_number)
        else:
            print("Some columns in 'cash_cols' are not found in the quarterly cash flow data.")
        
        if set(balance_sheet_cols).issubset(quarterly_balancesheet.columns):
            quarterly_balancesheet = quarterly_balancesheet[balance_sheet_cols].map(format_number)
        else:
            print("Some columns in 'balance_sheet_cols' are not found in the quarterly balance sheet data.")
        
        # Format index
        quarterly_financials.index = quarterly_financials.index.date
        quarterly_cashflow.index = quarterly_cashflow.index.date
        quarterly_balancesheet.index = quarterly_balancesheet.index.date
        
        # Rename columns
        income_cols_formatted = ["Revenue", "Expenses", "EPS", "AvgShares"]
        cash_cols_formatted = ["FreeCashFlow", "Capex", "CashBurn"]
        balance_sheet_cols_formatted = ["Equity", "Debt","Assets", "Capital"]
        
        quarterly_financials.columns = income_cols_formatted
        quarterly_cashflow.columns = cash_cols_formatted
        quarterly_balancesheet.columns = balance_sheet_cols_formatted
        
        return quarterly_financials, quarterly_cashflow, quarterly_balancesheet
    
    except Exception as e:
        print("An error occurred:", e)
        return None, None, None

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



# Sidebar for user input
#######################
#st.markdown('---')

#st.sidebar.subheader('Settings')
#st.sidebar.caption(":chart_with_upwards_trend: Stock Analysis")

with st.sidebar.form('inputs'):
     available_tickers, tickers_companies_dict = get_sp500_components()
     selected_ticker = st.sidebar.selectbox("Select Ticker", available_tickers, format_func=tickers_companies_dict.get)
     start_date = st.sidebar.date_input("Start date", datetime.date(2010, 1, 1))
     end_date = st.sidebar.date_input("End date", datetime.date.today())
     df = load_data(selected_ticker, start_date, end_date)['Close'].reset_index()
     # Find the maximum date in the DataFrame
     df['Date'] = pd.to_datetime(df['Date'])
     max_date = pd.to_datetime(df['Date'].max())
     one_year_ago = max_date - pd.DateOffset(years=1)
     last_year_data = df[df['Date'] >= one_year_ago]
     x = round(last_year_data['Close'].describe(), 2)
     temp_str=""
     temp_index=""
     out_str=""
     # Iterate over the Series using items()
     for index, value in x.items():
        if index not in ['count']:
            temp_index = index + " is "
            temp_str = temp_index + str(value)

        out_str = out_str + " --- " + temp_str
      
     expander_title = f"{selected_ticker} Stats."
     with st.sidebar.expander(expander_title):
            #st.write(out_str)
            #st.markdown(hide, unsafe_allow_html=True)
            st.markdown(out_str, unsafe_allow_html=True)

     
     if start_date > end_date:
        st.sidebar.error("The end date must fall after the start date")

     #stock_df = yf.download(['SPY', selected_ticker], period='1d', start=start_date, end=end_date)['Close']
     print(f"selected_ticker is {selected_ticker}")
     if selected_ticker == 'SPY':
         stock_df = load_data(['SPY'], start=start_date, end=end_date)
         print(stock_df.columns)
         fig = px.line(stock_df, x=stock_df.index, y=stock_df.Close, title = f"Closing Prices vs Benchmark", template= 'simple_white')
     
     else:
         stock_df = load_data(['SPY', selected_ticker], start=start_date, end=end_date)['Close']
         fig = px.line(stock_df, x=stock_df.index, y=stock_df.columns,title = f"Closing Prices vs Benchmark", template= 'simple_white' )

     st.sidebar.plotly_chart(fig,use_container_width=True)
     # Fetch company information from asset profile
     if selected_ticker not in ['SPY', 'QQQ' ]:
         longBusinessSummary,fund_ownership = get_company_info(selected_ticker)
         st.sidebar.write("Key Information")
     else:
         longBusinessSummary = "Selecteed ASSET is an ETF and does not have a summary information at this time"

     expander_title = f"Business Summary"
     with st.sidebar.expander(expander_title):
         st.write(longBusinessSummary)
         st.markdown(hide, unsafe_allow_html=True)


     expander_title = f"FundOwnership"
     with st.sidebar.expander(expander_title):
        if selected_ticker not in ['SPY', 'QQQ']:
            fund_ownership.reset_index(inplace=True)
            fund_ownership.drop(fund_ownership.columns[[0, 1]], axis=1, inplace=True)
            st.dataframe(fund_ownership.iloc[:5, 2:])
            st.markdown(hide, unsafe_allow_html=True)
        else:
            st.write(longBusinessSummary)
            st.markdown(hide, unsafe_allow_html=True)

     with st.sidebar:
        chart_styles = [
            'default', 'binance', 'blueskies', 'brasil', 
            'charles', 'checkers', 'classic', 'yahoo',
            'mike', 'nightclouds', 'sas', 'starsandstripes'
        ]
        
        chart_style = st.selectbox('Chart style', options=chart_styles, index=chart_styles.index('starsandstripes'))
        
        chart_types = [
            'candle', 'ohlc', 'line', 'renko', 'pnf'
        ]
        #chart_type = st.selectbox('Chart type', options=chart_types, index=chart_types.index('candle'))

     st.form_submit_button('Refresh')

#######################
# Dashboard Main Panel
#st.write("Summary")
st.markdown(hide, unsafe_allow_html=True)

col = st.columns((1, 3, 1.5), gap='small')
with col[0]:
    st.markdown('#### Cumulative returns')
    monthly_prices = stock_df.resample('M').last()
    outlier_cutoff = 0.01
    data = pd.DataFrame()
    lags = [1, 2, 3, 6, 9, 12]
    for lag in lags:
        data[f'return_{lag}m'] = (monthly_prices
                            .pct_change(lag)
                            .stack()
                            .pipe(lambda x: x.clip(lower=x.quantile(outlier_cutoff),
                                                    upper=x.quantile(1-outlier_cutoff)))
                            .add(1)
                            .pow(1/lag)
                            .sub(1)
                            )

    data = data.reset_index().rename(columns={'level_1': 'Ticker'}).dropna().sort_values(by=['Date','Ticker'],ascending = False)
    temp_df = pd.DataFrame()
    temp_df['Asset'] = data['Ticker']
    temp_df['1 Month'] = data['return_1m']
    temp_df['3 Months'] = data['return_3m']
    temp_df['6 Months'] = data['return_6m']
    temp_df['1 Year'] = data['return_12m']
    temp_df = temp_df.head(2)
    st.dataframe(temp_df.style.format({'1 Month': '{:.2f}', '3 Months': '{:.2f}','6 Months': '{:.2f}', '1 Year': '{:.2f}'}).highlight_max(color='blue'),hide_index=True)
    
    security = get_stock_type(selected_ticker)
    print(security)
    if security == 'EQUITY':
        forwardPE,priceToSalesTrailing12Months,enterpriseValue,profitMargins,dividend_yield = get_stock_info(selected_ticker,security)
        st.metric(label="Dividend Yeild", value='{:.2%}'.format(dividend_yield))
        st.metric(label="Forward PE", value='{:.2f}'.format(forwardPE))
        st.metric(label="PriceToSales", value='{:.2f}'.format(priceToSalesTrailing12Months))
        st.metric(label="EValue", value=format_number(enterpriseValue))
        st.metric(label="ProfitMargins", value='{:.2%}'.format(profitMargins))
    elif security == 'ETF':
        trailingPE,navPrice,threeYearAverageReturn,fiveYearAverageReturn,fiftyDayAverage = get_stock_info(selected_ticker, security)
        st.metric(label="TrailingPE", value=trailingPE)
        st.metric(label="NAV", value=navPrice)
        st.metric(label="3Year AvgReturn", value=threeYearAverageReturn)
        st.metric(label="5Year AvgReturn", value=fiveYearAverageReturn)
        st.metric(label="50 day AvgPrice", value=fiftyDayAverage)
        
    else:
        st.metric(label="Stock Info is ", value= "N/A at this time")

    #################################################################
    
    # Get the quarterly financial data for the last 8 quarters
    if security != 'ETF':
        quarterly_financials, quarterly_cashflow, quarterly_balancesheet = fetch_financials(selected_ticker)
        if quarterly_financials is not None:
            st.caption('#### Quarterly Income')
            st.dataframe(quarterly_financials)
        
        if quarterly_cashflow is not None:
            st.caption('#### Quarterly CashFlow')
            st.dataframe(quarterly_cashflow)

        if quarterly_balancesheet is not None:
            st.caption('#### Quarterly BalSheet')
            st.dataframe(quarterly_balancesheet)


    
with col[1]:
        ticker = selected_ticker
        period = "1d"
        interval = "15m"
        #st.markdown(f"Technical Analysis of {selected_ticker}")
        technical_insights = yquery_technical_insights(selected_ticker)
        short_term_outlook = technical_insights['instrumentInfo']['technicalEvents']['shortTermOutlook']
        st_state_description = short_term_outlook['stateDescription']
        st_outlook_direction = short_term_outlook['direction']
        st.write(f"Short term Outlook :- {st_state_description}....Direction is {st_outlook_direction}")
        #data = yf.download(tickers = ticker, start= start_date , end=end_date)
        data = get_historical_data(selected_ticker, start_date, end_date)
        #data.columns = ['close', 'volume', 'open', 'high', 'low']
        ohlc = data.sort_values(by=['Date'],ascending=True)
        ta_df = pd.DataFrame()
        ta_df['close'] = ohlc['close']
        ta_df['high'] = ohlc['high']
        ta_df['low'] = ohlc['low']
        ta_df['SMA_9'] = round(TA.SMA(ohlc,9),2)
        ta_df['SMA_20'] = round(TA.SMA(ohlc,20),2)
        ta_df['SMA_50'] = round(TA.SMA(ohlc,50),2)
        ta_df['SMA_100'] = round(TA.SMA(ohlc,100),2)
        ta_df['MACD'] =  round(TA.MACD(ohlc),2)['MACD']
        ta_df['SAR'] = round(TA.SAR(ohlc),2)
        ta_df['BB_Upper'] = round(TA.MOBO(ohlc),2)['BB_UPPER']
        #ta_df['BB_Middle'] = round(TA.MOBO(ohlc),2)['BB_MIDDLE']
        ta_df['BB_Lower'] = round(TA.MOBO(ohlc),2)['BB_LOWER']
        ta_df = ta_df.tail(5)
        ta_df = ta_df.reset_index()
        ta_df["Date"] = pd.to_datetime(ta_df["Date"])
        ta_df["Date"] = ta_df["Date"].dt.strftime("%Y-%m-%d")
        st.dataframe(ta_df, hide_index=True)
        ###### Next Plot the charts
        ###--------MACD Chart---------------------------------------------------------------------------------------------###
        df_macd = data.tail(200).copy()
        #Get the 26-day EMA of the closing price
        k = df_macd['close'].ewm(span=12, adjust=False, min_periods=12).mean()
        #Get the 12-day EMA of the closing price
        d = df_macd['close'].ewm(span=26, adjust=False, min_periods=26).mean()
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
            mpf.make_addplot(macd,color='#2962FF', panel=1),
            mpf.make_addplot(signal,color='#FF6D00', panel=1),
            mpf.make_addplot(histogram,type='bar',width=0.7,panel=1, color=macd_color,alpha=1,secondary_y=True),
                ]
        ###-----------------------------------------------------------------------------------------------------###
        fig1, ax = mpf.plot(
                df_macd,
                title=f'{selected_ticker} MACD',
                volume=False,
                type='line', 
                style=chart_style,
                addplot=apds1,
                volume_panel=2,
                figsize=(20,10),
                tight_layout=True,
                returnfig=True
                )
        st.pyplot(fig1)

        ###--------MACD Chart---------------------------------------------------------------------------------------------###
        ###-----------   Golden cross---------------------------------------------------------------------------###
        ### A golden cross is a chart pattern in which a relatively short-term moving average crosses above a long-term moving average ###
        ### 
        st.write("A golden cross is a chart pattern in which a relatively short-term moving average crosses above a long-term moving average. Choose short and long periods to proceed")
        frames = [9,21,50,100,200]
        maflag1 = ""
        df = data.tail(200).copy()
        with st.form("aform"):
            short_period = st.selectbox("Select Short Period", frames)
            long_period = st.selectbox("Select Long Period", frames)
            submitted = st.form_submit_button("Submit", type='primary')
            if int(long_period) < int(short_period):
                st.write("You have not selected periods in the right oreder. Try again !")
                st.stop()
            
            if submitted:
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
                    df[maflag1] = df['close'].rolling(window=9).mean()
                    df[maflag2] = df['close'].rolling(window=21).mean()
                elif (short_period == 9 and long_period == 50):
                    df[maflag1] = df['close'].rolling(window=9).mean()
                    df[maflag2] = df['close'].rolling(window=50).mean()
                elif (short_period == 9 and long_period == 100):
                    df[maflag1] = df['close'].rolling(window=9).mean()
                    df[maflag2] = df['close'].rolling(window=100).mean()
                elif (short_period == 9 and long_period == 200):
                    df[maflag1] = df['close'].rolling(window=9).mean()
                    df[maflag2] = df['close'].rolling(window=200).mean()
                elif (short_period == 21 and long_period == 50):
                    df[maflag1] = df['close'].rolling(window=21).mean()
                    df[maflag2] = df['close'].rolling(window=50).mean()
                elif (short_period == 21 and long_period == 100):
                    df[maflag1] = df['close'].rolling(window=21).mean()
                    df[maflag2] = df['close'].rolling(window=100).mean()
                elif (short_period == 21 and long_period == 200):
                    df[maflag1] = df['close'].rolling(window=21).mean()
                    df[maflag2] = df['close'].rolling(window=200).mean()        
                elif (short_period == 50 and long_period == 100):
                    df[maflag1] = df['close'].rolling(window=50).mean()
                    df[maflag2] = df['close'].rolling(window=100).mean()
                elif (short_period == 50 and long_period == 200):
                    df[maflag1] = df['close'].rolling(window=50).mean()
                    df[maflag2] = df['close'].rolling(window=200).mean()        
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
                        mpf.make_addplot(up_sma100,color = 'green',panel=0,),
                        mpf.make_addplot(down_sma100,color = '#FF8849',panel=0,),
                        mpf.make_addplot(up_sma21,color = '#0496ff',panel=0,linestyle='dashdot'),
                        mpf.make_addplot(gco,type='scatter',markersize=200,marker='v',color='red',panel=0),
                        mpf.make_addplot(dco,type='scatter',markersize=200,marker='^',color='green',panel=0),
                    ]
                
                fig3, ax = mpf.plot(
                        df,
                        volume=True,
                        type="candle", 
                        style= chart_style,
                        addplot=ic,
                        figsize=(25,15),
                        tight_layout=True,
                        returnfig=True
                    )
                st.pyplot(fig3)

with col[2]:
    #For Calcultation Dochian Channel
        df_dc = data.tail(100).copy()
        st.write("DONCHAIN CHANNEL PLot:-Donchian Channel is a volatility indicator that helps technical analysts to identify and define price trends as well as determine the optimal entry and exit points in ranging markets.")
        period = 10
        df_dc['Upper'] = df_dc['high'].rolling(period).max()
        df_dc['Lower'] = df_dc['low'].rolling(period).min()
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
                figsize=(20,10),
                tight_layout=True,
                returnfig=True
                
            ) 

        st.pyplot(fig2)

    
    
        #################################
        third_col = st.columns((2.75,1))
        with third_col[1]:
            st.write('FibLevels')
            spy_data = data.copy()
            # Find the start and end dates
            start_date = spy_data.index[0]
            end_date = spy_data.index[-1]
            # Find the close price for the start and end dates
            start_close = spy_data.loc[start_date, 'close']
            end_close = spy_data.loc[end_date, 'close']
            # Calculate Fibonacci levels
            # We'll consider Fibonacci retracement levels from the top (end_close) to the bottom (start_close)
            fib_levels = [0, 23.6, 38.2, 50, 61.8, 100]  # Fibonacci retracement levels in percentage
            fib_prices = {}
            for level in fib_levels:
                fib_price = end_close - (level / 100) * (end_close - start_close)
                fib_prices[level] = fib_price

            # Convert the result to a DataFrame
            fib_df = pd.DataFrame.from_dict(fib_prices, orient='index', columns=['Price'])
            fib_df.index.name = 'Fib Levels'
            fib_df = fib_df.sort_index(ascending=False)
            #fib_df=fib_df.reset_index()
            st.dataframe(fib_df,
                        column_order=("Level", "Price"),
                        hide_index=True,
                        width=None,
                        column_config={
                            "FibLevel": st.column_config.NumberColumn(
                                "Fib",
                                format ="%d",
                                width="small"
                            ),
                            "Price": st.column_config.NumberColumn(
                                "Price",
                                format="$ %d",
                                width="small"
                            )}
                        )
        
        ##############################
            with third_col[0]:
                st.write('Volume Profile')
                data.index = data.index.date
                data=data.sort_index(ascending=False).head(6)
                vol_df = pd.DataFrame()
                vol_df['volume'] = data['volume']
                vol_df = vol_df.reset_index()
                #st.markdown('##### Volume Profile')
                st.dataframe(vol_df,
                            column_order=("index", "volume"),
                            hide_index=True,
                            width=None,
                            column_config={
                                "index": st.column_config.DateColumn(
                                    "Date",
                                    format ="MMMM Do",
                                    width="small"
                                ),
                                "volume": st.column_config.ProgressColumn(
                                    "volume",
                                    format="%f",
                                    min_value=0,
                                    max_value=max(vol_df.volume)
                                )}
                            )
                
        with st.expander('About', expanded=True):
            st.write('''
                - Data: [yfinance](https://finance.yahoo.com/)
                - :orange[**Summary**]: power of technical & financial analysis in single app
                - :orange[**Developer**]: mayank.dangwal2019@gmail.com
                - :orange[**Future**]: alphatrend chart & other powerfull indicators,additional fundamental analysis features.)
                - :blue[**Version**]: [git](https://github.com/dangwam/stockdashboardapp.git)
                    
                ''')
    
   







   
