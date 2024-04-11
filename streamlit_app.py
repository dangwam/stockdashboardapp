import streamlit as st
import altair as alt  # Import altair for themes
import pandas as pd
import yfinance as yf
from finta import TA
import mplfinance as mpf
import yahooquery as yq
from pandas_datareader import data as pdr
from plotly import express as px
import datetime
from matplotlib import pyplot as plt
from plotly import graph_objects as go
yf.pdr_override()

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

alt.themes.enable("dark")

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

st.title("Equity Analysis App :chart_with_upwards_trend:")
st.markdown(title_format, unsafe_allow_html=True)
#st.write("---------------------------------------------------------------------------------------------------------------------------------------------------------")
# Load data -- data functions
@st.cache_data
def get_sp500_components():
    df = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")
    df = df[0]
    tickers = df["Symbol"].to_list()
    extended_symbols = ['RIVN', 'AVGO', 'SPY', 'QQQ', 'TSLA', 'MA']
    extended_companies = ['Rivian Automotive', 'Broadcom Inc', 'SPDR S&P 500 ETF', 'Invesco QQQ Trust', 'Tesla', 'Mastercard']
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
  tk = Ticker(symbol)
  sp_dict = tk.asset_profile
  overall_risk = sp_dict[symbol]['overallRisk']
  num_employees = sp_dict[symbol]['fullTimeEmployees']
  longBusinessSummary = sp_dict[symbol]['longBusinessSummary']
  industry = sp_dict[symbol]['industry']
  sector = sp_dict[symbol]['sector']
  website = sp_dict[symbol]['website']
  return industry, sector, longBusinessSummary, num_employees, website ,pd.DataFrame(tk.fund_ownership)


@st.cache_data
def convert_df_to_csv(df):
    return df.to_csv().encode("utf-8")


@st.cache_data
def get_company_info(ticker):
  symbol = ticker
  tk = yq.Ticker(symbol)
  sp_dict = tk.asset_profile
  overall_risk = sp_dict[symbol]['overallRisk']
  num_employees = sp_dict[symbol]['fullTimeEmployees']
  longBusinessSummary = sp_dict[symbol]['longBusinessSummary']
  industry = sp_dict[symbol]['industry']
  sector = sp_dict[symbol]['sector']
  website = sp_dict[symbol]['website']
  return industry, sector, longBusinessSummary, num_employees, website ,pd.DataFrame(tk.fund_ownership)


@st.cache_data
def yquery_technical_insights(symbol):
    tk = yq.Ticker(symbol)
    technical_insights = tk.technical_insights
    return technical_insights[symbol]


@st.cache_data
def get_historical_data(symbol, start_date, end_date):

    df = pdr.get_data_yahoo(symbol, start=start_date, end=end_date)
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

def get_dividend_yield(symbol):
    try:
        # Download the stock data
        stock = yf.Ticker(symbol)
        
        # Get dividend yield
        dividend_yield = stock.info.get('dividendYield')
        
        if dividend_yield is not None:
            # Format dividend yield to display correct number of decimal places
            formatted_yield = "{:.2%}".format(dividend_yield)
            return formatted_yield
        else:
            return "NIL"
    except Exception as e:
        return f"Error occurred: {str(e)}"
# Sidebar for user input
#######################
st.markdown('---')

st.sidebar.subheader('Settings')
st.sidebar.caption(":chart_with_upwards_trend: Stock Analysis")

with st.sidebar:
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
      
     expander_title = f"Key Stats for {selected_ticker}"
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
         print(stock_df.head(4))
         print(stock_df.columns)
         fig = px.line(stock_df, x=stock_df.index, y=stock_df.Close, title = f"Closing Prices vs Benchmark", template= 'simple_white')
     
     else:
         stock_df = load_data(['SPY', selected_ticker], start=start_date, end=end_date)['Close']
         fig = px.line(stock_df, x=stock_df.index, y=stock_df.columns,title = f"Closing Prices vs Benchmark", template= 'simple_white' )
#fig.update_traces(line_color = 'purple')
     st.sidebar.plotly_chart(fig,use_container_width=True)
     #color_theme_list = ['blues', 'cividis', 'greens', 'inferno', 'magma', 'plasma', 'reds', 'rainbow', 'turbo', 'viridis']
     #selected_color_theme = st.selectbox('Select a color theme', color_theme_list)
     # Fetch company information from asset profile
     if selected_ticker != 'SPY':
         industry, sector, longBusinessSummary, num_employees, website, fund_ownership = get_company_info(selected_ticker)
         st.sidebar.write("Key Information")
     else:
         longBusinessSummary = "Selecteed ASSET is an ETF and does not have a summary information at this time"

     color_theme_list = ['blues', 'cividis', 'greens', 'inferno', 'magma', 'plasma', 'reds', 'rainbow', 'turbo', 'viridis']
     selected_color_theme = st.selectbox('Select a color theme', color_theme_list)    
     
expander_title = f"Business Summary of {selected_ticker}"
with st.sidebar.expander(expander_title):
     st.write(longBusinessSummary)
     st.markdown(hide, unsafe_allow_html=True)


expander_title = f"FundOwnership {selected_ticker}"
with st.sidebar.expander(expander_title):
     if selected_ticker != 'SPY':
         fund_ownership.reset_index(inplace=True)
         fund_ownership.drop(fund_ownership.columns[[0, 1]], axis=1, inplace=True)
         st.dataframe(fund_ownership.iloc[:5, 2:])
         st.markdown(hide, unsafe_allow_html=True)
     else:
         st.write(longBusinessSummary)
         st.markdown(hide, unsafe_allow_html=True)


#######################
# Dashboard Main Panel
#st.write("Summary")
st.markdown(hide, unsafe_allow_html=True)

col = st.columns((2, 4.5, 2), gap='small')
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
    temp_df = temp_df.head(5)
    #st.dataframe(temp_df.style.format("{:.2%}").highlight_max(color='blue'),hide_index=True)
    st.dataframe(temp_df.style.format({'1 Month': '{:.2f}', '3 Months': '{:.2f}','6 Months': '{:.2f}', '1 Year': '{:.2f}'}).highlight_max(color='blue'),hide_index=True)
    st.markdown('#### Valuation Measures')
    
with col[1]:
    chart_type = 'renko'
    chart_style = 'starsandstripes'
    ticker = selected_ticker
    period = "1d"
    interval = "15m"
    st.markdown(f"Technical Analysis of {selected_ticker}")
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
    # Create a Matplotlib figure object
    fig, ax = mpf.plot(data,
                       mav=(int(20),int(50),int(100)),
                       title=f'{selected_ticker}-> SMA(20/50/100)',
                       volume=True,
                       style=chart_style,
                       type=chart_type,
                       panel_ratios=(4,1),
                       tight_layout=True,
                       returnfig=True
                       )
    
    st.pyplot(fig)
    
    


with col[2]:
     
     st.markdown("TBD")
     




    
    

        







   
