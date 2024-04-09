import streamlit as st
import altair as alt  # Import altair for themes
import pandas as pd
import yfinance as yf
from financetoolkit import Toolkit
from yahooquery import Ticker
from yahooquery import Screener
import pytz
import os, warnings
import plotly.express as px
import datetime
warnings.filterwarnings('ignore')

s = Screener()

#############################################
#####API_KEY = "DZvlZlf0Q0sDAwwwh59Z90CVB05FjQlS"
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
                            padding-top:1rem;
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

#st.title("Equity Analysis App :chart_with_upwards_trend:")
#st.markdown(title_format, unsafe_allow_html=True)
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
def yquery_summary_detail(symbol):
    tk = Ticker(symbol)
    summary_detail = tk.summary_detail
    return summary_detail[symbol]

@st.cache_data
def yquery_technical_insights(symbol):
    tk = Ticker(symbol)
    technical_insights = tk.technical_insights
    return technical_insights[symbol]


@st.cache_data
def yquery_valuation_measures(symbol):
    tk = Ticker(symbol)
    types = ['periodType','EnterprisesValueEBITDARatio', 'EnterprisesValueRevenueRatio', 'PeRatio', 'PbRatio','PegRatio','PsRatio']
    valuation_measures_df = tk.valuation_measures[types]
    valuation_measures_df = valuation_measures_df[valuation_measures_df["periodType"] == "3M"]
    return valuation_measures_df
                              
@st.cache_data
def convert_df_to_csv(df):
    return df.to_csv().encode("utf-8")

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
  # This function will eventually fetch company information from an API based on ticker
  # Replace this with your actual API call logic
  # Sample company information (replace with actual data)
  return industry, sector, longBusinessSummary, num_employees, website ,pd.DataFrame(tk.fund_ownership)
# Available years (last 10 years)

@st.cache_data
def fetch_all_ratios(ticker,start_date):
     companies = Toolkit([ticker], api_key=API_KEY, start_date=start_date)
     all_ratios = companies.ratios.collect_all_ratios()
     return all_ratios

#all_ratios = companies.ratios.collect_all_ratios()

# Convert number to text 
def format_number(num):
    if num > 1000000:
        if not num % 1000000:
            return f'{num // 1000000} M'
        return f'{round(num / 1000000, 1)} M'
    return f'{num // 1000} K'


years = range(2024, 2014, -1)  # Generates list from 2024 to 2014 (inclusive)

date_now = int(datetime.datetime.now(pytz.utc).strftime('%d'))
month_now = int(datetime.datetime.now(pytz.utc).strftime('%m'))
year_now = int(datetime.datetime.now(pytz.utc).strftime('%Y'))
time_start = datetime.date(year=min(2010,year_now-10), month=1, day=1) # Take first day and month of 10 years ago / or take first day of 2010
#time_start = datetime.date(year=2014, month=1, day=1)
time_end = datetime.date(year=year_now, month=month_now, day=1) # Take first day of month of today's date
# Streamlit App
#st.title("Equity Analysis App :chart_with_upwards_trend:")  # Title with emoji

# Sidebar for user input
#######################

with st.sidebar:
     st.title(":chart_with_upwards_trend: Stock Analysis")
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

col = st.columns((1.5, 4.5, 2.5), gap='small')
with col[0]:
    
    st.write("Financial summary(Millions)")
    stock = Ticker(selected_ticker)
    #balance_sheet = stock.balance_sheet
    fin_df = stock.balance_sheet()
    #Total Debt, Ordinary Shares Number, Tangible Book Value, Common Stock Equity, Total Capitalization, 
    #TotalAssets	TotalCapitalization	TotalDebt StockholdersEquity CashFinancial
    #columns = ['TotalAssets', 'TotalCapitalization', 'TotalDebt', 'StockholdersEquity', 'CashFinancial']
    # Extract the columns and assign them to the new DataFrame
    new_df = pd.DataFrame()
    new_df['TotalAssets'] = fin_df['TotalAssets'] / 1000000
    new_df['TotalCapitalization'] = fin_df['TotalCapitalization'] /  1000000
    new_df['TotalDebt'] = fin_df['TotalDebt'] / 1000000
    new_df['StockholdersEquity'] = fin_df['StockholdersEquity'] / 1000000
    ##
    new_df['TotalAssets_Difference'] = fin_df['TotalAssets'].diff() / 1000000
    new_df['TotalDebt_Difference'] = fin_df['TotalDebt'].diff() / 1000000
    new_df['StockholdersEquity_Difference'] = fin_df['StockholdersEquity'].diff() / 1000000
    ##
    ##
    #new_df = new_df.dropna()
    st.metric(label="Total Asset", value=new_df.TotalAssets[0], delta=new_df.TotalAssets_Difference[1])
    st.metric(label="TotalDebt", value=new_df.TotalDebt[0],delta=new_df.TotalDebt_Difference[1])
    st.metric(label="Stockholders Equity", value=new_df.StockholdersEquity[0],delta=new_df.StockholdersEquity_Difference[1])

    #st.markdown(lnk + htmlstr, unsafe_allow_html=True)

    st.markdown('#### Cumulative returns')
    monthly_prices = stock_df.resample('M').last()
    #monthly returns

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
    #temp_df['9 Months'] = data['return_9m']
    temp_df['1 Year'] = data['return_12m']
    temp_df = temp_df.head(5)
    #st.dataframe(temp_df.style.format("{:.2%}").highlight_max(color='blue'),hide_index=True)
    #st.table(temp_df)
    st.dataframe(temp_df.style.format({'1 Month': '{:.2f}', '3 Months': '{:.2f}','6 Months': '{:.2f}', '1 Year': '{:.2f}'}).highlight_max(color='blue'),hide_index=True)
    
    st.markdown('#### Valuation Measures')
    yq_valuation_measures = yquery_valuation_measures(selected_ticker)
    st.dataframe(yq_valuation_measures, hide_index=True)

    #st.markdown('Ratio Analysis')
    #ratio_df = fetch_all_ratios(selected_ticker,start_date)
    #st.dataframe(ratio_df)

with col[1]:
    import matplotlib.pyplot as plt
    import seaborn as sns 
    import plotly.graph_objects as go
    from talib import RSI, BBANDS, MACD, HT_TRENDLINE, SAR, SMA
    ticker = selected_ticker
    period = "1d"
    interval = "15m"
    st.markdown(f"Technical Analysis of {selected_ticker}")
    technical_insights = yquery_technical_insights(selected_ticker)
    short_term_outlook = technical_insights['instrumentInfo']['technicalEvents']['shortTermOutlook']
    st_state_description = short_term_outlook['stateDescription']
    st_outlook_direction = short_term_outlook['direction']
    st.write(f"Outlook->{st_state_description}Direction is {st_outlook_direction}")

    data = yf.download(tickers = ticker, start= start_date , end=end_date)
    data = data.reset_index()
    #### Compute Bollinger Bands , rsi , macd
    up, mid, low = BBANDS(data.Close, timeperiod=21, nbdevup=2, nbdevdn=2, matype=0)
    rsi = round(RSI(data.Close, timeperiod=14), 2)
    macd, macdsignal, macdhist = MACD(data.Close, fastperiod=9, slowperiod=20, signalperiod=7)
    #data['Close'] = round(up,2)
    data['BB Up'] = round(up,2)
    data['BB Mid'] = round(mid,2)
    data['BB down'] = round(low,2)
    data['RSI'] = round(rsi,2)
    data['MACD'] = round(macd,4)
    
    data_df = pd.DataFrame()
    data_df['Date'] = pd.to_datetime(data.Date).dt.date
    data_df['Close'] = data['Close']
    #data_df['BB Up'] = data['BB Up'] 
    data_df['BB Mid'] = data['BB Mid']
    #data_df['BB down'] = data['BB down']
    data_df['RSI'] = data['RSI']
    data_df['MACD'] = data['MACD']
    sar_df = data.loc[:, ['Close', 'High', 'Low']]
    sar_df['SAR'] = SAR(sar_df.High, sar_df.Low, 
                      acceleration=0.02, # common value
                      maximum=0.2) 
    
    data_df['SAR'] = sar_df['SAR']
    
#### Now calculate SMA's
    

    sma_df = pd.DataFrame()
    sma_df = data_df.loc[:, ['Date','Close']]
    for t in [9, 20, 50, 100]:
        sma_df[f'SMA_{t}'] = round(SMA(sma_df.Close, timeperiod=t), 2)
    
    
    data_df = data_df.dropna().sort_values(by=['Date'],ascending = False)
    sma_df = sma_df.dropna().sort_values(by=['Date'],ascending = False)
    data_df = data_df.set_index('Date')
    sma_df = sma_df.set_index('Date')
    #st.dataframe(data_df)
    #st.dataframe(sma_df)
    data_df['SMA_9'] = sma_df['SMA_9']
    data_df['SMA_20'] = sma_df['SMA_20']
    data_df['SMA_50'] = sma_df['SMA_50']
    data_df['SMA_100'] = sma_df['SMA_100']
    data_df  = data_df.head(8)
    st.dataframe(data_df)

###### Next Plot the charts
    ### Plotting SMA's
    ax = sma_df.plot(figsize=(14, 6), rot=0, title = f"Moving Averages",)
    ax.set_xlabel('')
    sns.despine()
    # Convert the matplotlib plot to Plotly
    fig = go.Figure()
    for line in ax.get_lines():
        x, y = line.get_data()
        fig.add_trace(go.Scatter(x=x, y=y, mode='lines', name=line.get_label()))
    
    st.plotly_chart(fig)
    
    ### Plotting SAR
    ax = sar_df[['Close', 'SAR']].plot(figsize=(14, 5), style=['-', '--'], title='Parabolic SAR')
    ax.set_xlabel('')
    sns.despine()
    plt.tight_layout()
    st.pyplot(plt)

with col[2]:
     
     st.markdown(" Market Today ")
     industry = 'most_shorted_stocks'
     #today = datetime.today().strftime('%Y%m%d')
     st.write(f"Most Shorted Stocks")
     my_dict = s.get_screeners([industry])
     my_data_list = my_dict[industry]['quotes']

     columns = ['symbol','regularMarketPrice', 'regularMarketVolume', 'fiftyTwoWeekRange','dividendYield','averageDailyVolume10Day', 'averageAnalystRating' ]
     df_ss = pd.DataFrame(my_data_list, columns=columns)
     df_ss.set_index(keys=['symbol'], inplace = True)
     df_ss = df_ss.head(5)
     st.dataframe(df_ss,hide_index=False)


    ####most_watched_tickers
     st.write(f"Most Watched Tickers")
     most_watched_tickers_dict = s.get_screeners(['most_watched_tickers'])
     most_watched_tickers_data_list = most_watched_tickers_dict['most_watched_tickers']['quotes']
     columns = ['symbol', 'fiftyTwoWeekRange', 'regularMarketPrice','regularMarketVolume', 'averageDailyVolume3Month','averageAnalystRating' ]
     df_most_watched_tickers = pd.DataFrame(most_watched_tickers_data_list, columns=columns)
     df_most_watched_tickers.set_index(keys=['symbol'], inplace = True)
     df_most_watched_tickers = df_most_watched_tickers.head(5)
     st.dataframe(df_most_watched_tickers)

     #### most active
     st.write(f"undervalued_growth_stocks")
     industry = 'undervalued_growth_stocks'
     columns = ['symbol', 'averageAnalystRating', 'regularMarketPrice', 'fiftyTwoWeekRange','priceToBook', 'marketCap', 'forwardPE', 'priceEpsCurrentYear', 'bookValue']
     my_dict = industry + '_dict'
     my_data_list = industry + '_data_list'
     my_df = 'df_' + industry
     my_dict = s.get_screeners([industry])
     my_data_list = my_dict[industry]['quotes']

     undervalued_growth_stocks_df = pd.DataFrame(my_data_list, columns=columns)
     undervalued_growth_stocks_df.set_index(keys=['symbol'], inplace = True)
     undervalued_growth_stocks_df = undervalued_growth_stocks_df.head(5)
     st.dataframe(undervalued_growth_stocks_df)
     




    
    

        







   
