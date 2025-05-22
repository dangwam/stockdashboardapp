import streamlit as st
import pandas as pd
#import yfinance as yf
from finta import TA
import numpy as np
import mplfinance as mpf
#import yahooquery as yq
from yahooquery import Screener
from yahooquery import Ticker
#from pandas_datareader import data as pdr
from plotly import express as px
import datetime
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

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
# CSS styling
st.markdown("""
<style>

/* Main container adjustments */
[data-testid="block-container"] {
    padding-left: 1rem;
    padding-right: 1rem;
    padding-top: 3rem;
    padding-bottom: 0rem;
    margin-bottom: -7rem;
}

/* General block layout */
[data-testid="stVerticalBlock"] {
    padding-left: 0rem;
    padding-right: 0rem;
}

/* Metric display improvements */
[data-testid="stMetric"] {
    background-color: #393939;
    text-align: center;
    padding: 15px 0;
    border-radius: 10px;
}

[data-testid="stMetricLabel"] {
  display: flex;
  justify-content: center;
  align-items: center;
}

/* Up/down trend icon adjustments */
[data-testid="stMetricDeltaIcon-Up"], 
[data-testid="stMetricDeltaIcon-Down"] {
    position: relative;
    left: 38%;
    transform: translateX(-50%);
}

/* Sidebar header */
.sidebar-header {
    color: white;
    background: linear-gradient(90deg, #003366, #00509e);
    padding: 12px;
    font-size: 20px;
    font-weight: bold;
    border-radius: 8px;
    text-align: center;
    margin-bottom: 10px;
}

/* Sidebar subheader */
.sidebar-subheader {
    color: white;
    background: linear-gradient(90deg, #00509e, #0073e6);
    padding: 8px;
    font-size: 16px;
    border-radius: 8px;
    text-align: center;
    margin-bottom: 15px;
}

/* Sidebar padding adjustments */
[data-testid="stSidebar"] {
    background-color: #1e1e1e;
    padding: 10px;
}

</style>
""", unsafe_allow_html=True)

#######################
print("------------------------------------------------------------Starting-----------------------------------------------------------------------------")
st.markdown(title_format, unsafe_allow_html=True)

s = Screener()

st.header(" Starting")
def day_movers():
    industry = 'day_gainers'
    columns = ['symbol', 'averageAnalystRating', 'regularMarketPrice', 'fiftyTwoWeekRange', 
            'regularMarketVolume',  'averageDailyVolume10Day', 'averageDailyVolume3Month', 
            'priceToBook', 'marketCap', 'forwardPE', 'priceEpsCurrentYear', 'bookValue']
    my_dict = industry + '_dict'
    my_data_list = industry + '_data_list'
    my_df = 'df_' + industry
    my_dict = s.get_screeners([industry])
    my_data_list = my_dict[industry]['quotes']

    day_gainers_df = pd.DataFrame(my_data_list, columns=columns)
    day_gainers_df.set_index(keys=['symbol'], inplace = True)
    day_gainers_df
    day_gainers_df = day_gainers_df.rename(columns = {'averageAnalystRating' : 'analystRtng',
                                                    'regularMarketPrice' : 'close',
                                                    'fiftyTwoWeekRange' : '52weekRange',
                                                    'regularMarketVolume' : 'volume',
                                                    'averageDailyVolume10Day' : '10avgvol',
                                                    'averageDailyVolume3Month' : '3M-ADV',
                                                    'priceToBook' : 'MP-BV',
                                                    'marketCap' : 'MCAP',
                                                    'priceEpsCurrentYear' : 'cr-eps'
                                                    })
    
    return day_gainers_df


def most_shorted_stocks():
    select_count = 25
    industry = 'most_shorted_stocks'

    #data_path = r'C:\Users\pythonProject\data\data_produced\yquerry_screens\options'
    #today = datetime.today().strftime('%Y%m%d')
    #file_name = data_path + f"\\{today}_option_data_multiple_tickers.csv"
    #file_name_oi = data_path + f"\\{today}_oi_data_multiple_tickers.csv"
    my_dict = industry + '_dict'
    my_data_list = industry + '_data_list'
    #my_df = 'df_' + industry
    my_dict = s.get_screeners([industry])
    my_data_list = my_dict[industry]['quotes']
    columns = ['symbol', 'longName', 'regularMarketTime','quoteType', 'dividendYield', 'yieldTTM', 'annualReturnNavY3', 'annualReturnNavY5', 'fiftyTwoWeekRange', 
            'regularMarketPrice', 'regularMarketDayRange', 'regularMarketVolume',
            'regularMarketOpen', 'averageDailyVolume3Month', 'averageDailyVolume10Day', 'averageAnalystRating'
            ]
    df_most_shorted_stocks = pd.DataFrame(my_data_list, columns=columns)
    #df_temp.set_index(keys=['symbol'], inplace = True)

    #df_temp.to_csv(file_name)
    return df_most_shorted_stocks

@st.cache_data
def most_actives():
    industry = 'most_actives'
    select_count = 25
    columns = ['symbol','shortName','triggerable', 'customPriceAlertConfidence',
       'lastClosePriceToNNWCPerShare','priceHint', 'regularMarketChange',
       'regularMarketPrice', 'regularMarketDayHigh',
       'regularMarketDayRange', 'regularMarketDayLow', 'regularMarketVolume',
       'regularMarketPreviousClose', 'bid', 'ask', 'bidSize', 'askSize',
       'averageDailyVolume3Month','averageDailyVolume10Day', 'corporateActions', 'fiftyTwoWeekLowChange',
       'fiftyTwoWeekLowChangePercent', 'fiftyTwoWeekRange',
       'fiftyTwoWeekHighChange', 'fiftyTwoWeekHighChangePercent',
       'fiftyTwoWeekChangePercent', 'dividendDate', 'earningsTimestamp',
       'trailingAnnualDividendRate', 'trailingPE',
       'dividendRate', 'trailingAnnualDividendYield', 'marketState',
       'epsTrailingTwelveMonths', 'epsForward', 'epsCurrentYear',
       'priceEpsCurrentYear', 'sharesOutstanding', 'bookValue',
       'fiftyDayAverage', 'fiftyDayAverageChange',
       'fiftyDayAverageChangePercent', 'twoHundredDayAverage',
       'twoHundredDayAverageChange', 'twoHundredDayAverageChangePercent',
       'marketCap', 'forwardPE', 'priceToBook', 'sourceInterval',
       'fiftyTwoWeekHigh', 'fiftyTwoWeekLow', 'averageAnalystRating',
       'dividendYield', 'exchange', 'regularMarketChangePercent', 
       'lastCloseTevEbitLtm', 'ipoExpectedDate']

    most_actives_dct = s.get_screeners(['most_actives'], int(select_count))
    most_actives_list = most_actives_dct[industry]['quotes']
    most_actives_df = pd.DataFrame(most_actives_list, columns=columns)
    return most_actives_df

df = most_actives()
st.dataframe(df)
