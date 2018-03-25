
import time
import datetime
from pytz import timezone
import numpy as np
import pandas as pd
from pandas.tseries.offsets import BDay

## requires:
## sudo pip install --upgrade pandas-datareader==0.5.0
import pandas_datareader.data as web

def download_north_america_symbols(n=10):
    start_time = time.time()
    # Download from the 3 exchanges we have access to
    nasdaq = pd.read_csv("http://www.nasdaq.com/screening/companies-by-region.aspx?exchange=NASDAQ&render=download")
    nyse   = pd.read_csv("http://www.nasdaq.com/screening/companies-by-region.aspx?exchange=NYSE&render=download")
    amex   = pd.read_csv("http://www.nasdaq.com/screening/companies-by-region.aspx?exchange=AMEX&render=download")

    # Insert label for which exange came from where
    nasdaq['Exchange'] = ['NASDAQ']*len(nasdaq.index)
    nyse['Exchange']   = ['NYSE']*len(nyse.index)
    amex['Exchange']   = ['AMEX']*len(amex.index)

    symbols_combined   = pd.concat([nasdaq, nyse, amex])
    symbol_list        = symbols_combined.sort_values('MarketCap').Symbol.tail(n).values #n=25
    company_name_list  = symbols_combined.sort_values('MarketCap').Name.tail(n).values #n=25
            
    symbol_list        = list(symbol_list) #+ ['^GSPC', '^DJI', '^IXIC']

    elapsed_time       = time.time() - start_time
    print("\nSymbol Universe Download from Nasdaq.com took %d seconds returned %d assets after mkt cap filter."%(elapsed_time, len(symbol_list)))

    return symbol_list, company_name_list

def download_historical(symbol):
    print('\nDownloading\t\"%s\"'%symbol)

    # setup timespan per asset
    hist_len = 252*10
    end   = datetime.datetime.now(tz=timezone('US/Eastern')) + BDay(0) # Last full business day
    start = end - BDay(hist_len)#+hist_len*0.1) 

    f = None

    try:
        f = web.DataReader(symbol.strip(), 'yahoo', start, end).dropna(axis=0).astype(np.float32)
        #f = par.get_data_yahoo(symbol.strip(), start, end).dropna(axis=0).astype(np.float32)

        if len(f.index) < 200:#hist_len*0.95: #required domain check
            print "Failed to download enough history for: %s"%symbol, len(f.index), hist_len
            f = None
    except:
        print "Failed to download: %s"%symbol

    return f