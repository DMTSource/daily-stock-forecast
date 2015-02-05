"""
Author: Derek M. Tishler
Library : TishlerFinancial v1.0.0
Description: Fetch american stock symbols from web
Date: 10/13/2014 - DD/MM/YY
http://www.nasdaq.com/screening/company-list.aspx
"""

#Module imports
import csv
import urllib2
import numpy as np

def GetAllSymbols():
    """
    Download cvs from nasdaq site of all North American companies
    across: NYSE, NASDAQ, AMEX
  

    Parameters
    ----------
    data : array like
        2D array of float data.

    Returns
    -------
    fullSymbols : numpy array (Transposed/Vertical)
        !D numpy array containing returned company symbols
    fullNames : numpy array (Transposed/Vertical)
        1D numpy array containing returned full company name

    """

    #URL of all american companies, provided by nasdaq site: http://www.nasdaq.com/screening/company-list.aspx
    opener = urllib2.build_opener()
    opener.addheaders = [('User-agent', 'Mozilla/5.0')]
    #All countries, all 3 exchanges(6,682 as of 12-29-2014)
    #url = 'http://www.nasdaq.com/screening/companies-by-industry.aspx?&render=download'

    """if exchange == "ALL":
        url = 'http://www.nasdaq.com/screening/companies-by-industry.aspx?&render=download'
    elif exchange == "AMEX":
        url = 'http://www.nasdaq.com/screening/companies-by-industry.aspx?exchange=AMEX&render=download'
    elif exchange == "NASDAQ":
        url = 'http://www.nasdaq.com/screening/companies-by-industry.aspx?exchange=NASDAQ&render=download'
    elif exchange == "NYSE":
        url = 'http://www.nasdaq.com/screening/companies-by-industry.aspx?exchange=NYSE&render=download'"""

    exchangeURL = {'AMEX':'http://www.nasdaq.com/screening/companies-by-industry.aspx?exchange=AMEX&render=download',
                   'NASDAQ':'http://www.nasdaq.com/screening/companies-by-industry.aspx?exchange=NASDAQ&render=download',
                   'NYSE':'http://www.nasdaq.com/screening/companies-by-industry.aspx?exchange=NYSE&render=download'}

    #AMEX
    #url = 'http://www.nasdaq.com/screening/companies-by-industry.aspx?exchange=AMEX&render=download'

    #NASDAQ
    #url = 'http://www.nasdaq.com/screening/companies-by-industry.aspx?exchange=NASDAQ&render=download'

    #NYSE
    #url = 'http://www.nasdaq.com/screening/companies-by-industry.aspx?exchange=NYSE&render=download'
    

    #Parse out and form lists of items
    skipFirstLine = True
    symbol_dict = {}
    exchange_dict = {}
    sector_dict = {}
    industry_dict = {}

    for exchange in exchangeURL:
    
        cr = opener.open(exchangeURL[exchange])
        
        for line in cr:
            #Skip first row, it contains the keys
            if(skipFirstLine):
                #print line.rstrip()
                skipFirstLine = False
            else:
                #print np.array(line.rstrip().replace("'s","s").replace('"', '').split(','))[[0,1,6,7]]
                #exit()
                (key, symbol, sector, industry) =  np.array(line.rstrip().replace("'s","s").replace('"', '').split(','))[[0,1,6,7]]
                symbol_dict[str(key)]   = symbol
                exchange_dict[str(key)] = exchange
                sector_dict[str(key)]   = sector
                industry_dict[str(key)] = industry

    print "%d symbols in Universe."%(len(symbol_dict))
    
    #return the tranposed arrays
    return np.array(list(symbol_dict.keys())).T, \
           np.array(list(symbol_dict.values())).T, \
           np.array(list(exchange_dict.values())).T, \
           np.array(list(sector_dict.values())).T, \
           np.array(list(industry_dict.values())).T 
