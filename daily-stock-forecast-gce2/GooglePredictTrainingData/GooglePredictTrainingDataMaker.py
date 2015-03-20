#!/usr/bin/env python
#
# Copyright 2014 DMT SOURCE, LLC.
#
#     DMTSOURCE.COM | CONTACT: DEREK M TISLER lstrdean@gmail.com
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

import sys

from GetSymbols import *
from GetHistoricalFromYahoo import *
#from MeasureVolatility import *
#from Cluster import *
#from GaussianProcess import *
from SupportVectorRegression import *
from GaussianProcess import *

import pandas as pd
from pandas.tseries.offsets import BDay

from scipy import stats

from pytz import timezone

from datetime import datetime, time
import calendar
import time as tt

import platform
if platform.system() != 'Windows':
    import googledatastore as datastore

import logging

#import threading
#import multiprocessing

#Force error if warning needs to be traced
#import warnings
#warnings.simplefilter("error")

if __name__ == "__main__":

    #launch example python DailyForecast.py

    #Track time of the simulation
    startTime = tt.time()

    #First day of trading
    nowTime = datetime.now(tz=timezone('US/Eastern')).time()
    if nowTime >= time(23,59):
        dayToPredict = datetime.now(tz=timezone('US/Eastern')) + BDay(1)
    else:
        dayToPredict = datetime.now(tz=timezone('US/Eastern')) + BDay(0)
            
    print "Predicting %s\n"%dayToPredict.date()
    logging.info("Predicting %s\n"%dayToPredict.date())

    endOfHistoricalDate   = dayToPredict - BDay(1)
    startOfHistoricalDate = endOfHistoricalDate - BDay(1000+1)

    #Download symbols
    #fullSymbols, fullNames, fullExchange, fullSector, fullIndustry  = GetAllSymbols()
    """fullSymbols, fullNames, fullExchange, fullSector, fullIndustry  = (['HEES','NFLX','AAPL'],
                                                         ['Google Inc.','Netflix, Inc.','Apple Inc.'],
                                                         ['NASDAQ','NASDAQ','NASDAQ'],
                                                         ['Technology','Consumer Services ','Technology'],
                                                         ['Software','Entertainment','Software'])"""
    fullSymbols, fullNames, fullExchange, fullSector, fullIndustry  = (['SPY'],
                                                                       ['SPY Index'],
                                                                       ['NYSE'],
                                                                       ['Technology'],
                                                                       ['Software'])

    #Download historical data
    symbols, names, exchanges, sector, industry, dates, high, low, openPrice, closePrice, volume = \
                    GetHistoricalFromYahoo(fullSymbols,
                                           fullNames,
                                           fullExchange,
                                           fullSector,
                                           fullIndustry,
                                           (startOfHistoricalDate.month,
                                            startOfHistoricalDate.day,
                                            startOfHistoricalDate.year),
                                           (endOfHistoricalDate.month,
                                            endOfHistoricalDate.day,
                                            endOfHistoricalDate.year),
                                           priceFilterLow=1.0,
                                           priceFilterHigh=1e6,
                                           minVolume=1000.0,
                                           useThreading=True)
    
    #If no stocks in universe, exit
    if(len(symbols) == 0):
        print 'No symbols in universe'
        exit()
    
    #Cross validate pred against history at end of simulation
    savedPrediction = {}
    
    #Confirm that the day before the prediction is our last history day
    if endOfHistoricalDate.day !=  dates[0][-1].day:
        print endOfHistoricalDate
        print dates[0][-1]
        print 'We do not have previous day''s values'
        exit()
    
    #Create a training data file for each symbol, for each prediction day in the backtest(or just the current)
    for j in np.arange(len(symbols)):
        #create a text file for this symbol
        with open("GooglePredictTrainingData/test_{0:s}.txt".format(symbols[j]), "w") as text_file:
            for k in np.arange(1,len(closePrice[j])):
                text_file.write("{0},{1},{2},{3},{4},{5}\n".format(closePrice[j][k],
                                                                   high[j][k-1],
                                                                   low[j][k-1],
                                                                   openPrice[j][k-1],
                                                                   closePrice[j][k-1],
                                                                   volume[j][k-1]))
        print "The predict string for this test on {0} is:\n X,{1},{2},{3},{4},{5}".format(
                                                                             dates[j][-1],
                                                                             high[j][-1],
                                                                             low[j][-1],
                                                                             openPrice[j][-1],
                                                                             closePrice[j][-1],
                                                                             volume[j][-1])
        
        #Format of prediction training data is
        # example_value, feature_1, feature_2, ...
        
