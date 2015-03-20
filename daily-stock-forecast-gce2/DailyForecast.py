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
if platform.system() != 'Windows' and platform.system() != 'Darwin':
    import googledatastore as datastore

import logging

#import threading
#import multiprocessing

#Force error if warning needs to be traced
#import warnings
#warnings.simplefilter("error")

def percentDiff(x1,x2):
    return np.abs(x1-x2)/((x1+x2)/2.0)*100.0

def AddIntToDS(entity, name, item, indx=False):
    prop = entity.property.add()
    prop.name = name
    prop.value.indexed = indx
    prop.value.integer_value = int(item)

def AddFloatToDS(entity, name, item, indx=False):
    prop = entity.property.add()
    prop.name = name
    prop.value.indexed = indx
    prop.value.double_value = float("{0:.2f}".format(item))

def AddDoubleListToDS(entity, name, items):
    prop = entity.property.add()
    prop.name = name
    for item in items:
        prop.value.list_value.add().double_value = float("{0:.2f}".format(item))

def AddStrListToDS(entity, name, items):
    prop = entity.property.add()
    prop.name = name
    for item in items:
        prop.value.list_value.add().string_value = item        

def AddStringToDS(entity, name, item, indx=False):
    prop = entity.property.add()
    prop.name = name
    prop.value.indexed = indx
    prop.value.string_value = item

if __name__ == "__main__":

    #launch example python DailyForecast.py

    #Track time of the simulation
    startTime = tt.time()

    #First day of trading
    nowTime = datetime.now(tz=timezone('US/Eastern')).time()
    if nowTime >= time(19,00):
        dayToPredict = datetime.now(tz=timezone('US/Eastern')) + BDay(1)
    else:
        dayToPredict = datetime.now(tz=timezone('US/Eastern')) + BDay(0)
            
    print "\nPredicting %s\n"%dayToPredict.date()
    logging.info("Predicting %s\n"%dayToPredict.date())

    out_of_sameple_bin_size = 10   #days
    history_len             = 500 #days
    saftey_days             = 25  #Number of extra history days to fetch,to ensure domain length matches above

    endOfHistoricalDate   = dayToPredict - BDay(1)
    startOfHistoricalDate = endOfHistoricalDate - BDay(history_len) - BDay(saftey_days)

    predDays = pd.bdate_range(dayToPredict - BDay(10), dayToPredict)

    #Download symbols
    #fullSymbols, fullNames, fullExchange, fullSector, fullIndustry  = GetAllSymbols()
    fullSymbols, fullNames, fullExchange, fullSector, fullIndustry  = (['SPY','HEES','NFLX','AAPL'],
                                                         ['SPY Index','Google Inc.','Netflix, Inc.','Apple Inc.'],
                                                         ['NYSE','NASDAQ','NASDAQ','NASDAQ'],
                                                         ['Technology','Technology','Consumer Services','Technology'],
                                                         ['Software','Software','Entertainment','Software'])
    """fullSymbols, fullNames, fullExchange, fullSector, fullIndustry  = (['SPY'],
                                                                       ['SPY Index'],
                                                                       ['NYSE'],
                                                                       ['Technology'],
                                                                       ['Software'])"""

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
                                           minVolume=1.0,
                                           useThreading=True,
                                           requiredDomain = history_len)
    
    #We need to fetch extra days so we have the right # to handle the fixed dx indexing
    if len(dates[0]) != history_len:
        print "Insufficient domain, increase saftey_days."
        print len(dates[0])
        exit()
    
    #If no stocks in universe, exit
    if(len(symbols) == 0):
        exit()
    
    #Cross validate pred against history at end of simulation
    savedPrediction = {}
    savedScores     = {}

    #Confirm that the day before the prediction is our last history day
    if endOfHistoricalDate.day !=  dates[0][-1].day:
        print "---- WARNING WARNING WARNING ----"
        print str(endOfHistoricalDate.date()) + ' : Expected end of historical'
        print str(dates[0][-1]) + ' : Recieved end of historical'
        print 'We do not have previous day''s values, reject unless holiday\n'
    
    cycleTime = tt.time() #track time of each percent sim progress
    
            
    #SVD on each stock
    progress  = 0
    messageId = 0
    for j in np.arange(len(symbols)):

        #print status
        if int(float(j+1)/float(len(symbols))*100) != progress:
            print "Simulation progress: %.1f%%, took %0.0f seconds"%(float(j+1)/float(len(symbols))*100.0, tt.time()-cycleTime)
            progress  = int(float(j+1)/float(len(symbols))*100)
            cycleTime = tt.time() #track time of each percent sim progress
        
        #print "PRE %s"%symbols[j]
        #Test date sync, should be n-1 on left, w. all 3 matching on both sides
        #print dates[j][hWF-1:hWF][0], predDays[i].date()
        
        pHigh, pLow, pOpen, pClose, pVolume, pHighScore, pLowScore, pOpenScore, pCloseScore, pVolumeScore = SupportVectorRegression(symbols[j],
                                                                         [high[j],
                                                                          low[j],
                                                                          openPrice[j],
                                                                          closePrice[j],
                                                                          #volume[j]],
                                                                          np.log(volume[j])],
                                                                         c = 100.0, #100 is cool w 170 sec cycle
                                                                         Gamma = 0.01, #0.007
                                                                         Epsilon = 0.1, #0.1
                                                                         oosd_bin_size= out_of_sameple_bin_size,
                                                                         oosd_lookback=history_len)
        """pHighBest, pLowBest, pOpenBest, pCloseBest, pVolumeBest = GaussianProcessRegressions(symbols[j],
                                                                         [high[j][hWI:hWF],
                                                                          low[j][hWI:hWF],
                                                                          openPrice[j][hWI:hWF],
                                                                          closePrice[j][hWI:hWF],
                                                                          np.log(volume[j][hWI:hWF])],
                                                                         genPlot = False)"""
        pVolume = np.exp(pVolume)
        #Save items to pred array, final item just gets passed through as no real value exists to compare with
        if not symbols[j] in savedPrediction:
            savedPrediction[symbols[j]] = []
        if not symbols[j] in savedScores:
            savedScores[symbols[j]] = []
        savedPrediction[symbols[j]].append([pHigh, pLow, pOpen, pClose, pVolume])
        savedScores[symbols[j]].append([pHighScore, pLowScore, pOpenScore, pCloseScore, pVolumeScore])
        #print "POST %s"%symbols[j]

    logging.info("\nTime of Simulation: {0:,.0f} seconds\n".format((tt.time() - startTime)))
    print("\nTime of Simulation: {0:,.0f} seconds, {1:,.0f} minutes\n".format((tt.time() - startTime), (tt.time() - startTime)/60.0))
    startTime = tt.time()
        
    #Check the simulation result, need to save best

    #test ability to change dates to days
    dayOfWeekAsInt = pd.DatetimeIndex( predDays ).weekday
    dayOfWeekAsStr = []
    for dayInt in dayOfWeekAsInt:
        dayOfWeekAsStr.append(
            {
              0: 'M',
              1: 'Tu',
              2: 'W',
              3: 'Th',
              4: 'F'
            }[dayInt])
        
    #Easy names for index of prediction array
    OPEN   = 2
    CLOSE  = 3
    HIGH   = 0
    LOW    = 1
    VOLUME = 4

    #Rank each stock based on its model score(for the close price)
    rank = {}
    rankScore = []
    for i in np.arange(len(symbols)):
        rankScore.append(np.array(savedScores[symbols[i]])[:,CLOSE][0])
    rankIndex = np.array(rankScore).argsort()[::-1]
    counter = 1
    for i in rankIndex:
        rank[symbols[i]] = counter
        counter += 1
        #print rankIndex[i]+1, symbols[i] , np.array(savedScores[symbols[i]])[:,CLOSE][0]
    #
        
    if platform.system() != 'Windows' and platform.system() != 'Darwin':
        # Set the dataset from the command line parameters.
        datastore.set_options(dataset="daily-stock-forecast")

        #Save each symbol into the datastore
        for i in np.arange(len(symbols)):
            if rank[symbols[i]] <= 100000:
                try:
                    req = datastore.CommitRequest()
                    req.mode = datastore.CommitRequest.NON_TRANSACTIONAL
                    entity = req.mutation.insert_auto_id.add()

                    # Create a new entity key.
                    key = datastore.Key()
                    
                    # Set the entity key with only one `path_element`: no parent.
                    path = key.path_element.add()
                    path.kind = 'Forecast'

                    # Copy the entity key.
                    entity.key.CopyFrom(key)
                    
                    # - a dateTimeValue 64bit integer: `date`
                    prop = entity.property.add()
                    prop.name = 'date'
                    prop.value.timestamp_microseconds_value = long(tt.mktime(dayToPredict.timetuple()) * 1e6)
                    #prop.value.timestamp_microseconds_value = long(tt.time() * 1e6)

                    AddIntToDS(entity, 'rank', rank[symbols[i]], True)
                    AddStringToDS(entity, 'symbol', symbols[i], True)
                    AddStringToDS(entity, 'company', names[i], True)
                    AddStringToDS(entity, 'exchange', exchanges[i], True)
                    AddStringToDS(entity, 'sector', sector[i], True)
                    AddStringToDS(entity, 'industry', industry[i], True)

                    #predictions
                    AddDoubleListToDS(entity, 'openPredPrice', np.array(savedPrediction[symbols[i]])[:,OPEN])
                    AddDoubleListToDS(entity, 'closePredPrice', np.array(savedPrediction[symbols[i]])[:,CLOSE])
                    AddDoubleListToDS(entity, 'highPredPrice', np.array(savedPrediction[symbols[i]])[:,HIGH])
                    AddDoubleListToDS(entity, 'lowPredPrice', np.array(savedPrediction[symbols[i]])[:,LOW])
                    AddDoubleListToDS(entity, 'volumePred', np.array(savedPrediction[symbols[i]])[:,VOLUME])
                    AddStrListToDS(entity, 'dayOfPred', dayOfWeekAsStr)

                    #History lists
                    #print type(volume[i][0]), type(low[i][0]), float("{0:.2f}".format(volume[i][0]))
                    AddDoubleListToDS(entity, 'openPriceHistory', openPrice[i][-NPredPast+1:])
                    AddDoubleListToDS(entity, 'closePriceHistory', closePrice[i][-NPredPast+1:])
                    AddDoubleListToDS(entity, 'highPriceHistory', high[i][-NPredPast+1:])
                    AddDoubleListToDS(entity, 'lowPriceHistory', low[i][-NPredPast+1:])
                    AddDoubleListToDS(entity, 'volumeHistory', volume[i][-NPredPast+1:])
    #                AddStrListToDS(entity, 'dayOfWeekHistory', dayOfWeekAsStr[:-1])

                    #prediction correlation value, R2
                    #print len(np.array(savedPrediction[symbols[i]])[:,OPEN][:-1]), len(openPrice[i][-NPredPast+1:])
                    openR2 = np.corrcoef(np.array(savedPrediction[symbols[i]])[:,OPEN][:-1], openPrice[i][-NPredPast+1:])[0][1]
                    AddFloatToDS(entity, 'openPredR2', openR2)
                    closeR2 = np.corrcoef(np.array(savedPrediction[symbols[i]])[:,CLOSE][:-1], closePrice[i][-NPredPast+1:])[0][1]
                    AddFloatToDS(entity, 'closePredR2', closeR2)
                    highR2 = np.corrcoef(np.array(savedPrediction[symbols[i]])[:,HIGH][:-1], high[i][-NPredPast+1:])[0][1]
                    AddFloatToDS(entity, 'highPredR2', highR2)
                    lowR2 = np.corrcoef(np.array(savedPrediction[symbols[i]])[:,LOW][:-1], low[i][-NPredPast+1:])[0][1]
                    AddFloatToDS(entity, 'lowPredR2', lowR2)
                    volR2 = np.corrcoef(np.array(savedPrediction[symbols[i]])[:,VOLUME][:-1], volume[i][-NPredPast+1:])[0][1]
                    AddFloatToDS(entity, 'volumePredR2', volR2)

                    #prediction correlation slope
                    #print len(openPrice[i][-NPredPast+1:]), len( np.array(savedPrediction[symbols[i]])[:,OPEN][:-1])
                    slope, intercept, r_value, p_value, std_err = stats.linregress(openPrice[i][-NPredPast+1:], np.array(savedPrediction[symbols[i]])[:,OPEN][:-1])
                    AddFloatToDS(entity, 'openPredSlope', slope)
                    if np.mean([1.0-openR2,abs(1.0-slope)]) <= 0.05:
                        AddIntToDS(entity, 'openModelAccuracy', 1)
                    elif np.mean([1.0-openR2,abs(1.0-slope)]) < 0.1 and np.mean([1.0-openR2,abs(1.0-slope)]) > 0.05:
                        AddIntToDS(entity, 'openModelAccuracy', 2)
                    else:
                        AddIntToDS(entity, 'openModelAccuracy', 3)

                    slope, intercept, r_value, p_value, std_err = stats.linregress(closePrice[i][-NPredPast+1:], np.array(savedPrediction[symbols[i]])[:,CLOSE][:-1])
                    AddFloatToDS(entity, 'closePredSlope', slope)
                    if np.mean([1.0-closeR2,abs(1.0-slope)]) <= 0.05:
                        AddIntToDS(entity, 'closeModelAccuracy', 1)
                    elif np.mean([1.0-closeR2,abs(1.0-slope)]) < 0.1 and np.mean([1.0-closeR2,abs(1.0-slope)]) > 0.05:
                        AddIntToDS(entity, 'closeModelAccuracy', 2)
                    else:
                        AddIntToDS(entity, 'closeModelAccuracy', 3)

                    slope, intercept, r_value, p_value, std_err = stats.linregress(high[i][-NPredPast+1:], np.array(savedPrediction[symbols[i]])[:,HIGH][:-1])
                    AddFloatToDS(entity, 'highPredSlope', slope)
                    if np.mean([1.0-highR2,abs(1.0-slope)]) <= 0.05:
                        AddIntToDS(entity, 'highModelAccuracy', 1)
                    elif np.mean([1.0-highR2,abs(1.0-slope)]) < 0.1 and np.mean([1.0-highR2,abs(1.0-slope)]) > 0.05:
                        AddIntToDS(entity, 'highModelAccuracy', 2)
                    else:
                        AddIntToDS(entity, 'highModelAccuracy', 3)

                    slope, intercept, r_value, p_value, std_err = stats.linregress(low[i][-NPredPast+1:], np.array(savedPrediction[symbols[i]])[:,LOW][:-1])
                    AddFloatToDS(entity, 'lowPredSlope', slope)
                    if np.mean([1.0-lowR2,abs(1.0-slope)]) <= 0.05:
                        AddIntToDS(entity, 'lowModelAccuracy', 1)
                    elif np.mean([1.0-lowR2,abs(1.0-slope)]) < 0.1 and np.mean([1.0-lowR2,abs(1.0-slope)]) > 0.05:
                        AddIntToDS(entity, 'lowModelAccuracy', 2)
                    else:
                        AddIntToDS(entity, 'lowModelAccuracy', 3)

                    slope, intercept, r_value, p_value, std_err = stats.linregress(volume[i][-NPredPast+1:], np.array(savedPrediction[symbols[i]])[:,VOLUME][:-1])
                    AddFloatToDS(entity, 'volumePredSlope', slope)
                    if np.mean([1.0-volR2,abs(1.0-slope)]) <= 0.05:
                        AddIntToDS(entity, 'volumeModelAccuracy', 1)
                    elif np.mean([1.0-volR2,abs(1.0-slope)]) < 0.1 and np.mean([1.0-volR2,abs(1.0-slope)]) > 0.05:
                        AddIntToDS(entity, 'volumeModelAccuracy', 2)
                    else:
                        AddIntToDS(entity, 'volumeModelAccuracy', 3)
                   
                    # Execute the Commit RPC synchronously and ignore the response:
                    # Apply the insert mutation if the entity was not found and close
                    # the transaction.
                    datastore.commit(req)
              
                except datastore.RPCError as e:
                    # RPCError is raised if any error happened during a RPC.
                    # It includes the `method` called and the `reason` of the
                    # failure as well as the original `HTTPResponse` object.
                    logging.error('Error while doing datastore operation')
                    logging.error('RPCError: %(method)s %(reason)s',
                                  {'method': e.method,
                                   'reason': e.reason})
                    logging.error('HTTPError: %(status)s %(reason)s',
                                  {'status': e.response.status,
                                   'reason': e.response.reason})
            if rank[symbols[i]] <= 25:
                #Also commit to the stock list, for faster and cheaper dataastore queries
                try:
                    req = datastore.CommitRequest()
                    req.mode = datastore.CommitRequest.NON_TRANSACTIONAL
                    entity = req.mutation.insert_auto_id.add()

                    # Create a new entity key.
                    key = datastore.Key()
                    
                    # Set the entity key with only one `path_element`: no parent.
                    path = key.path_element.add()
                    path.kind = 'StockList'

                    # Copy the entity key.
                    entity.key.CopyFrom(key)
                    
                    # - a dateTimeValue 64bit integer: `date`
                    prop = entity.property.add()
                    prop.name = 'date'
                    prop.value.timestamp_microseconds_value = long(tt.mktime(dayToPredict.timetuple()) * 1e6)
                    #prop.value.timestamp_microseconds_value = long(tt.time() * 1e6)

                    AddIntToDS(entity, 'rank', rank[symbols[i]], True)
                    AddStringToDS(entity, 'symbol', symbols[i], True)
                    AddStringToDS(entity, 'company', names[i], True)
                    AddStringToDS(entity, 'exchange', exchanges[i], True)

                    AddFloatToDS(entity, 'currentPrice', closePrice[i][-1])

                    AddFloatToDS(entity, 'forecastedPrice', np.array(savedPrediction[symbols[i]])[:,CLOSE][-1])

                    R2 = np.corrcoef(np.array(savedPrediction[symbols[i]])[:,CLOSE][:-1], closePrice[i][-NPredPast+1:])[0][1]
                    slope, intercept, r_value, p_value, std_err = stats.linregress(closePrice[i][-NPredPast+1:], np.array(savedPrediction[symbols[i]])[:,CLOSE][:-1])
                    if np.mean([1.0-R2,abs(1.0-slope)]) <= 0.05:
                        AddIntToDS(entity, 'modelAccuracy', 1)
                    elif np.mean([1.0-R2,abs(1.0-slope)]) < 0.1 and np.mean([R2,abs(1.0-slope)]) > 0.05:
                        AddIntToDS(entity, 'modelAccuracy', 2)
                    else:
                        AddIntToDS(entity, 'modelAccuracy', 3)
                   
                    # Execute the Commit RPC synchronously and ignore the response:
                    # Apply the insert mutation if the entity was not found and close
                    # the transaction.
                    datastore.commit(req)
              
                except datastore.RPCError as e:
                    # RPCError is raised if any error happened during a RPC.
                    # It includes the `method` called and the `reason` of the
                    # failure as well as the original `HTTPResponse` object.
                    logging.error('Error while doing datastore operation')
                    logging.error('RPCError: %(method)s %(reason)s',
                                  {'method': e.method,
                                   'reason': e.reason})
                    logging.error('HTTPError: %(status)s %(reason)s',
                                  {'status': e.response.status,
                                   'reason': e.response.reason})
    else:
        print "\nTop 25:\n\nRank\tSymbol\tCloseR2\tOpenR2\tHighR2\tLowR2\tVolumeR2"
        for i in np.arange(len(symbols)):
            if rank[symbols[i]] <= 25:
                print("{0}\t{1}\t{2:0.4f}\t{3:0.3f}\t{4:0.3f}\t{5:0.3f}\t{6:0.3f}".format(rank[symbols[i]],
                                                  symbols[i],
                                                  np.array(savedScores[symbols[i]])[:,CLOSE][0],
                                                  np.array(savedScores[symbols[i]])[:,OPEN][0],
                                                  np.array(savedScores[symbols[i]])[:,HIGH][0],
                                                  np.array(savedScores[symbols[i]])[:,LOW][0],
                                                  np.array(savedScores[symbols[i]])[:,VOLUME][0]))
        
    logging.info("\nTime to Upload: {0:,.0f} seconds\n".format((tt.time() - startTime)))
    print("\nTime to Upload: {0:,.0f} seconds, {1:,.0f} minutes\n".format((tt.time() - startTime), (tt.time() - startTime)/60.0))

    

    exit()

    #If a small run was done, view the results.
    if len(symbols) <= 3 and platform.system() == 'Windows':
        for i in np.arange(len(symbols)):

            
            labels = ["High","Low","Open","Close"]
            colors = ["r","g","b","c"]
            
            fig = plt.figure()
            #plt.subplots_adjust(left=0.12, bottom=0.06, right=0.90, top=0.96, wspace=0.20, hspace=0.08)
            plt.suptitle("%s Cross Validation of SVR"%symbols[i])
            plt.subplot(3, 2, 1)

            yPredHigh = np.array(savedPrediction[symbols[i]])[:,HIGH][:-1]
            #print high[i][-NPredPast+1:].shape, yPredHigh.shape
            plt.plot( high[i][-NPredPast+1:], yPredHigh, '%s.'%colors[0], label=labels[0], markersize=5, zorder=4)
            #linear fit
            slope, intercept, r_value, p_value, std_err = stats.linregress(high[i][-NPredPast+1:], np.array(savedPrediction[symbols[i]])[:,HIGH][:-1])
            line = slope*high[i][-NPredPast+1:]+intercept
            plt.plot(high[i][-NPredPast+1:],line,'k-', label="%.3f"%slope)
            plt.plot([],[],label= np.corrcoef(np.array(savedPrediction[symbols[i]])[:,HIGH][:-1], high[i][-NPredPast+1:])[0][1] )
            #
            
            plt.ylabel('Predicted')
            plt.grid(True)
            plt.legend(loc='upper left', numpoints=1, ncol=1, fancybox=True, prop={'size':10}, framealpha=0.50)
            
            plt.subplot(3, 2, 2)

            yPredLow = np.array(savedPrediction[symbols[i]])[:,LOW][:-1]
            plt.plot(low[i][-NPredPast+1:], yPredLow, '%s.'%colors[1], label=labels[1], markersize=5, zorder=4)
            #linear fit
            slope, intercept, r_value, p_value, std_err = stats.linregress(low[i][-NPredPast+1:], np.array(savedPrediction[symbols[i]])[:,LOW][:-1])
            line = slope*low[i][-NPredPast+1:]+intercept
            plt.plot(low[i][-NPredPast+1:],line,'k-', label="%.3f"%slope)
            plt.plot([],[],label= np.corrcoef(np.array(savedPrediction[symbols[i]])[:,LOW][:-1], low[i][-NPredPast+1:])[0][1] )
            #
            plt.grid(True)
            plt.legend(loc='upper left', numpoints=1, ncol=1, fancybox=True, prop={'size':10}, framealpha=0.50)

            plt.subplot(3, 2, 3)

            yPredOpen = np.array(savedPrediction[symbols[i]])[:,OPEN][:-1]
            plt.plot(openPrice[i][-NPredPast+1:], yPredOpen, '%s.'%colors[2], label=labels[2], markersize=5, zorder=4)
            #linear fit
            slope, intercept, r_value, p_value, std_err = stats.linregress(openPrice[i][-NPredPast+1:], np.array(savedPrediction[symbols[i]])[:,OPEN][:-1])
            line = slope*openPrice[i][-NPredPast+1:]+intercept
            plt.plot(openPrice[i][-NPredPast+1:],line,'k-', label="%.3f"%slope)
            plt.plot([],[],label= np.corrcoef(np.array(savedPrediction[symbols[i]])[:,OPEN][:-1], openPrice[i][-NPredPast+1:])[0][1] )
            #
            plt.xlabel('Real')
            plt.ylabel('Predicted')
            plt.grid(True)
            plt.legend(loc='upper left', numpoints=1, ncol=1, fancybox=True, prop={'size':10}, framealpha=0.50)

            
            plt.subplot(3, 2, 4)

            yPredClose = np.array(savedPrediction[symbols[i]])[:,CLOSE][:-1]
            plt.plot(closePrice[i][-NPredPast+1:], yPredClose, '%s.'%colors[3], label=labels[3], markersize=5, zorder=4)
            #linear fit
            slope, intercept, r_value, p_value, std_err = stats.linregress(closePrice[i][-NPredPast+1:], np.array(savedPrediction[symbols[i]])[:,CLOSE][:-1])
            line = slope*closePrice[i][-NPredPast+1:]+intercept
            plt.plot(closePrice[i][-NPredPast+1:],line,'k-', label="%.3f"%slope)
            plt.plot([],[],label=np.corrcoef(np.array(savedPrediction[symbols[i]])[:,CLOSE][:-1], closePrice[i][-NPredPast+1:])[0][1] )
            #
            plt.xlabel('Real')
            plt.grid(True)
            plt.legend(loc='upper left', numpoints=1, ncol=1, fancybox=True, prop={'size':10}, framealpha=0.50)

            #plt.show()




            plt.subplot(3, 2, 5)

            plt.ticklabel_format(style='sci')
            vol = np.array(savedPrediction[symbols[i]])[:,VOLUME][:-1]
            plt.plot(volume[i][-NPredPast+1:], vol, '%s.'%colors[3], label=labels[3], markersize=5, zorder=4)
            #linear fit
            slope, intercept, r_value, p_value, std_err = stats.linregress(volume[i][-NPredPast+1:], np.array(savedPrediction[symbols[i]])[:,VOLUME][:-1])
            line = slope*volume[i][-NPredPast+1:]+intercept
            plt.plot(volume[i][-NPredPast+1:],line,'k-', label="%.3f"%slope)
            #print np.corrcoef(np.array(savedPrediction[symbols[i]])[:,VOLUME][:-1], volume[i][-NPredPast+1:])[0][1]
            plt.plot([],[],label= np.corrcoef(np.array(savedPrediction[symbols[i]])[:,VOLUME][:-1], volume[i][-NPredPast+1:])[0][1] )
            #
            plt.xlabel('Real')
            plt.grid(True)
            plt.legend(loc='upper left', numpoints=1, ncol=1, fancybox=True, prop={'size':10}, framealpha=0.50)


            plt.show()


            
            plt.clf()
            plt.cla()
            plt.close()
    
