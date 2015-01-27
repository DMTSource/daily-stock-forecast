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

from datetime import datetime, time, timedelta
import time as tt

#import googledatastore as datastore

import logging

#import threading
#import multiprocessing

#Force error if warning needs to be traced
#import warnings
#warnings.simplefilter("error")

#Place the item for each stock into the ndb
OPEN   = 2
CLOSE  = 3
HIGH   = 0
LOW    = 1
VOLUME = 4

def mean(x1,x2):
    return (x1+x2)/2.0

def percentDiff(x1,x2):
    return np.abs(x1-x2)/((x1+x2)/2.0)*100.0

def AddIntToDS(entity, name, item, indx=False):
    prop = entity.property.add()
    prop.name = name
    prop.value.indexed = indx
    prop.value.integer_value = item

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

    #launch example python TishlerManagment.py 10 13 2014

    #Track time of the simulation
    startTime = tt.time()

    #First day of trading
    nowTime = datetime.now(tz=timezone('US/Eastern')).time()
    if nowTime >= time(16,45):
        dayToPredict = datetime.now(tz=timezone('US/Eastern')) + BDay(1) #Get the next bday, if weekend, same as 0
    else:
        dayToPredict = datetime.now(tz=timezone('US/Eastern')) + BDay(0) #Get the current bday, if weekend, same as 1
            
    print "Predicting %s\n"%dayToPredict.date()
    NPredPast             = 90

    startOfPredictSim     = dayToPredict - BDay(NPredPast)

    endOfHistoricalDate   = dayToPredict - BDay(1)
    startOfHistoricalDate = startOfPredictSim - BDay(101)
    
    #Perform a guess for each prediction day
    predDays = pd.bdate_range(startOfPredictSim, dayToPredict)

    #Download symbols
    #fullSymbols, fullNames, fullSector, fullIndustry  = GetAllSymbols()
    """fullSymbols, fullNames, fullSector, fullIndustry  = (['GOOG','SPY','AAPL'],
                                                         ['SPY Index','Google Inc','Apple'],
                                                         ['Technology','Apple','Apple'],
                                                         ['Software','Apple','Apple'])"""
    fullSymbols, fullNames, fullSector, fullIndustry  = (['SPY'],
                                                         ['Google Inc'],
                                                         ['Technology'],
                                                         ['Software'])

    #Download historical data
    symbols, names, sector, industry, dates, high, low, openPrice, closePrice, volume = \
                    GetHistoricalFromYahoo(fullSymbols,
                                           fullNames,
                                           fullSector,
                                           fullIndustry,
                                           (startOfHistoricalDate.month,
                                            startOfHistoricalDate.day,
                                            startOfHistoricalDate.year),
                                           (endOfHistoricalDate.month,
                                            endOfHistoricalDate.day,
                                            endOfHistoricalDate.year),
                                           priceFilterLow=0.0,
                                           priceFilterHigh=1e6,
                                           minVolume=0.0,
                                           useThreading=True) 
    
    #Store predictions best values
    savedPredictionBest  = {}
    savedPredictionSlope = {}
    savedPredictionR2    = {}
    savedVarHigh         = {}
    savedVarLow          = {}
    savedVarOpen         = {}
    savedVarClose        = {}
    savedVarVolume       = {}

    
    #Check that each stock has the right domain(dates), if we cant get historical, then drop that prediction's day
    #loop trough past pred days, remove any that are not in history, dont include final day
    dropList = []
    #We have to check each symbol, but then we must ensure all others are also adjusted
    #for i in np.arange(len(symbols)):
    #Check each date in the predDays
    for j in np.arange(len(predDays)-1):
        if predDays[j].date() not in dates[0]:
            #Track the items to be dropped from the predDays list
            dropList.append(j)
            #Check if this bad date IS contained in other symbols
                
    predDays = predDays.delete(dropList)
    NPredPast = len(predDays)
    #print len(predDays)
    #

    #Make sure the first past prediction day is in the history
    if predDays[0].date() not in dates[0]:
        print("FIRST PREDICTION DAY MISSING FROM HISTORY")
        exit()

    dropList = []
    for i in np.arange(len(predDays)-1):
        for j in np.arange(len(symbols)):
            hWI = i
            hWF = i+np.where(dates[0]==predDays[0].date())[0][0]#98
            #ensure the right day is being forecasted, holes could be in data
            if predDays[i].date() != dates[j][hWF]:
                print "Dates DONOT Match: %s vs %s"%(predDays[i].date(),dates[j][hWF])
                dropList.append(dropList)
    predDays = predDays.delete(dropList)
    #print len(predDays)
    #

    optCount = 0 #We need to fill the first bestPredItems with default values
    #print np.arange(1000,11000,1000)
    #print np.linspace(0.001,0.015,15)
    for C in [5000]:
        for gamma in np.linspace(0.001,0.01,10):
            savedPrediction = {}
            for i in np.arange(len(predDays)):
                #set up moving window on historical data
                #hWI = i
                hWI = 0
                hWF = i+np.where(dates[0]==predDays[0].date())[0][0]#98
         
                #SVD
                messageId = 0
                for j in np.arange(len(symbols)):
                    #Test date sync, should be n-1 on left, w. all 3 matching on both sides
                    #print dates[j][hWF-1:hWF][0], predDays[i].date()
                    pHigh, pLow, pOpen, pClose,  pVolume = SupportVectorRegression(symbols[j],
                                                                                     [high[j][hWI:hWF],
                                                                                      low[j][hWI:hWF],
                                                                                      openPrice[j][hWI:hWF],
                                                                                      closePrice[j][hWI:hWF],
                                                                                       np.log(volume[j][hWI:hWF])],
                                                                                     genPlot = False,
                                                                                     c=C,
                                                                                     Gamma=gamma)
                    """pHigh, pLow, pOpen, pClose,  pVolume = GaussianProcessRegressions(symbols[j],
                                                                             [high[j][hWI:hWF],
                                                                              low[j][hWI:hWF],
                                                                              openPrice[j][hWI:hWF],
                                                                              closePrice[j][hWI:hWF],
                                                                              np.log(volume[j][hWI:hWF])],
                                                                             genPlot = False)"""
                    #pVolume *= volume[j][hWI:hWF].mean()
                    pVolume = np.exp(pVolume)
                    #Save items to pred array, final item just gets passed through as no real value exists to compare with
                    if not symbols[j] in savedPrediction:
                        savedPrediction[symbols[j]] = []
                    savedPrediction[symbols[j]].append([pHigh, pLow, pOpen, pClose,  pVolume])
            #Inner loop is done, all stocks forecasted, lets check if we beat past values, per stock
            for i in np.arange(len(symbols)):
                if optCount == 0:
                    #
                    pHigh         = np.array(savedPrediction[symbols[i]])[:,HIGH]
                    pLow          = np.array(savedPrediction[symbols[i]])[:,LOW]
                    pOpen         = np.array(savedPrediction[symbols[i]])[:,OPEN]
                    pClose        = np.array(savedPrediction[symbols[i]])[:,CLOSE]
                    pVolume       = np.array(savedPrediction[symbols[i]])[:,VOLUME]
                    #store default best prediction values
                    pHighBest     = np.array(savedPrediction[symbols[i]])[:,HIGH]
                    pLowBest      = np.array(savedPrediction[symbols[i]])[:,LOW]
                    pOpenBest     = np.array(savedPrediction[symbols[i]])[:,OPEN]
                    pCloseBest    = np.array(savedPrediction[symbols[i]])[:,CLOSE]
                    pVolumeBest   = np.array(savedPrediction[symbols[i]])[:,VOLUME]
                    #Calculate default best R2 values
                    highR2Best    = np.corrcoef(np.array(savedPrediction[symbols[i]])[:,HIGH][:-1], high[i][-NPredPast+1:])[0][1]
                    lowR2Best     = np.corrcoef(np.array(savedPrediction[symbols[i]])[:,LOW][:-1], low[i][-NPredPast+1:])[0][1]
                    openR2Best    = np.corrcoef(np.array(savedPrediction[symbols[i]])[:,OPEN][:-1], openPrice[i][-NPredPast+1:])[0][1]
                    closeR2Best   = np.corrcoef(np.array(savedPrediction[symbols[i]])[:,CLOSE][:-1], closePrice[i][-NPredPast+1:])[0][1]
                    volumeR2Best  = np.corrcoef(np.array(savedPrediction[symbols[i]])[:,VOLUME][:-1], volume[i][-NPredPast+1:])[0][1]
                    #Calculate default best slope value
                    highSlopeBest, intercept, r_value, p_value, std_err = stats.linregress(high[i][-NPredPast+1:], np.array(savedPrediction[symbols[i]])[:,HIGH][:-1])
                    lowSlopeBest, intercept, r_value, p_value, std_err = stats.linregress(low[i][-NPredPast+1:], np.array(savedPrediction[symbols[i]])[:,LOW][:-1])
                    openSlopeBest, intercept, r_value, p_value, std_err = stats.linregress(openPrice[i][-NPredPast+1:], np.array(savedPrediction[symbols[i]])[:,OPEN][:-1])
                    closeSlopeBest, intercept, r_value, p_value, std_err = stats.linregress(closePrice[i][-NPredPast+1:], np.array(savedPrediction[symbols[i]])[:,CLOSE][:-1])
                    volumeSlopeBest, intercept, r_value, p_value, std_err = stats.linregress(volume[i][-NPredPast+1:], np.array(savedPrediction[symbols[i]])[:,VOLUME][:-1])
                    print 'INITAL HIGH: ',symbols[i],highSlopeBest,highR2Best
                    print 'INITAL LOW: ',symbols[i],lowSlopeBest,lowR2Best
                    print 'INITAL OPEN: ',symbols[i],openSlopeBest,openR2Best
                    print 'INITAL CLOSE: ',symbols[i],closeSlopeBest,closeR2Best
                    print 'INITAL VOLUME: ',symbols[i],volumeSlopeBest,volumeR2Best
                    print ''
                    #Save default values to dict
                    savedPredictionR2[symbols[i]] = [highR2Best, lowR2Best, openR2Best, closeR2Best, volumeR2Best]
                    #Save default values to dict
                    savedPredictionSlope[symbols[i]] = [highSlopeBest, lowSlopeBest, openSlopeBest, closeSlopeBest, volumeSlopeBest]
                    #save new list columns over old ones
                    savedPredictionBest[symbols[i]] = savedPrediction[symbols[i]]#[[],[],[],[],[]]
                    #Save the variables so we know where the opt ended up
                    savedVarHigh[symbols[i]] = [C, gamma]
                    savedVarLow[symbols[i]] = [C, gamma]
                    savedVarOpen[symbols[i]] = [C, gamma]
                    savedVarClose[symbols[i]] = [C, gamma]
                    savedVarVolume[symbols[i]] = [C, gamma]
                #check if themeanse predictions have better slope and r2 than best
                else:
                    #
                    pHigh           = np.array(savedPrediction[symbols[i]])[:,HIGH]
                    pLow            = np.array(savedPrediction[symbols[i]])[:,LOW]
                    pOpen           = np.array(savedPrediction[symbols[i]])[:,OPEN]
                    pClose          = np.array(savedPrediction[symbols[i]])[:,CLOSE]
                    pVolume         = np.array(savedPrediction[symbols[i]])[:,VOLUME]
                    #load the best values for processing, saved at end of statement
                    pHighBest       = np.array(savedPredictionBest[symbols[i]])[:,HIGH]
                    pLowBest        = np.array(savedPredictionBest[symbols[i]])[:,LOW]
                    pOpenBest       = np.array(savedPredictionBest[symbols[i]])[:,OPEN]
                    pCloseBest      = np.array(savedPredictionBest[symbols[i]])[:,CLOSE]
                    pVolumeBest     = np.array(savedPredictionBest[symbols[i]])[:,VOLUME]
                    #
                    highR2Best      = savedPredictionR2[symbols[i]][HIGH]
                    lowR2Best       = savedPredictionR2[symbols[i]][LOW]
                    openR2Best      = savedPredictionR2[symbols[i]][OPEN]
                    closeR2Best     = savedPredictionR2[symbols[i]][CLOSE]
                    volumeR2Best    = savedPredictionR2[symbols[i]][VOLUME]
                    #
                    highSlopeBest   = savedPredictionSlope[symbols[i]][HIGH]
                    lowSlopeBest    = savedPredictionSlope[symbols[i]][LOW]
                    openSlopeBest   = savedPredictionSlope[symbols[i]][OPEN]
                    closeSlopeBest  = savedPredictionSlope[symbols[i]][CLOSE]
                    volumeSlopeBest = savedPredictionSlope[symbols[i]][VOLUME]
                    #OPEN get slope and r2 values, compare to best
                    openR2 = np.corrcoef(np.array(savedPrediction[symbols[i]])[:,OPEN][:-1], openPrice[i][-NPredPast+1:])[0][1]
                    openSlope, intercept, r_value, p_value, std_err = stats.linregress(openPrice[i][-NPredPast+1:], np.array(savedPrediction[symbols[i]])[:,OPEN][:-1])
                    if(mean(1.0-openR2,1.0-openSlope) < mean(1.0-openR2Best,1.0-openSlopeBest)):
                        #print 'OPEN: ',symbols[i],openSlope,openR2 
                        openSlopeBest    = openSlope
                        openR2Best       = openR2
                        pOpenBest        = pOpen
                        savedVarOpen[symbols[i]] = [C, gamma]
                        #loop through each row to save items over
                        for j in np.arange(len(np.array(savedPrediction[symbols[i]])[:,OPEN])):
                            savedPredictionBest[symbols[i]][j][OPEN] = savedPrediction[symbols[i]][j][OPEN]
                        #print np.array(savedPredictionBest[symbols[i]])[:,OPEN]
                        #exit()
                    #CLOSE get slope and r2 values, compare to best
                    closeR2 = np.corrcoef(np.array(savedPrediction[symbols[i]])[:,CLOSE][:-1], closePrice[i][-NPredPast+1:])[0][1]
                    closeSlope, intercept, r_value, p_value, std_err = stats.linregress(closePrice[i][-NPredPast+1:], np.array(savedPrediction[symbols[i]])[:,CLOSE][:-1])
                    if(mean(1.0-closeR2,1.0-closeSlope) < mean(1.0-closeR2Best,1.0-closeSlopeBest)):
                        #print 'CLOSE: ',symbols[i],closeSlope,closeR2 
                        closeSlopeBest   = closeSlope
                        closeR2Best      = closeR2
                        pCloseBest       = pClose
                        savedVarClose[symbols[i]] = [C, gamma]
                        #save new list columns over old ones
                        for j in np.arange(len(np.array(savedPrediction[symbols[i]])[:,CLOSE])):
                            savedPredictionBest[symbols[i]][j][CLOSE] = savedPrediction[symbols[i]][j][CLOSE]
                    #HIGH get slope and r2 values, compare to best    
                    highR2 = np.corrcoef(np.array(savedPrediction[symbols[i]])[:,HIGH][:-1], high[i][-NPredPast+1:])[0][1]
                    highSlope, intercept, r_value, p_value, std_err = stats.linregress(high[i][-NPredPast+1:], np.array(savedPrediction[symbols[i]])[:,HIGH][:-1])
                    if(mean(1.0-highR2,1.0-highSlope) < mean(1.0-highR2Best,1.0-highSlopeBest)):
                        #print 'HIGH: ',symbols[i],highSlope,highR2 
                        highSlopeBest   = highSlope
                        highR2Best      = highR2
                        pHighBest       = pHigh
                        savedVarHigh[symbols[i]] = [C, gamma]
                        #save new list columns over old ones
                        for j in np.arange(len(np.array(savedPrediction[symbols[i]])[:,HIGH])):
                            savedPredictionBest[symbols[i]][j][HIGH] = savedPrediction[symbols[i]][j][HIGH]
                    #LOW get slope and r2 values, compare to best    
                    lowR2 = np.corrcoef(np.array(savedPrediction[symbols[i]])[:,LOW][:-1], low[i][-NPredPast+1:])[0][1]
                    lowSlope, intercept, r_value, p_value, std_err = stats.linregress(low[i][-NPredPast+1:], np.array(savedPrediction[symbols[i]])[:,LOW][:-1])
                    if(mean(1.0-lowR2,1.0-lowSlope) < mean(1.0-lowR2Best,1.0-lowSlopeBest)):
                        #print 'LOW: ',symbols[i],lowSlope,lowR2 
                        lowSlopeBest    = lowSlope
                        lowR2Best       = lowR2
                        pLowBest        = pLow
                        savedVarLow[symbols[i]] = [C, gamma]
                        #save new list columns over old ones
                        for j in np.arange(len(np.array(savedPrediction[symbols[i]])[:,LOW])):
                            savedPredictionBest[symbols[i]][j][LOW] = savedPrediction[symbols[i]][j][LOW]
                    #VOLUME get slope and r2 values, compare to best    
                    volumeR2 = np.corrcoef(np.array(savedPrediction[symbols[i]])[:,VOLUME][:-1], volume[i][-NPredPast+1:])[0][1]
                    volumeSlope, intercept, r_value, p_value, std_err = stats.linregress(volume[i][-NPredPast+1:], np.array(savedPrediction[symbols[i]])[:,VOLUME][:-1])
                    if(mean(1.0-volumeR2,1.0-volumeSlope) < mean(1.0-volumeR2Best,1.0-volumeSlopeBest)):
                        #print 'VOLUME: ',symbols[i],volumeSlope,volumeR2 
                        volumeSlopeBest = volumeSlope
                        volumeR2Best    = volumeR2
                        pVolumeBest     = pVolume
                        savedVarVolume[symbols[i]] = [C, gamma]
                        #save new list columns over old ones
                        for j in np.arange(len(np.array(savedPrediction[symbols[i]])[:,VOLUME])):
                            savedPredictionBest[symbols[i]][j][VOLUME] = savedPrediction[symbols[i]][j][VOLUME]
                    #Save values to dict
                    savedPredictionR2[symbols[i]] = [highR2Best, lowR2Best, openR2Best, closeR2Best, volumeR2Best]
                    #Save values to dict
                    savedPredictionSlope[symbols[i]] = [highSlopeBest, lowSlopeBest, openSlopeBest, closeSlopeBest, volumeSlopeBest]
            optCount += 1
            #End of loop: for i in np.arange(len(predDays)):
        #End of loop: for C in np.linspace(1,1000,100):
    #End of loop: for gamma in np.linspace(0.0001,100.0,100):
            
    savedPrediction = savedPredictionBest #replace so we dont have to retype everything below...

    #change dates to days for plotting on site
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
    #
    #print dayOfWeekAsStr
    
    #Get the rank of each stock, measure the price diff between open & close and rank
    rank = []
    for i in np.arange(len(symbols)):
        #print percentDiff(np.array(savedPrediction[symbols[i]])[:,CLOSE][-1], np.array(savedPrediction[symbols[i]])[:,OPEN][-1])
        rank.append(percentDiff(np.array(savedPrediction[symbols[i]])[:,CLOSE][-1], np.array(savedPrediction[symbols[i]])[:,OPEN][-1]))
    rank = np.array(sorted(range(len(rank)), reverse=True, key=lambda k: rank[k]))+1

    """
    # Set the dataset from the command line parameters.
    datastore.set_options(dataset="daily-stock-forecast")
    
    for i in np.arange(len(symbols)):
        
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
            prop.value.timestamp_microseconds_value = long((dayToPredict-datetime(1970,1,1,tzinfo=timezone('US/Eastern'))).total_seconds())

            AddIntToDS(entity, 'rank', rank[i], True)
            AddStringToDS(entity, 'symbol', symbols[i], True)
            AddStringToDS(entity, 'company', names[i], True)
            AddStringToDS(entity, 'exchange', 'NYSE', True)
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
            #AddStrListToDS(entity, 'dayOfWeekHistory', dayOfWeekAsStr[:-1])

            #prediction correlation value, R2
            #print len(np.array(savedPrediction[symbols[i]])[:,OPEN][:-1]), len(openPrice[i][-NPredPast+1:])
            AddFloatToDS(entity, 'openPredR2', np.corrcoef(np.array(savedPrediction[symbols[i]])[:,OPEN][:-1], openPrice[i][-NPredPast+1:])[0][1])
            AddFloatToDS(entity, 'closePredR2', np.corrcoef(np.array(savedPrediction[symbols[i]])[:,CLOSE][:-1], closePrice[i][-NPredPast+1:])[0][1])
            AddFloatToDS(entity, 'highPredR2', np.corrcoef(np.array(savedPrediction[symbols[i]])[:,HIGH][:-1], high[i][-NPredPast+1:])[0][1])
            AddFloatToDS(entity, 'lowPredR2', np.corrcoef(np.array(savedPrediction[symbols[i]])[:,LOW][:-1], low[i][-NPredPast+1:])[0][1])
            AddFloatToDS(entity, 'volumePredR2', np.corrcoef(np.array(savedPrediction[symbols[i]])[:,VOLUME][:-1], volume[i][-NPredPast+1:])[0][1])

            #prediction correlation slope
            #print len(openPrice[i][-NPredPast+1:]), len( np.array(savedPrediction[symbols[i]])[:,OPEN][:-1])
            slope, intercept, r_value, p_value, std_err = stats.linregress(openPrice[i][-NPredPast+1:], np.array(savedPrediction[symbols[i]])[:,OPEN][:-1])
            AddFloatToDS(entity, 'openPredSlope', slope)
            slope, intercept, r_value, p_value, std_err = stats.linregress(closePrice[i][-NPredPast+1:], np.array(savedPrediction[symbols[i]])[:,CLOSE][:-1])
            AddFloatToDS(entity, 'closePredSlope', slope)
            slope, intercept, r_value, p_value, std_err = stats.linregress(high[i][-NPredPast+1:], np.array(savedPrediction[symbols[i]])[:,HIGH][:-1])
            AddFloatToDS(entity, 'highPredSlope', slope)
            slope, intercept, r_value, p_value, std_err = stats.linregress(low[i][-NPredPast+1:], np.array(savedPrediction[symbols[i]])[:,LOW][:-1])
            AddFloatToDS(entity, 'lowPredSlope', slope)
            slope, intercept, r_value, p_value, std_err = stats.linregress(volume[i][-NPredPast+1:], np.array(savedPrediction[symbols[i]])[:,VOLUME][:-1])
            AddFloatToDS(entity, 'volumePredSlope', slope)

            # Execute the Commit RPC synchronously and ignore the response:
            # Apply the insert mutation if the entity was not found and close
            # the transaction.
            if(rank[i] <= 100):
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
    """

    print "\nTime of Simulation: %f minutes\n"%((tt.time() - startTime)/60.0)

    
    for i in np.arange(len(symbols)):

        print symbols[i]
        print 'HIGH :',np.array(savedPredictionSlope[symbols[i]])[HIGH],np.array(savedPredictionR2[symbols[i]])[HIGH],savedVarHigh[symbols[i]]
        print 'LOW :',np.array(savedPredictionSlope[symbols[i]])[LOW],np.array(savedPredictionR2[symbols[i]])[LOW],savedVarLow[symbols[i]]
        print 'OPEN :',np.array(savedPredictionSlope[symbols[i]])[OPEN],np.array(savedPredictionR2[symbols[i]])[OPEN],savedVarOpen[symbols[i]]
        print 'CLOSE :',np.array(savedPredictionSlope[symbols[i]])[CLOSE],np.array(savedPredictionR2[symbols[i]])[CLOSE],savedVarClose[symbols[i]]
        print 'VOLUME :',np.array(savedPredictionSlope[symbols[i]])[VOLUME],np.array(savedPredictionR2[symbols[i]])[VOLUME],savedVarVolume[symbols[i]]
        print ''
        
        labels = ["High","Low","Open","Close"]
        colors = ["r","g","b","c"]
        
        fig = plt.figure()
        #plt.subplots_adjust(left=0.12, bottom=0.06, right=0.90, top=0.96, wspace=0.20, hspace=0.08)
        plt.suptitle("%s Cross Validation of SVR"%symbols[i])
        plt.subplot(2, 2, 1)

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
        
        plt.subplot(2, 2, 2)

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

        plt.subplot(2, 2, 3)

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

        
        plt.subplot(2, 2, 4)

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




        fig = plt.figure()
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
  

    
