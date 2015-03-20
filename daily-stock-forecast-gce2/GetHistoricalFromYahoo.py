"""
Author: Derek M. Tishler
Library : TishlerFinancial v1.0.0
Download hisorical data from Yahoo Finance and organize
Date: 10/13/2014 - DD/MM/YY
"""

#Module imports
import csv
import urllib2
import datetime
import numpy as np
from scipy import stats
from matplotlib import finance
import threading

def GetStockDataWithThreading(symbol, index, d1, d2, quotes, symbols, names, fullSymbols, fullNames, badIndx):
    try:
        quote = finance.quotes_historical_yahoo(symbol, d1, d2, asobject=True)
        #print("Succesfully Loaded %s, %s\n"%(fullSymbols[index], fullNames[index]))
        quotes[index]  = [quote]
        symbols[index] = fullSymbols[index]
        names[index]   = fullNames[index]
        badIndx[index] = -1 #Filter these out before we apply the mask
    except:
        #print("Failed to Load %s, %s\n"%(fullSymbols[index],fullNames[index]))
        badIndx[index] = index



def GetHistoricalFromYahoo(fullSymbols, fullNames, fullExchange, fullSector, fullIndustry, date1, date2, priceFilterLow=0.0, priceFilterHigh=1e9, minVolume=0.0, useThreading=False, requiredDomain=1):
    """
    Download historical daily data from yahoo finance
  

    Parameters
    ----------
    symbols : array like
        2D array of float data.
    d1 : tuple
        Start date of the historical data, format:(MM, DD, YYYY)
    d2 : array like
        End date of the historical data, format:(MM, DD, YYYY)
    useThreading : boolean
        Boolean telling if to use multithreading or not

    Returns
    -------
    quotes : list
        1D numpy array containing returned company symbols
    symbols : numpy array (Transposed/Vertical)
        1D numpy array containing returned full company name
    names : numpy array (Transposed/Vertical)
        !D numpy array containing returned company symbols
    threads : numpy array (Transposed/Vertical)
        !D numpy array containing returned company symbols

    """

    # Choose a time period reasonnably calm (not too long ago so that we get
    # high-tech firms, and before the 2008 crash)
    #d1 = datetime.datetime(2014, 7, 1)
    #d2 = datetime.datetime(2014, 10, 3)
    d1 = datetime.datetime(date1[2], date1[0], date1[1])
    d2 = datetime.datetime(date2[2], date2[0], date2[1])

    #Lists to populate with histprical data
    quotes      = np.empty((len(fullSymbols),1), dtype=object)
    symbols     = np.empty((len(fullSymbols),1), dtype=object)
    names       = np.empty((len(fullSymbols),1), dtype=object)
    exchanges   = np.copy(fullExchange)
    sectors     = np.copy(fullSector)
    industries  = np.copy(fullIndustry)
    badDataIndx = np.empty((len(fullSymbols),), dtype=int)
    threads     = []

    print "Starting historical download."


    numCPU = 64.0
    offset = 0
    while((len(fullSymbols)-offset)%numCPU != 0.0):
        offset += 1
    numberOfLoops = (len(fullSymbols)-offset)/numCPU
    print "%d blocks with %d threads\n"%(numCPU, numberOfLoops)

    if(numCPU==1):
        useThreading = False

    #Download data for each stock, optional threading for speedup
    if(useThreading):
            
        if offset > 0:
            outterLoop = numCPU + 1
        else:
            outterLoop = numCPU
            
        for i in np.arange(outterLoop):
            threads = []
            
            if offset > 0 and i == outterLoop-1:
                innerLoop = offset
            else:
                innerLoop = numberOfLoops
 
            for j in np.arange(innerLoop):
                #unFlatIndex = (int)(numberOfLoops*i+j)
                unFlatIndex = (int) (numberOfLoops*i+j)
                #print unFlatIndex
                #print numberOfLoops*i+j,i, j, numberOfLoops
                #print numCPU*i+j,  len(fullSymbols)
                t = threading.Thread(target=GetStockDataWithThreading,
                                     args=(fullSymbols[unFlatIndex],
                                           unFlatIndex,
                                           d1,
                                           d2,
                                           quotes,
                                           symbols,
                                           names,
                                           fullSymbols,
                                           fullNames,
                                           badDataIndx))
                threads.append(t)
                t.start()
            [x.join() for x in threads]
    else:

        for i in np.arange(len(fullSymbols)):
            try:
                quote = finance.quotes_historical_yahoo(fullSymbols[i], d1, d2, asobject=True)
                quotes[i]  = [quote]
                symbols[i] = fullSymbols[i]
                names[i]   = fullNames[i]
                badDataIndx[i] = -1 #Filter these out before we apply the mask
            except:
                #print("Failed to Load %s, %s\n"%(fullSymbols[i],fullNames[i]))
                badDataIndx[i] = i
    #Delete the unused array cells
    badDataIndx = np.delete(badDataIndx,np.where(badDataIndx == [-1]))
    quotes = np.delete(quotes,badDataIndx)
    quotes = list(quotes)
    symbols = np.delete(symbols,badDataIndx)
    names   = np.delete(names,badDataIndx)
    exchanges = np.delete(exchanges,badDataIndx)
    sectors = np.delete(sectors,badDataIndx)
    industries = np.delete(industries,badDataIndx)
    print "Failed to download %d of %d symbols"%(len(badDataIndx), len(fullSymbols))
    

    #Covert to vertical array(ease of use for numpy and machine learning)
    symbols = np.array(symbols).T
    names   = np.array(names).T
    exchanges  = np.array(exchanges).T
    sectors  = np.array(sectors).T
    industries  = np.array(industries).T
    
    print "Completing historical download."

    #remove none types
    counterBadDim1 = 0
    badIndex = []
    for i in np.arange(len(symbols)):
        if quotes[i] == None or quotes[i] == []:
            badIndex.append(i)
            counterBadDim1 += 1
    if len(badIndex) > 0:
        #quotes  = np.delete(quotes,badIndex)
        temp = 0
        for i in badIndex:
            del quotes[i - temp]
            temp += 1
        symbols = np.delete(symbols,badIndex)
        names   = np.delete(names,badIndex)
        exchanges = np.delete(exchanges,badIndex)
        sectors = np.delete(sectors,badIndex)
        industries = np.delete(industries,badIndex)
    #
    print "NoneType returned for %d of %d symbols"%(counterBadDim1, len(fullSymbols))

    #remove stocks that dont have the full history(non matching dimensions)
    mode = stats.mode(np.array([len(q) for q in quotes]))[0][0]
    counterBadDim2 = 0
    badIndex = []
    for i in np.arange(len(symbols)):
        if len(quotes[i]) != int(mode) or quotes[i] == None:
            #print len(quotes[i])
            badIndex.append(i)
            counterBadDim2 += 1
    if len(badIndex) > 0:
        #quotes  = np.delete(quotes,badIndex)
        temp = 0
        for i in badIndex:
            del quotes[i - temp]
            temp += 1
        symbols = np.delete(symbols,badIndex)
        names   = np.delete(names,badIndex)
        exchanges = np.delete(exchanges,badIndex)
        sectors = np.delete(sectors,badIndex)
        industries = np.delete(industries,badIndex)
    #
    print "Bad dimension for additional %d of %d symbol"%(counterBadDim2, len(symbols)+counterBadDim2)


    # check ahead of time for bad values to prevent breaking the fit with bad division of std in next step
    counterBadDim3 = 0
    badIndex = []
    for i in np.arange(len(symbols)):
        std = (quotes[i].high - quotes[i].low)/((quotes[i].high - quotes[i].low).std())
        if np.isnan(std).any() or np.isinf(std).any():
            badIndex.append(i)
            counterBadDim3 += 1
    if len(badIndex) > 0:
        #quotes  = np.delete(quotes,badIndex)
        temp = 0
        for i in badIndex:
            del quotes[i - temp]
            temp += 1
        symbols = np.delete(symbols,badIndex)
        names   = np.delete(names,badIndex)
        exchanges = np.delete(exchanges,badIndex)
        sectors = np.delete(sectors,badIndex)
        industries = np.delete(industries,badIndex)
    #
    print "INF/NAN error for %d of %d symbol"%(counterBadDim3, len(symbols)+counterBadDim3)

    # Filter the companies by stock price range. Avoid mem error and over plot
    counterBadDim4 = 0
    badIndex = []
    for i in np.arange(len(symbols)):
        if quotes[i].close.min() < priceFilterLow or quotes[i].close.min() > priceFilterHigh:
            badIndex.append(i)
            counterBadDim4 += 1
    if len(badIndex) > 0:
        #quotes  = np.delete(quotes,badIndex)
        temp = 0
        for i in badIndex:
            del quotes[i - temp]
            temp += 1
        symbols = np.delete(symbols,badIndex)
        names   = np.delete(names,badIndex)
        exchanges = np.delete(exchanges,badIndex)
        sectors = np.delete(sectors,badIndex)
        industries = np.delete(industries,badIndex)
    #
    print "Price range filterd %d of %d symbol"%(counterBadDim4, len(symbols)+counterBadDim4)

    # Filter the companies by MISSING volume.
    counterBadDim5 = 0
    badIndex = []
    for i in np.arange(len(symbols)):
        if quotes[i].volume.min() == 0.0:
            badIndex.append(i)
            counterBadDim5 += 1
    if len(badIndex) > 0:
        #quotes  = np.delete(quotes,badIndex)
        temp = 0
        for i in badIndex:
            del quotes[i - temp]
            temp += 1
        symbols = np.delete(symbols,badIndex)
        names   = np.delete(names,badIndex)
        exchanges = np.delete(exchanges,badIndex)
        sectors = np.delete(sectors,badIndex)
        industries = np.delete(industries,badIndex)
    #
    print "Volume missing filterd %d of %d symbol"%(counterBadDim5, len(symbols)+counterBadDim5)

    # Filter the companies by min mean volume.
    counterBadDim6 = 0
    badIndex = []
    for i in np.arange(len(symbols)):
        if quotes[i].volume.min() < minVolume:
            badIndex.append(i)
            counterBadDim6 += 1
    if len(badIndex) > 0:
        #quotes  = np.delete(quotes,badIndex)
        temp = 0
        for i in badIndex:
            del quotes[i - temp]
            temp += 1
        symbols = np.delete(symbols,badIndex)
        names   = np.delete(names,badIndex)
        exchanges = np.delete(exchanges,badIndex)
        sectors = np.delete(sectors,badIndex)
        industries = np.delete(industries,badIndex)
    #
    print "Volume Minimum filterd %d of %d symbol"%(counterBadDim6, len(symbols)+counterBadDim6)
    print "Stock universe contains %d symbol(s)\n"%(len(symbols))


    #Extract needed series for each stock, slice to ensure we get the right domain len
    dates = []
    for q in quotes:
        dates.append(q.date[-requiredDomain:])
    dates = np.array(dates)
    
    high = []
    for q in quotes:
        high.append(q.high[-requiredDomain:])
    high = np.array(high)

    low = []
    for q in quotes:
        low.append(q.low[-requiredDomain:])
    low = np.array(low)

    openPrice = []
    for q in quotes:
        openPrice.append(q.open[-requiredDomain:])
    openPrice = np.array(openPrice)

    closePrice = []
    for q in quotes:
        closePrice.append(q.close[-requiredDomain:])
    closePrice = np.array(closePrice)

    volume = []
    for q in quotes:
        volume.append(q.volume[-requiredDomain:])
    volume = np.array(volume)

    return symbols, names, exchanges, sectors, industries, dates, high, low, openPrice, closePrice, volume
