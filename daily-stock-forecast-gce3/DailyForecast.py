
# Use scoop to run across all cpu, suggested high cpu 32 core google cloud platform.
#python -m scoop DailyForecast.py
# can add -n <n_cpu> to adjust resource useage

import time
import datetime
from pytz import timezone

import numpy as np
import pandas as pd
from scipy import stats
from pandas.tseries.offsets import BDay
import pandas_datareader.data as web
from pandas.tseries.offsets import BDay

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
#from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV

from google.cloud import datastore

# import magic, seriously...
from scoop import futures

import tweepy
from keys import keys


global_start_time = time.time()

# Used to post results to twitter
class TwitterAPI:
    def __init__(self):
        consumer_key        = keys['consumer_key']
        consumer_secret     = keys['consumer_secret']
        auth                = tweepy.OAuthHandler(consumer_key, consumer_secret)
        access_token        = keys['access_token']
        access_token_secret = keys['access_token_secret']
        auth.set_access_token(access_token, access_token_secret)
        self.api = tweepy.API(auth)

    def tweet(self, message):
        self.api.update_status(status=message)

# Download universe from yahoo using pandas_datareader
class Data():
    def __init__(self, symbol_universe=None):

        # Download North American companies from the Nasdaq site. Includes Exchanges: NASDAQ, NYSE, AMEX
        if symbol_universe is None:
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

            # Filter by market cap to focus on liquid assets
            symbol_list = symbols_combined.Symbol[symbols_combined.MarketCap > 5e8].values
    
            self.symbols = list(symbol_list)
            elapsed_time = time.time() - start_time
            print("\nSymbol Universe Download from Nasdaq.com took %d seconds returned %d assets after mkt cap filter."%(elapsed_time, len(self.symbols)))
            
        else:
            self.symbols = list(symbol_universe)

        self.hist_len = 252*2
        self.end   = datetime.datetime.now(tz=timezone('US/Eastern')) + BDay(0) # Last full business day
        self.start = self.end - BDay(self.hist_len+self.hist_len*0.1)

        self.get_universe()

        print('Universe download complete.')

        #new_index = self.hist[self.hist.keys()[0]].index[-self.hist_len:]
        self.new_index = self.hist[self.hist.keys()[0]].index[-self.hist_len:]

        self.close_prices = pd.DataFrame(index = self.new_index, columns=self.hist.keys())
        self.open_prices  = pd.DataFrame(index = self.new_index, columns=self.hist.keys())
        self.high_prices  = pd.DataFrame(index = self.new_index, columns=self.hist.keys())
        self.low_prices   = pd.DataFrame(index = self.new_index, columns=self.hist.keys())
        self.volume       = pd.DataFrame(index = self.new_index, columns=self.hist.keys())

        for stock in self.hist.keys():
            self.close_prices[stock] = self.hist[stock].Close[self.new_index]
            self.open_prices[stock]  = self.hist[stock].Open[self.new_index]
            self.high_prices[stock]  = self.hist[stock].High[self.new_index]
            self.low_prices[stock]   = self.hist[stock].Low[self.new_index]
            self.volume[stock]       = self.hist[stock].Volume[self.new_index]
        #print self.hist[stock][['Close','Open']].pct_change()
        #list(futures.map(self.prep_asset_series, self.hist.keys()))

        # filtering
        if symbol_universe is None:
            print '\nclose shape raw: ',self.close_prices.values.shape
            print 'close shape after symbol with nan scrub: ', self.close_prices.replace(0.0, np.nan).dropna(axis=1).values.shape
            print 'open shape raw: ',self.open_prices.values.shape
            print 'open shape after symbol with nan scrub: ', self.open_prices.replace(0.0, np.nan).dropna(axis=1).values.shape
            print 'high shape raw: ',self.high_prices.values.shape
            print 'high shape after symbol with nan scrub: ', self.high_prices.replace(0.0, np.nan).dropna(axis=1).values.shape
            print 'low shape raw: ',self.low_prices.values.shape
            print 'low shape after symbol with nan scrub: ', self.low_prices.replace(0.0, np.nan).dropna(axis=1).values.shape
        self.close_prices = self.close_prices.dropna(axis=1)
        self.open_prices  = self.open_prices.dropna(axis=1)
        self.high_prices  = self.high_prices.dropna(axis=1)
        self.low_prices   = self.low_prices.dropna(axis=1)
        if symbol_universe is None:
            print 'volume shape raw: ',self.volume.values.shape
            print 'volume shape after symbol with nan scrub: ', self.volume[self.close_prices.columns].fillna(method='bfill').values.shape
        self.volume  = self.volume[self.close_prices.columns].fillna(method='bfill')

        # Print out which columns contain a massive spike in a single days(close to close) percent change
        #print '\nGreater than 60% single day drawdown, removing(should investigate)'
        #print self.close_prices.columns[((self.close_prices.pct_change().replace(np.nan, 0.0) > 0.6).any()).values].values
        #print self.open_prices.columns[((self.open_prices.pct_change().replace(np.nan, 0.0) > 0.6).any()).values].values
        #print self.high_prices.columns[((self.high_prices.pct_change().replace(np.nan, 0.0) > 0.6).any()).values].values
        #print self.low_prices.columns[((self.low_prices.pct_change().replace(np.nan, 0.0) > 0.6).any()).values].values
        #print self.close_prices.columns[((self.close_prices.pct_change().replace(np.nan, 0.0) > 0.6).any()).values].values
        """self.close_prices = self.close_prices.drop(self.close_prices.columns[((self.close_prices.pct_change().replace(np.nan, 0.0).abs() > 0.6).any()).values], 1)
        self.open_prices  = self.open_prices.drop(self.open_prices.columns[((self.open_prices.pct_change().replace(np.nan, 0.0).abs() > 0.6).any()).values], 1)
        self.high_prices  = self.high_prices.drop(self.high_prices.columns[((self.high_prices.pct_change().replace(np.nan, 0.0).abs() > 0.6).any()).values], 1)
        self.low_prices   = self.low_prices.drop(self.low_prices.columns[((self.low_prices.pct_change().replace(np.nan, 0.0).abs() > 0.6).any()).values], 1)
        self.volume       = self.volume.drop(self.close_prices.columns[((self.close_prices.pct_change().replace(np.nan, 0.0).abs() > 0.6).any()).values], 1)
        print '\nclose shape post high price scrub: ',self.close_prices.values.shape
        print 'open  shape post high price scrub: ',self.open_prices.values.shape
        print 'high shape post high price scrub: ',self.high_prices.values.shape
        print 'low  shape post high price scrub: ',self.low_prices.values.shape
        print 'volume shape post high price scrub: ',self.volume.values.shape"""

        if symbol_universe is None:
            newidx = [np.where(a==symbols_combined.Symbol)[0][0] for a in self.close_prices.columns]
            self.names     = symbols_combined.iloc[newidx].Name.values
            self.exchanges = symbols_combined.iloc[newidx].Exchange.values
            self.sector    = symbols_combined.iloc[newidx].Sector.values
            self.industry  = symbols_combined.iloc[newidx].Industry.values

        # Update symbols with the new truncated universe
        self.symbols = self.close_prices.columns.values
    
    def prep_asset_series(self, stock):
        self.close_prices[stock] = self.hist[stock].Close[self.new_index].astype(np.float32)
        self.open_prices[stock]  = self.hist[stock].Open[self.new_index].astype(np.float32)
        self.high_prices[stock]  = self.hist[stock].High[self.new_index].astype(np.float32)
        self.low_prices[stock]   = self.hist[stock].Low[self.new_index].astype(np.float32)
        self.volume[stock]       = self.hist[stock].Volume[self.new_index].astype(np.float32)
        return None

    def get_historical(self, symbol):
        print('Downloading\t\"%s\"'%symbol.strip())
        f = None
        try:
            f = web.DataReader(symbol.strip(), 'yahoo', self.start, self.end)
            if len(f.index) < self.hist_len: #required domain check
                self.symbols.remove(symbol)
                #print 1, symbol
                return None, None
            return symbol, f
        except:
            #print "Failed to download: %s"%symbol
            # A None return will be scrubbed out before saving to hist dictionary
            #print 2,symbol
            self.symbols.remove(symbol)
            return None, None

    def get_universe(self):
        # Download historical data for our universe
        key_value_pairs = list(futures.map(self.get_historical, self.symbols))
        #key_value_pairs = map(self.get_historical, self.symbols)

        #key_value_pairs.remove((None, None)) # remove any failed items
        if (None, None) in key_value_pairs:
            key_value_pairs.remove((None, None)) # remove any failed items

        self.hist       = dict(key_value_pairs)
        if None in self.hist:
            del self.hist[None]

def forecast_asset(close_prices, open_prices, high_prices, low_prices, volume):
    series_to_model = [close_prices, open_prices, high_prices, low_prices, volume]
    series_to_return = []
    for series in series_to_model:
        
        # Input to our model will consist of a 2d 'image' spanning historica data and each candle feature
        window     = 10 # days in window
        n_features = 5   # each item in the candle
        samples    = np.zeros((len(series)-window, window*n_features), dtype=np.float32)
        for i in np.arange(samples.shape[0]):
            aggregate = []
            for j,series2 in enumerate(series_to_model):
                if j!=4:
                    aggregate.append(series2[i:i+window])
                else:
                    # Scale the volume to close price range pre standardization
                    aggregate.append(series_to_model[0][i+window-1]*series2[i:i+window]/series2.max())
            samples[i, :] = np.array(aggregate).flatten()

        # Training Labels, leave out first day to shift ahead by 1 from training input
        labels        = series[window+1:] #Shift ahead by 1 for next day regression

        # Split the data into test/train
        X_train, X_test, y_train, y_test = train_test_split(samples[:-10], labels[:-9], train_size=0.8, random_state=622)
        X_validate = samples[-10:-1]# These are the 10-1 recent samples we use to showcase the model
        y_validate = labels[-9:]
        X_forecast = [samples[-1]] # This is the final value of our series, we will forecast the next day using this

        # Standardize the data. We desire Zero Mean Unit Variance to prevent scaling issues in hyperplane.
        # http://scikit-learn.org/stable/modules/svm.html#svm-regression
        series_scaler = StandardScaler().fit(X_train)
        X_train        = series_scaler.transform(X_train)
        X_test         = series_scaler.transform(X_test)
        X_validate     = series_scaler.transform(X_validate)
        X_forecast     = series_scaler.transform(X_forecast)

        # Perform a random search to optimize hyperparameters
        # (we cant afford an exhaustive grid search so random helps us reduce steps)
        # http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html
        # http://scikit-learn.org/stable/modules/svm.html#svm-regression
        param_dist = {"C": np.logspace(-5,1,7),
                      "epsilon": np.logspace(-6,-1,6),
                      #"kernel": ['linear','poly','rbf','sigmoid'],
                      "kernel": ['linear','rbf'],
                      "degree": np.linspace(1,4,4,dtype=int),
                      #"gamma": np.logspace(-8,-4,5),
                      "coef0": np.logspace(-8,-4,5),
                      "shrinking": [True, False],
                      "tol": np.logspace(-5,-2,4),
                      "cache_size": [100000.],
                      #"max_iter": [True, False],
                      }
        model_template = SVR()
        """param_dist = {"n_estimators": np.linspace(1,25,20,dtype=int),
                      "criterion": ['mse','mae'],
                      "max_features": ['auto','sqrt','log2'],
                      #"max_depth": np.linspace(1,4,4,dtype=int),
                      #"min_samples_split": np.logspace(-8,-4,5),
                      #"min_samples_leaf": np.logspace(-8,-4,5),
                      #"min_weight_fraction_leaf": [True, False],
                      #"max_leaf_nodes": np.logspace(-6,-2,4),
                      "bootstrap": [True, False],
                      #"oob_score": [True, False],
                      #"n_jobs": [True, False],
                      #"random_state": [True, False],
                      "warm_start": [True, False],
                      }
        model_template = RandomForestRegressor(n_jobs=1)"""

        # run randomized search
        n_iter_search = 25
        random_search = RandomizedSearchCV(model_template, 
                                           param_distributions=param_dist,
                                           n_iter=n_iter_search,
                                           n_jobs=1, # Should be 1, unless you use something like scoop+SGE and run on nodes with cpu > 1
                                           random_state=622)

        random_search.fit(X_train, y_train)

        # Utilize the best model to perform forecasts(test and validation)
        model = random_search.best_estimator_
        test_inference     = model.predict(X_test)
        validate_inference = model.predict(X_validate)
        forecast_inference = model.predict(X_forecast)

        series_to_return.append(np.append(validate_inference, forecast_inference))

        # Analysis
        """print model.score(X_test, y_test)
        print model.score(X_validate, y_validate)
        print validate_inference
        print y_validate
        print forecast_inference
        print ''"""

    # order of returned items related to older code for datastore
    return series_to_return[2], series_to_return[3], series_to_return[1], series_to_return[0], series_to_return[4]

# For scoop, this section only run once when parallel
if __name__ == "__main__":

    ## Download a simple market benchmark #############################
    #start_time = time.time()
    #benchmark = Data(['SPY'])
    #print benchmark.close_prices['SPY'].values
    #elapsed_time = time.time() - start_time
    #print("\nBenchmark Download took %d seconds."%elapsed_time)
    ###################################################################


    ## 
    start_time = time.time()
    data = Data()
    print '\nNumber of Stocks in Universe: ',len(data.close_prices.columns)
    elapsed_time = time.time() - start_time
    print("\nUniverse Historical Candle Download took %0.1f minutes."%(elapsed_time/60.0))
    ###################################################################


    # Perform regression for 10 days to showcase model per asset #####
    start_time = time.time()

    predDays = data.close_prices.index[-10:]
    dayToPredict = predDays[-1]+BDay(1)
    print '\nPerforming forecast for next day: %s, Last day in History: %s'%(dayToPredict, predDays[-1])

    #print data.close_prices.values.T
    results_packed = list(futures.map(forecast_asset, 
                                     data.close_prices.values.T,
                                     data.open_prices.values.T,
                                     data.high_prices.values.T,
                                     data.low_prices.values.T,
                                     data.volume.values.T))

    # savedPrediction is dict with key symbols and list of n forecast tuples
    savedPrediction = {}
    for i,item_asset in enumerate(results_packed):
        # reshape tuple to a per day not per series(for use with previous code's datastore upload)
        savedPrediction[data.symbols[i]] = [[item_asset[j][k] for j in np.arange(5)] for k in np.arange(10)]


    elapsed_time = time.time() - start_time
    print("\nUniverse Forecast took %0.1f minutes."%(elapsed_time/60.0))
    ###################################################################


    ## Save results to DataStore #######################################
    start_time = time.time()


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

    NPredPast = 10

    #Get the rank of each stock, measure the price diff between open & close and rank
    rankItems = []
    rankScore = []
    rankIndexOriginal = []
    for i in np.arange(len(data.symbols)):
        rankItems.append(abs((np.array(savedPrediction[data.symbols[i]])[:,CLOSE][-1] - data.close_prices[data.symbols[i]][-1])/abs(data.close_prices[data.symbols[i]][-1])*100.0))
        R2 = np.corrcoef(np.array(savedPrediction[data.symbols[i]])[:,CLOSE][:-1], data.close_prices[data.symbols[i]][-NPredPast+1:])[0][1]
        slope, intercept, r_value, p_value, std_err = stats.linregress(data.close_prices[data.symbols[i]][-NPredPast+1:], np.array(savedPrediction[data.symbols[i]])[:,CLOSE][:-1])
        if np.mean([1.0-R2,abs(1.0-slope)]) <= 0.05:
            rankScore.append(1)
        elif np.mean([1.0-R2,abs(1.0-slope)]) < 0.1 and np.mean([1.0-R2,abs(1.0-slope)]) > 0.05:
            rankScore.append(2)
        else:
            rankScore.append(3)
        rankIndexOriginal.append(i)
    #rankIndex = np.array(rankItems).argsort()[::-1]
    rankItems = np.array(rankItems)
    rankScore = np.array(rankScore)


    #Get the index of each accuracy group
    indexRank1 = np.where(rankScore == 1)
    indexRank2 = np.where(rankScore == 2)
    indexRank3 = np.where(rankScore == 3)
    
    #Sort each accuracy group from high to low close price change
    sortedIndexRank1 = rankItems[indexRank1].argsort()[::-1]
    sortedIndexRank2 = rankItems[indexRank2].argsort()[::-1]
    sortedIndexRank3 = rankItems[indexRank3].argsort()[::-1]

    #Now we can sort the original index list by sliceing with the above groups to save symbol indicies in order of 1,2,3 accuracy
    sortedRankIndexOriginal = np.zeros(len(rankIndexOriginal))
    if len(sortedIndexRank1) > 0:
        sortedRankIndexOriginal[0:len(sortedIndexRank1)] = np.array(rankIndexOriginal)[indexRank1][sortedIndexRank1]
    if len(sortedIndexRank2) > 0:
        sortedRankIndexOriginal[len(sortedIndexRank1):len(sortedIndexRank1)+len(sortedIndexRank2)] = np.array(rankIndexOriginal)[indexRank2][sortedIndexRank2]
    if len(sortedIndexRank3) > 0:
        sortedRankIndexOriginal[len(sortedIndexRank1)+len(sortedIndexRank2):len(sortedIndexRank1)+len(sortedIndexRank2)+len(sortedIndexRank3)] = np.array(rankIndexOriginal)[indexRank3][sortedIndexRank3]

    #loop through the now sorted index list, and use that to fetch each symbol and apply a rank(ascending).
    rank = {}
    counter = 1
    for i in sortedRankIndexOriginal:
        rank[data.symbols[int(i)]] = counter
        counter += 1
    
    # Set the dataset from the command line parameters.
    #datastore.set_options(dataset="daily-stock-forecast")
    client = datastore.Client("daily-stock-forecast")


    top_5 = []
    #Save each symbol into the datastore
    for i in np.arange(len(data.symbols)):
        if rank[data.symbols[i]] <= 100000:
            #try:
            #req = datastore.CommitRequest()
            #req.mode = datastore.CommitRequest.NON_TRANSACTIONAL
            #entity = req.mutation.insert_auto_id.add()

            # Create a new entity key.
            key    = client.key('Forecast')
            entity = datastore.Entity(key)
            
            # Set the entity key with only one `path_element`: no parent.
            #path = key.path_element.add()
            #path.kind = 'Forecast'

            # Copy the entity key.
            #entity.key.CopyFrom(key)
            
            # - a dateTimeValue 64bit integer: `date`
            #prop = entity.property.add()
            #prop.name = 'date'
            #prop.value.timestamp_microseconds_value = long(tt.mktime(dayToPredict.timetuple()) * 1e6)
            entity['date'] = long(time.mktime(dayToPredict.timetuple()) * 1e6)
            #prop.value.timestamp_microseconds_value = long(tt.time() * 1e6)

            entity['rank']     = rank[data.symbols[i]]
            entity['symbol']   = data.symbols[i]
            entity['company']  = data.names[i]
            entity['exchange'] = data.exchanges[i]
            entity['sector']   = data.sector[i]
            entity['industry'] = data.industry[i]

            #predictions
            entity['openPredPrice']  = list(np.array(savedPrediction[data.symbols[i]])[:,OPEN].astype(float))
            entity['closePredPrice'] = list(np.array(savedPrediction[data.symbols[i]])[:,CLOSE].astype(float))
            entity['highPredPrice']  = list(np.array(savedPrediction[data.symbols[i]])[:,HIGH].astype(float))
            entity['lowPredPrice']   = list(np.array(savedPrediction[data.symbols[i]])[:,LOW].astype(float))
            entity['volumePred']     = list(np.array(savedPrediction[data.symbols[i]])[:,VOLUME].astype(float))
            entity['dayOfPred']      = list(dayOfWeekAsStr)
            
            #History lists
            #print type(volume[i][0]), type(low[i][0]), float("{0:.2f}".format(volume[i][0]))
            entity['openPriceHistory']  = list(data.open_prices[data.symbols[i]][-NPredPast+1:].astype(float))
            entity['closePriceHistory'] = list(data.close_prices[data.symbols[i]][-NPredPast+1:].astype(float))
            entity['highPriceHistory']  = list(data.high_prices[data.symbols[i]][-NPredPast+1:].astype(float))
            entity['lowPriceHistory']   = list(data.low_prices[data.symbols[i]][-NPredPast+1:].astype(float))
            entity['volumeHistory']     = list(data.volume[data.symbols[i]][-NPredPast+1:].astype(float))
#                AddStrListToDS(entity, 'dayOfWeekHistory', dayOfWeekAsStr[:-1])

            #prediction correlation value, R2
            #print len(np.array(savedPrediction[symbols[i]])[:,OPEN][:-1]), len(openPrice[i][-NPredPast+1:])
            openR2 = np.corrcoef(np.array(savedPrediction[data.symbols[i]])[:,OPEN][:-1], data.open_prices[data.symbols[i]].values[-NPredPast+1:])[0][1]
            entity['openPredR2'] = float('%0.2f'%openR2)
            closeR2 = np.corrcoef(np.array(savedPrediction[data.symbols[i]])[:,CLOSE][:-1], data.close_prices[data.symbols[i]].values[-NPredPast+1:])[0][1]
            entity['closePredR2'] = float('%0.2f'%closeR2)
            highR2 = np.corrcoef(np.array(savedPrediction[data.symbols[i]])[:,HIGH][:-1], data.high_prices[data.symbols[i]].values[-NPredPast+1:])[0][1]
            entity['highPredR2'] = float('%0.2f'%highR2)
            lowR2 = np.corrcoef(np.array(savedPrediction[data.symbols[i]])[:,LOW][:-1], data.low_prices[data.symbols[i]].values[-NPredPast+1:])[0][1]
            entity['lowPredR2'] = float('%0.2f'%lowR2)
            volR2 = np.corrcoef(np.array(savedPrediction[data.symbols[i]])[:,VOLUME][:-1], data.volume[data.symbols[i]].values[-NPredPast+1:])[0][1]
            entity['volumePredR2'] = float('%0.2f'%volR2)

            #prediction correlation slope
            #print len(openPrice[i][-NPredPast+1:]), len( np.array(savedPrediction[symbols[i]])[:,OPEN][:-1])
            slope, intercept, r_value, p_value, std_err = stats.linregress(data.open_prices[data.symbols[i]].values[-NPredPast+1:], np.array(savedPrediction[data.symbols[i]])[:,OPEN][:-1])
            entity['openPredSlope'] = float('%0.2f'%slope)
            if np.mean([1.0-openR2,abs(1.0-slope)]) <= 0.05:
                entity['openModelAccuracy'] = 1
            elif np.mean([1.0-openR2,abs(1.0-slope)]) < 0.1 and np.mean([1.0-openR2,abs(1.0-slope)]) > 0.05:
                entity['openModelAccuracy'] = 2
            else:
                entity['openModelAccuracy'] = 3

            slope, intercept, r_value, p_value, std_err = stats.linregress(data.close_prices[data.symbols[i]][-NPredPast+1:], np.array(savedPrediction[data.symbols[i]])[:,CLOSE][:-1])
            entity['closePredSlope'] = float('%0.2f'%slope)
            if np.mean([1.0-closeR2,abs(1.0-slope)]) <= 0.05:
                entity['closeModelAccuracy'] = 1
            elif np.mean([1.0-closeR2,abs(1.0-slope)]) < 0.1 and np.mean([1.0-closeR2,abs(1.0-slope)]) > 0.05:
                entity['closeModelAccuracy'] = 2
            else:
                entity['closeModelAccuracy'] = 3

            slope, intercept, r_value, p_value, std_err = stats.linregress(data.high_prices[data.symbols[i]][-NPredPast+1:], np.array(savedPrediction[data.symbols[i]])[:,HIGH][:-1])
            entity['highPredSlope'] = float('%0.2f'%slope)
            if np.mean([1.0-highR2,abs(1.0-slope)]) <= 0.05:
                entity['highModelAccuracy'] = 1
            elif np.mean([1.0-highR2,abs(1.0-slope)]) < 0.1 and np.mean([1.0-highR2,abs(1.0-slope)]) > 0.05:
                entity['highModelAccuracy'] = 2
            else:
                entity['highModelAccuracy'] = 3

            slope, intercept, r_value, p_value, std_err = stats.linregress(data.low_prices[data.symbols[i]][-NPredPast+1:], np.array(savedPrediction[data.symbols[i]])[:,LOW][:-1])
            entity['lowPredSlope'] = float('%0.2f'%slope)
            if np.mean([1.0-lowR2,abs(1.0-slope)]) <= 0.05:
                entity['lowModelAccuracy'] = 1
            elif np.mean([1.0-lowR2,abs(1.0-slope)]) < 0.1 and np.mean([1.0-lowR2,abs(1.0-slope)]) > 0.05:
                entity['lowModelAccuracy'] = 2
            else:
                entity['lowModelAccuracy'] = 3

            slope, intercept, r_value, p_value, std_err = stats.linregress(data.volume[data.symbols[i]][-NPredPast+1:], np.array(savedPrediction[data.symbols[i]])[:,VOLUME][:-1])
            entity['volumePredSlope'] = float('%0.2f'%slope)
            if np.mean([1.0-volR2,abs(1.0-slope)]) <= 0.05:
                entity['volumeModelAccuracy'] = 1
            elif np.mean([1.0-volR2,abs(1.0-slope)]) < 0.1 and np.mean([1.0-volR2,abs(1.0-slope)]) > 0.05:
                entity['volumeModelAccuracy'] = 2
            else:
                entity['volumeModelAccuracy'] = 3

                #computed values
#                AddFloatToDS(entity, 'openPriceChange', np.array(savedPrediction[symbols[i]])[:,OPEN][-1] - openPrice[i][-1])
#                AddFloatToDS(entity, 'openPriceChangePercent', (np.array(savedPrediction[symbols[i]])[:,OPEN][-1] - openPrice[i][-1])/abs(openPrice[i][-1])*100.0)
#                AddFloatToDS(entity, 'closePriceChange', np.array(savedPrediction[symbols[i]])[:,CLOSE][-1] - closePrice[i][-1])
#                AddFloatToDS(entity, 'closePriceChangePercent', (np.array(savedPrediction[symbols[i]])[:,CLOSE][-1] - closePrice[i][-1])/abs(closePrice[i][-1])*100.0)
#                AddFloatToDS(entity, 'highPriceChange', np.array(savedPrediction[symbols[i]])[:,HIGH][-1] - high[i][-1])
#                AddFloatToDS(entity, 'highPriceChangePercent', (np.array(savedPrediction[symbols[i]])[:,HIGH][-1] - high[i][-1])/abs(high[i][-1])*100.0)
#                AddFloatToDS(entity, 'lowPriceChange', np.array(savedPrediction[symbols[i]])[:,LOW][-1] - low[i][-1])
#                AddFloatToDS(entity, 'lowPriceChangePercent', (np.array(savedPrediction[symbols[i]])[:,LOW][-1] - low[i][-1])/abs(low[i][-1])*100.0)
#                AddFloatToDS(entity, 'volumeChange', np.array(savedPrediction[symbols[i]])[:,VOLUME][-1] - volume[i][-1])
#                AddFloatToDS(entity, 'volumeChangePercent', (np.array(savedPrediction[symbols[i]])[:,VOLUME][-1] - volume[i][-1])/abs(volume[i][-1])*100.0)

                #Market snapshot
 #               AddFloatToDS(entity, 'predOpen', np.array(savedPrediction[symbols[i]])[:,OPEN][-1])
 #               AddFloatToDS(entity, 'predClose', np.array(savedPrediction[symbols[i]])[:,CLOSE][-1])
 #               AddFloatToDS(entity, 'predHigh', np.array(savedPrediction[symbols[i]])[:,HIGH][-1])
 #               AddFloatToDS(entity, 'predLow', np.array(savedPrediction[symbols[i]])[:,LOW][-1])
 #               AddFloatToDS(entity, 'predVolume', np.array(savedPrediction[symbols[i]])[:,VOLUME][-1])
               
                # Execute the Commit RPC synchronously and ignore the response:
                # Apply the insert mutation if the entity was not found and close
                # the transaction.
                #datastore.commit(req)
            client.put(entity)
          
            """except datastore.RPCError as e:
                # RPCError is raised if any error happened during a RPC.
                # It includes the `method` called and the `reason` of the
                # failure as well as the original `HTTPResponse` object.
                logging.error('Error while doing datastore operation')
                logging.error('RPCError: %(method)s %(reason)s',
                              {'method': e.method,
                               'reason': e.reason})
                logging.error('HTTPError: %(status)s %(reason)s',
                              {'status': e.response.status,
                               'reason': e.response.reason})"""
        if rank[data.symbols[i]] <= 25:
            #Also commit to the stock list, for faster and cheaper dataastore queries
            #try:
            #req = datastore.CommitRequest()
            #req.mode = datastore.CommitRequest.NON_TRANSACTIONAL
            #entity = req.mutation.insert_auto_id.add()

            # Create a new entity key.
            key = client.key('StockList')
            entity = datastore.Entity(key)
            
            # Set the entity key with only one `path_element`: no parent.
            #path = key.path_element.add()
            #path.kind = 'StockList'

            # Copy the entity key.
            #entity.key.CopyFrom(key)
            
            # - a dateTimeValue 64bit integer: `date`
            #prop = entity.property.add()
            #prop.name = 'date'
            #prop.value.timestamp_microseconds_value = long(tt.mktime(dayToPredict.timetuple()) * 1e6)
            entity['date'] = long(time.mktime(dayToPredict.timetuple()) * 1e6)
            #prop.value.timestamp_microseconds_value = long(tt.time() * 1e6)

            entity['rank']     = rank[data.symbols[i]]
            entity['symbol']   = data.symbols[i]
            entity['company']  = data.names[i]
            entity['exchange'] = data.exchanges[i]

            entity['currentPrice'] = float('%0.2f'%data.close_prices[data.symbols[i]][-1])

            entity['forecastedPrice'] = float('%0.2f'%np.array(savedPrediction[data.symbols[i]])[:,CLOSE][-1])

            R2 = np.corrcoef(np.array(savedPrediction[data.symbols[i]])[:,CLOSE][:-1], data.close_prices[data.symbols[i]][-NPredPast+1:])[0][1]
            slope, intercept, r_value, p_value, std_err = stats.linregress(data.close_prices[data.symbols[i]][-NPredPast+1:], np.array(savedPrediction[data.symbols[i]])[:,CLOSE][:-1])
            if np.mean([1.0-R2,abs(1.0-slope)]) <= 0.05:
                entity['modelAccuracy'] = 1
            elif np.mean([1.0-R2,abs(1.0-slope)]) < 0.1 and np.mean([R2,abs(1.0-slope)]) > 0.05:
                entity['modelAccuracy'] = 2
            else:
                entity['modelAccuracy'] = 3
           
            # Execute the Commit RPC synchronously and ignore the response:
            # Apply the insert mutation if the entity was not found and close
            # the transaction.
            #datastore.commit(req)
            client.put(entity)

            if rank[data.symbols[i]] <= 5:
                if np.sign(float('%0.2f'%np.array(savedPrediction[data.symbols[i]])[:,CLOSE][-1]) - data.close_prices[data.symbols[i]][-1]) > 0:
                    direction = 'Long'
                else:
                    direction = 'Short'
                top_5.append([data.symbols[i], direction])
          
            """except datastore.RPCError as e:
                # RPCError is raised if any error happened during a RPC.
                # It includes the `method` called and the `reason` of the
                # failure as well as the original `HTTPResponse` object.
                logging.error('Error while doing datastore operation')
                logging.error('RPCError: %(method)s %(reason)s',
                              {'method': e.method,
                               'reason': e.reason})
                logging.error('HTTPError: %(status)s %(reason)s',
                              {'status': e.response.status,
                               'reason': e.response.reason})"""


    elapsed_time = time.time() - start_time
    print("\nDatastore creation and upload took %0.1f minutes."%(elapsed_time/60.0))
    ####################################################################


    elapsed_time = time.time() - global_start_time
    print("\nForecast Complete. Took %d seconds."%elapsed_time)

    twitter = TwitterAPI()
    twitter.tweet("Top 5 forecasted #stocks for %s: $%s - %s, $%s - %s, $%s - %s, $%s - %s, $%s - %s. http://daily-stock-forecast.com"%(dayToPredict.strftime('%b, %a %d'),
                                          top_5[0][0],top_5[0][1],
                                          top_5[1][0],top_5[1][1],
                                          top_5[2][0],top_5[2][1],
                                          top_5[3][0],top_5[3][1],
                                          top_5[4][0],top_5[4][1]))