
from itertools import chain
import numpy as np
from decimal import Decimal
#from Data import Data
#from Model import TensorFlow_Basic_MLP

import time
import datetime
from pytz import timezone

import pandas as pd
#import pandas_datareader as par
from pandas.tseries.offsets import BDay

from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from scipy.stats import randint as sp_randint
from scipy.stats import expon
from sklearn.metrics import classification_report
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import precision_recall_fscore_support

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.model_selection import GridSearchCV

from sklearn.pipeline import Pipeline


# https://stackoverflow.com/questions/1447287/format-floats-with-standard-json-module
import json
from json import encoder
encoder.FLOAT_REPR = lambda o: format(o, '.2f')

# dicts for iterating on classifiers
from Classifiers import *
# downloader for symbols and historical data
from Data import download_north_america_symbols, download_historical

max_window = 100 #100

# do at start in case it works past midnight
inference_dt = datetime.datetime.now(tz=timezone('US/Eastern')) + BDay(1)
inference_day = (datetime.datetime.now(tz=timezone('US/Eastern')) + BDay(1)).strftime("%a")

# Utility function to report best scores
def report(results, n_top=1):
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            print("Model with rank: {0}".format(i))
            print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
                  results['mean_test_score'][candidate],
                  results['std_test_score'][candidate]))
            print("Parameters: {0}".format(results['params'][candidate]))
            print("")

def model(clf_name, features, labels):

    start_time = time.time()
    # specify parameters and distributions to sample from
    clf        = make_pipeline(StandardScaler(), PCA(), classifiers[clf_name]) #PCA optional: n_components=2

    '''clf = Pipeline([
        ('reduce_dim', PCA()),
        ('classify',  classifiers[clf_name])
    ])'''
    
    # select correct param set, adjust the pca to current window
    param_dist = param_dict[clf_name]

    param_dist["pca__n_components"] = sp_randint(2, features.shape[1]-1)

    #if 'randomforestclassifier__max_features' in param_dist:
    #    param_dist['randomforestclassifier__max_features'] = sp_randint(2, features.shape[1])

    # run randomized search
    n_iter_search = 2
    random_search = RandomizedSearchCV(clf, param_distributions=param_dist,
                                       n_iter=n_iter_search,
                                       scoring="f1_weighted")

    #start = time.time()
    random_search.fit(features, labels)
    #print("RandomizedSearchCV took %.2f seconds for %d candidates"
    #      " parameter settings." % ((time.time() - start), n_iter_search))
    #report(random_search.cv_results_)

    elapsed_time = time.time() - start_time
    #print 'Optimizing %s on window %d took %d sec'%(clf_name, features.shape[1]/6, elapsed_time)

    return random_search

def model_scan(data, window):

    # assemble the features and labels for current window
    features = []
    labels   = []
    inference_features = []
    for i in np.arange(max_window+1, len(data.index)):
        # select region of data
        feat_i = data.iloc[i-window-1:i].copy()
        # switch to log ret for each column
        feat_i = np.log(feat_i) - np.log(feat_i.shift(1))
        feat_i = feat_i.iloc[1:].replace([np.inf, -np.inf], np.nan)
        feat_i = feat_i.fillna(0.0)    
        features.append(list(feat_i.values.flatten()))

        # use next day intraday change as delta
        s_dp   = np.sign(data.iloc[i].Close-data.iloc[i].Open)
        if s_dp != 1.:
            s_dp = 0.
        labels.append(s_dp)

    # also collect the feature input for the inference
    feat_i = data.iloc[-window-1:].copy()#.replace([np.inf, -np.inf], np.nan).fillna(0.0).replace(0.0, 1e-6)

    feat_i = np.log(feat_i) - np.log(feat_i.shift(1))
    feat_i = feat_i.iloc[1:].replace([np.inf, -np.inf], np.nan)
    feat_i = feat_i.fillna(0.0)
    inference_features.append(list(feat_i.values.flatten()))

    features = np.array(features)
    labels   = np.array(labels)
    
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.1, random_state=42)

    # Get a tuned version of each model
    models = map(model, classifier_names, [X_train[:]]*len(classifier_names), [y_train[:]]*len(classifier_names))

    target_names = ['short', 'long']
    y_preds = [m.best_estimator_.predict(X_test) for m in models]

    forecasts = [m.best_estimator_.predict(inference_features)[0] for m in models]

    clf_reports     = [precision_recall_fscore_support(y_test, item) for item in y_preds]
    clf_reports_tot = [precision_recall_fscore_support(y_test, item, average='weighted') for item in y_preds]
    clf_support_tot = [item[3][0] + item[3][1] for item in clf_reports] 
    print_reports   = [classification_report(y_test, item, target_names=target_names) for item in y_preds]

    scores   = [item.best_score_ for item in models]
    best_idx = np.argmax(scores)

    report_dicts = [{
        "name": classifier_names[i],
        "opt_period": window,
        "precision_sell": clf_reports[i][0][0],
        "precision_buy": clf_reports[i][0][1],
        "precision_avg": clf_reports_tot[i][0],
        "recall_sell": clf_reports[i][1][0],
        "recall_buy": clf_reports[i][1][1],
        "recall_avg": clf_reports_tot[i][1],
        "f1_sell": clf_reports[i][2][0],
        "f1_buy": clf_reports[i][2][1],
        "f1_avg": clf_reports_tot[i][2],
        "support_sell": clf_reports[i][3][0],
        "support_buy": clf_reports[i][3][1],
        "support_avg": clf_support_tot[i]
    } for i in np.arange(len(models))]

    #print classifier_names[best_idx], window
    #report(models[best_idx].cv_results_)

    # get classification report for best model using held out data

    return [models, classifier_names, report_dicts, print_reports, forecasts]

def asset_to_report(symbol, company_name):
    
    # Download availible data
    hist = download_historical(symbol)
    #print hist.head()

    # Begin the model opt process, we work on exterior hyper params(ex: window len) here and let sklearn scan the model itself
    #possible_windows = np.arange(1,max_window+1)[::9]
    #possible_windows = np.linspace(1,max_window,21, dtype=int)
    # array([  1,   5,  10,  15,  20,  25,  30,  35,  40,  45,  50,  55,  60,
    #     65,  70,  75,  80,  85,  90,  95, 100])
    possible_windows = [5,max_window]
    print possible_windows

    print 'Scanning %s for best window + hyperparams...'%symbol
    results = map(model_scan, [hist]*len(possible_windows), possible_windows)
    
    models        = list(chain.from_iterable([item[0] for item in results]))
    names         = list(chain.from_iterable([item[1] for item in results]))
    reports       = list(chain.from_iterable([item[2] for item in results]))
    print_reports = list(chain.from_iterable([item[3] for item in results]))
    forecasts     = list(chain.from_iterable([item[4] for item in results]))
    
    #scores = [item[0].best_score_ for item in results]
    scores        = [item['f1_avg'] for item in reports]
    windows       = [item['opt_period'] for item in reports]
    
    sorted_idx = np.argsort(scores)[::-1]

    best_idx = np.argmax(scores)
    print names[best_idx], windows[best_idx]
    report(models[best_idx].cv_results_)
    print print_reports[best_idx]
    print '\n'

    # create a json report for the site
    report_dict = {}
    report_dict['symbol']          = symbol
    report_dict['name']            = company_name
    report_dict['inference_day']   = inference_day
    if forecasts[best_idx] == 1.0:
        report_dict['inference']       = "upward"# "downward"
        report_dict['inference_color'] = "#4CAF50"# "#F44336"
    else:
        report_dict['inference']       = "downward"
        report_dict['inference_color'] = "#F44336"
    
    report_dict['hist_prices']     = [
        ["Date", "Open", { "role": "style"}], 
        ["%s"%hist.iloc[[-5]].index[0].strftime("%a"), float("%0.2f"%hist.Close.iloc[-5]), "color: #2196F3"],
        ["%s"%hist.iloc[[-4]].index[0].strftime("%a"), float("%0.2f"%hist.Close.iloc[-4]), "color: #2196F3"],
        ["%s"%hist.iloc[[-3]].index[0].strftime("%a"), float("%0.2f"%hist.Close.iloc[-3]), "color: #2196F3"],
        ["%s"%hist.iloc[[-2]].index[0].strftime("%a"), float("%0.2f"%hist.Close.iloc[-2]), "color: #2196F3"],
        ["%s"%hist.iloc[[-1]].index[0].strftime("%a"), float("%0.2f"%hist.Close.iloc[-1]), "color: #2196F3"]
    ]
    report_dict['model_f1s']       = [["Model", "f1", { "role": "style"}]]
    for i in np.arange(len(sorted_idx[:8])):
        report_dict['model_f1s'].append(["%s (%d)"%(names[sorted_idx[i]], windows[sorted_idx[i]]), float("%0.2f"%reports[sorted_idx[i]]['f1_avg']), "color: #2196F3"])

    report_dict['top_model']       = reports[best_idx]

    return report_dict

if __name__ == "__main__":

    
    #symbols,company_names = ['AAPL'], ['Apple Inc.']
    # 5 indicies
    company_names,symbols = [
        'S&P 500',
        'Dow Jones Industrial Average',
        'Nasdaq Composite',
        'CBOE Interest Rate 10 Year T No',
        'CBOE Volatility Index'
        ], [
        '^GSPC',
        '^DJI',
        '^IXIC',
        '^TNX',
        '^VIX'
        ]

    # Get 10 top stocks
    t_symbols, t_company_names = download_north_america_symbols(n=10)
    symbols = symbols + list(t_symbols)
    company_names = symbols + list(t_company_names)

    print symbols
    
    # Get report for each symbol, heavy work per asset
    results = map(asset_to_report, symbols, company_names)

    scores = [item['top_model']['f1_avg'] for item in results]
    sorted_idx = np.argsort(scores)[::-1]

    save_dict = {}
    save_dict['date']  = inference_dt.strftime("%a, %b %d %Y")
    save_dict['items'] = [results[item] for item in sorted_idx]

    # note we are throwing the forecasts file into the polymer site folder
    with open('../polymer-site/src/forecasts.json', 'w') as fp:
        json.dump(save_dict, fp)


    print str(save_dict)
