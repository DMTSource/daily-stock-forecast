# -*- coding: cp1252 -*-
"""
Author: Derek M. Tishler
Library : TishlerFinancial v1.0.0
Description: Create SVD Model
Date: 09/14/2011 - DD/MM/YY
"""

#Module imports
import numpy as np
import matplotlib
#matplotlib.use("Agg")
import matplotlib.pyplot as plt
import time
from sklearn.svm import SVR
from sklearn.preprocessing import normalize

def SupportVectorRegression(symbol, seriesSet, c=100, Gamma=0.01, Epsilon=0.1, oosd_bin_size=10, oosd_lookback=100 ): # returnList, indexId,
    """
    regression on the data and optionally display
  
    Parameters
    ----------
    data : array like
        2D array of float data.

    Returns
    -------
    maskArray : numpy array
        2D numpy array containing GaussianProcess fit 
    """
    
    fileName="Images/SVR/%s.png" % (symbol)

    labels = ["High","Low","Open","Close"]
    colors = ["r","g","b","c"]
    
    #import matplotlib.pyplot as plt
    
    predictionSets = []
    scoreSets = []
    count = 0
    
    for series in seriesSet:

        #Perform an analysis of the model w/ ISD and OOSD
        full_series = np.array(series)
        #H = len(series) - oosd_lookback #Len of history data
        #L = oosd_lookback-oosd_bin_size               #Len of analysis lookback
        l = oosd_bin_size            #Len of out of sample AND prediction domain(how many days forecasted)
        power = 1 #N where X^n for weight function
        prediction_history = []
        for i in np.arange(oosd_lookback/oosd_bin_size):
            #Index of current in same, and out of sample data.
            # 3 cases of this slicing
            if   i == 0:
                #First run, only two bins to work with(First OOSD bin, and the rest of the data)
                ISD = full_series[l:]
                OOSD = full_series[:l]
                X = np.arange(l,len(full_series))

                # use a variable weight (~0 - 1.0)
                weight_training = np.power(np.arange(l,len(full_series),dtype=float), power)[::-1]/np.power(np.arange(l,len(full_series),dtype=float), power)[::-1].max()
                # use a variable weight, focus on next day prediction (~0 - 1.0 - ~0)
                weight_score = np.concatenate((np.power(np.arange(1,l+1,dtype=float), power)/np.power(np.arange(1,l+1,dtype=float), power).max(),
                                               np.power(np.arange(l+1,len(full_series)+1,dtype=float), power)[::-1]/np.power(np.arange(l+1,len(full_series)+2,dtype=float), power)[::-1].max()))
                """print len (weight_training)
                print weight_training
                print len (weight_score)
                print weight_score
                print exit()"""
            elif i == oosd_lookback/oosd_bin_size - 1:
                #Last run, only two bins to work with(Last OOSD bin, and the rest of the data)
                ISD = full_series[:-l]
                OOSD = full_series[-l:]
                X = np.arange(0,len(full_series)-l)

                # use a variable weight (~0 - 1.0)
                weight_training = np.power(np.arange(l,len(full_series),dtype=float)+1, power)/np.power(np.arange(l,len(full_series),dtype=float)+1, power).max()
                # use a variable weight, focus on next day prediction (~0 - 1.0 - ~0)
                weight_score = np.concatenate((np.power(np.arange(1,len(full_series)-l+1,dtype=float), power)/np.power(np.arange(1,len(full_series)-l+2,dtype=float), power).max(),
                                               np.power(np.arange(1,l+1,dtype=float), power)[::-1]/np.power(np.arange(1,l+1,dtype=float), power)[::-1].max()))
                """print len (weight_training)
                print weight_training
                print len (weight_score)
                print weight_score
                print exit()"""
            else:
                #Any other run, we have a sandwhich of OOSD in the middle of two ISD sets so we need to aggregate.
                ISD = np.concatenate((full_series[:(l*i)], full_series[l*(i+1):]))
                OOSD = full_series[l*i:l*(i+1)]
                X = np.concatenate(( np.arange(0,(l*i)), np.arange(l*(i+1),len(full_series)) ))

                # use a variable weight (~0 - 1.0)
                weight_training = np.concatenate(( np.power(np.arange(1, l*i+1, dtype=float), power)/np.power(np.arange(1, l*i+1, dtype=float), power).max(),
                                                   np.power(np.arange(l*(i+1), len(full_series), dtype=float), power)[::-1]/np.power(np.arange(l*(i+1), len(full_series),dtype=float), power)[::-1].max() ))
                # use a variable weight, focus on next day prediction (~0 - 1.0 - ~0)
                weight_score = np.concatenate(( np.power(np.arange(1, l*(i+1)+1, dtype=float), power)/np.power(np.arange(1, l*(i+1)+1, dtype=float), power).max(),
                                                np.power(np.arange(l*(i+1), len(full_series), dtype=float), power)[::-1]/np.power(np.arange(l*(i+1), len(full_series)+1, dtype=float), power)[::-1].max() ))
                """print len (weight_training)
                print weight_training
                print len (weight_score)
                print weight_score
                exit()"""
                
            # Domain and range of training data
            #X = np.arange(len(ISD))
            X = np.atleast_2d(X).T
            y = ISD

            # Domain of prediction set
            #x = np.atleast_2d(np.linspace(0, len(ISD)+len(OOSD)-1, len(ISD)+len(OOSD))).T
            #x = np.atleast_2d(np.linspace(len(ISD) ,len(ISD)+len(OOSD)-1, len(OOSD))).T
            x = np.atleast_2d(np.linspace(0, len(full_series)-1, len(full_series))).T

            # epsilon-Support Vector Regression using scikit-learn
            # Read more here: http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html
            SVR_model = SVR(kernel='rbf', C=c,gamma=Gamma, epsilon=Epsilon)
            SVR_model.fit(X,y, weight_training)
            y_predSVR = SVR_model.predict(x)
            
            if np.isnan(full_series).any() or np.isinf(full_series).any():
                log.debug(stock.symbol + " Failed due to data INF or NAN")
                y_score = 0
                break
            else:
                y_score = SVR_model.score(x, full_series, weight_score) #y_predSVR[-len(OOSD):]   np.atleast_2d(y_predSVR[::-1]).T
            
            #log.debug(y_score)
            #print y_score
            prediction_history.append(y_score)
            
        score = np.mean(y_score)
        #print ""
        #print score
        #exit()

        #Make the next day's prediction
        X = np.arange(series.shape[0])
        X = np.atleast_2d(X).T
        y = series
        
        startTime = time.time()
        # Mesh the input space for evaluations of the real function, the prediction and
        # its MSE
        x = np.atleast_2d(np.linspace(len(series), len(series), 1.0)).T

        #kernel='rbf', ‘linear’, ‘poly’, ‘sigmoid’, ‘precomputed’
        SVR_model = SVR(kernel='rbf',C=c,gamma=Gamma, epsilon=Epsilon)

        # use a variable weight, focus on next day prediction (~0 - 1.0 - ~0)
        weight_training = np.power(np.arange(1,len(X)+1,dtype=float), power)/np.power(np.arange(1,len(X)+1), power).max()       
 
        # Fit to data using Maximum Likelihood Estimation of the parameters
        SVR_model.fit(X, y, weight_training)
        
        # Make the prediction on the meshed x-axis (ask for MSE as well)
        y_pred = SVR_model.predict(x)

        """print len(X)
        print X
        print len(x)
        print x
        exit()"""

        #print SVR_model.score(x,y_pred)
        
        #score = gp.score(y, y_pred)
        #print score
        predictionSets.append(y_pred)
        scoreSets.append(score)
        #print "{0:0.1f} minutes to compute Gaussian Process & Fit.".format((time.time() - startTime)/60.0)

            
        count += 1


    
    lookBack = -1


    return predictionSets[0], predictionSets[1], predictionSets[2], predictionSets[3], predictionSets[4], scoreSets[0], scoreSets[1], scoreSets[2], scoreSets[3], scoreSets[4]
