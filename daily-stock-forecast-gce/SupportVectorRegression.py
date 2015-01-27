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

def SupportVectorRegression(symbol, seriesSet, genPlot=False, returnItems=[], indexId=0, useThreading=False, c=100, Gamma=0.01, Epsilon=0.1): # returnList, indexId,
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

    if genPlot:
        fig = plt.figure()
    
    predictionSets = []
    count = 0
    
    for series in seriesSet:

        

        X = np.arange(series.shape[0])
        X = np.atleast_2d(X).T
        y = series
        
        
        startTime = time.time()
        # Mesh the input space for evaluations of the real function, the prediction and
        # its MSE
        x = np.atleast_2d(np.linspace(0, len(series), (len(series))*1.0)).T

        #kernel='rbf', ‘linear’, ‘poly’, ‘sigmoid’, ‘precomputed’
        SVR_model = SVR(kernel='rbf',C=c,gamma=Gamma, epsilon=Epsilon)
        
 
        # Fit to data using Maximum Likelihood Estimation of the parameters
        SVR_model.fit(X,y)
        
        # Make the prediction on the meshed x-axis (ask for MSE as well)
        y_pred = SVR_model.predict(x)
        

        #print SVR_model.score(x,y_pred)
        
        #score = gp.score(y, y_pred)
        #print score
        predictionSets.append(y_pred)
        #print "{0:0.1f} minutes to compute Gaussian Process & Fit.".format((time.time() - startTime)/60.0)

        if genPlot:
            # Plot the function, the prediction and the 95% confidence interval based on
            # the MSE
            #fig = plt.figure()
            plt.plot(X, y, '%s.'%colors[count], label=labels[count], markersize=5, zorder=4)
            #plt.plot(dataPost[:,0], dataPost[:,1], 'g.', label=u'Live Data', markersize=5, zorder=4)

            plt.title("Support Vector Regression")
                
            plt.plot(x, y_pred, '%s-'%colors[count], zorder=5)#label=u'Prediction',zorder=5)
            
        count += 1

    if genPlot:
        plt.xlabel('Days')
        plt.ylabel('Stock Price ($)')
        #plt.ylim(-10, 20)
        plt.grid(True)
        plt.legend(loc='upper left', numpoints=1, ncol=2, fancybox=True, prop={'size':10}, framealpha=0.50)
        
        plt.savefig(fileName)
        plt.clf()
        plt.cla()
        plt.close()
        #end gaussian preocess regression
    
    lookBack = -1

    if useThreading:
        returnItems = [indexId, [predictionSets[0][lookBack:].mean(), predictionSets[1][lookBack:].mean(), predictionSets[2][lookBack:].mean(), predictionSets[3][lookBack:].mean(), predictionSets[4][lookBack:].mean()]]
    else:
        return predictionSets[0][lookBack:].mean(), predictionSets[1][lookBack:].mean(), predictionSets[2][lookBack:].mean(), predictionSets[3][lookBack:].mean(), predictionSets[4][lookBack:].mean()
