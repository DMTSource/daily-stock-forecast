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
from sklearn import grid_search
from sklearn import preprocessing

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

        """ Data Configuration & Preprocessing """
        # What features does our model have? We can pick from the bar(open, close, high, low, volume)
        trainingVectors       = np.zeros((series.shape[0]-1, 5),dtype=np.float32)
        testSamples           = np.zeros((1, 5), dtype=np.float32)
        
        scalers = []
        count = 0
        features = [0,1,2,3,4]
        for i in features:
            dataToScale               = seriesSet[i]
            scalers.append(preprocessing.MinMaxScaler(feature_range=(-1, 1)).fit(dataToScale))
            scaledData                = scalers[count].transform(dataToScale)
            
            trainingVectors[:, count] = scaledData[:-1]
            
            testSamples[:, count]     = scaledData[-1]
            count += 1

        """trainingVectors[:, 0] = seriesSet[0][:-2]
        trainingVectors[:, 1] = seriesSet[1][:-2]
        trainingVectors[:, 2] = seriesSet[2][:-2]
        trainingVectors[:, 3] = seriesSet[3][:-2]
        trainingVectors[:, 4] = seriesSet[4][:-2]
        
        # create our scaling transformer to achieve a zero mean and unit variance(std=1). Scale the training data with it.
        scaler0               = preprocessing.MinMaxScaler(feature_range=(-1, 1)).fit(trainingVectors[:, 0])
        scaler1               = preprocessing.MinMaxScaler(feature_range=(-1, 1)).fit(trainingVectors[:, 1])
        scaler2               = preprocessing.MinMaxScaler(feature_range=(-1, 1)).fit(trainingVectors[:, 2])
        scaler3               = preprocessing.MinMaxScaler(feature_range=(-1, 1)).fit(trainingVectors[:, 3])
        scaler4               = preprocessing.MinMaxScaler(feature_range=(-1, 1)).fit(trainingVectors[:, 4])
        
        # Apply the scale transform
        trainingVectors[:, 0] = scaler0.transform(trainingVectors[:, 0])
        trainingVectors[:, 1] = scaler1.transform(trainingVectors[:, 1])
        trainingVectors[:, 2] = scaler2.transform(trainingVectors[:, 2])
        trainingVectors[:, 3] = scaler3.transform(trainingVectors[:, 3])
        trainingVectors[:, 4] = scaler4.transform(trainingVectors[:, 4])"""

        # Target values, we want to use ^ yesterdays bar to predict this day's close price. Use close scaler????????
        targetValues          = np.zeros((series.shape[0]-1, ),dtype=np.float32)
        targetValues[:]       = series[1:]
        scalerTarget          = preprocessing.MinMaxScaler(feature_range=(-1, 1)).fit(targetValues)
        targetValues[:]       = scalerTarget.transform(targetValues)
                
        # Test Samples, scaled using the feature training scaler
        """testSamples           = np.zeros((1, 5), dtype=np.float32)
        testSamples[:, 0]     = seriesSet[0][-1]
        testSamples[:, 0]     = scaler0.transform(testSamples[:, 0])
        testSamples[:, 1]     = seriesSet[1][-1]
        testSamples[:, 1]     = scaler1.transform(testSamples[:, 1])
        testSamples[:, 2]     = seriesSet[2][-1]
        testSamples[:, 2]     = scaler2.transform(testSamples[:, 2])
        testSamples[:, 3]     = seriesSet[3][-1]
        testSamples[:, 3]     = scaler3.transform(testSamples[:, 3])
        testSamples[:, 4]     = seriesSet[4][-1]
        testSamples[:, 4]     = scaler4.transform(testSamples[:, 4])"""
        
        """ Training Weight """
        weight_training = np.power(np.arange(1, targetValues.shape[0]+1,dtype=float), 1)/ \
                          np.power(np.arange(1, targetValues.shape[0]+1,dtype=float), 1).max()
        
        """ Model Optommization """
        parameters    = {'C':[1, 10, 100], 'gamma': np.logspace(-3, -1, 3)} #'kernel':('linear', 'rbf'),
        SVR_model     = SVR()
        clf           = grid_search.GridSearchCV(SVR_model, parameters)
        clf.fit(trainingVectors, targetValues)
        
        """ Forecast next close price """
        #y_predSVR     = clf.predict(testSamples) [0]
        SVR_model     = SVR(C=clf.best_params_["C"], gamma=clf.best_params_["gamma"]) #kernel=clf.best_params_["kernel"]
        #SVR_model     = SVR(C=1, gamma=0.001) #kernel=clf.best_params_["kernel"]
        SVR_model.fit(trainingVectors, targetValues, weight_training)
        y_pred     = scalerTarget.inverse_transform(SVR_model.predict(testSamples))[0]

        """print len(X)
        print X
        print len(x)
        print x
        exit()"""

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
        return predictionSets[0], predictionSets[1], predictionSets[2], predictionSets[3], predictionSets[4]
