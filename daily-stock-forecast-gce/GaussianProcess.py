"""
Author: Derek M. Tishler
Library : TishlerFinancial v1.0.0
Description: Create Isotopic Regression of dataset
Date: 09/14/2011 - DD/MM/YY
"""

#Module imports
import numpy as np
import matplotlib
#matplotlib.use("Agg")
import matplotlib.pyplot as plt
import time
from sklearn.gaussian_process import GaussianProcess

def GaussianProcessRegressions(symbol, seriesSet, genPlot=False, withError=True,returnItems=[], indexId=0, useThreading=False): # returnList, indexId,
    """
    Performs GaussianProcess regression on the data and optionally display
  

    Parameters
    ----------
    data : array like
        2D array of float data.

    Returns
    -------
    maskArray : numpy array
        2D numpy array containing GaussianProcess fit 

    """
    
    if withError:
        #fileName="Images/GaussianProcess/GaussianProcessRegression-SquExp-%s.png" % (symbol)
        fileName="Images/GaussianProcess/%s.png" % (symbol)
    else:
        #fileName="Images/GaussianProcess/GaussianProcessRegression-Cubic-%s.png" % (symbol)
        fileName="Images/GaussianProcess/%s.png" % (symbol)

    labels = ["High","Low","Open","Close"]
    colors = ["r","g","b","c"]
    
    #calibration function
    def f(x):
        """The function to predict."""
        return x * numpy.sin(x)
    #fast rolling windows for averages, etc
    def rolling_window(a, window):
        shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
        strides = a.strides + (a.strides[-1],)
        return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)
   
    #import matplotlib.pyplot as plt

    if genPlot:
        fig = plt.figure()
    
    predictionSets = []
    count = 0
    
    for series in seriesSet:

        """X = numpy.linspace(0.1, 9.9, 20)
        X = numpy.atleast_2d(X).T"""

        X = np.arange(series.shape[0])
        #X = data[:, 0]
        X = np.atleast_2d(X).T

        # Observations and noise
        """y = f(X).ravel()
        dy = 0.5 + 1.0 * numpy.random.random(y.shape)
        noise = numpy.random.normal(0, dy)
        y += noise"""

        #y = data[:, 1]
        y = series
        
        """
        ##dy = data[:, 2] - data[:, 3]
        #Create a rolling std for each stock price
        stdArr = numpy.zeros(data.shape[0])
        for i in range(data.shape[0]):
            stdArr[i] = numpy.std(data[0:i, 1])
        stdArr[0] = 1e-9
        stdArr[1] = 1e-6
        dy = stdArr
        """
        if withError:
            
            #"In the special case of the squared exponential correlation function, the nugget
            #mathematically represents the variance of the input values."
            #http://scikit-learn.org/stable/modules/generated/sklearn.gaussian_process.GaussianProcess.html#sklearn.gaussian_process.GaussianProcess.fit
            startTime = time.time()
            varianceSubI = np.zeros(series.shape[0])
            for i in range(1,series.shape[0]):
                varianceSubI[i] = np.var( series[0:i]-np.mean(series[0:i]) ) 
            ###dy = varianceSubI-np.mean(varianceSubI) #WOAH Really makes fit proper
#            dy = (varianceSubI-np.mean(varianceSubI)) / ((varianceSubI-np.mean(varianceSubI))**2) #WOAH Really makes fit proper
            #dy = (varianceSubI) / ((np.sum(varianceSubI**2))) #WOAH Really makes fit proper
            #dy -= dy.min()
            #print dy

            dy = series.std()

            #dy[0] = 1e-9
            #dy[1] = 1e-6
            #print "{0:0.1f} seconds to compute variance of the time series.".format(time.time() - startTime)
            #dy = numpy.var(y,axis=0)
            #print dy
            #print type(dy[0])
            
            #dy = np.ones(data.shape[0])*y.std()

            """
            #save the singular value of the stock matrix using SVD for the nugged(sigma_i)
            startTime = time.time()
            singularValues = numpy.zeros(data.shape[0])
            for i in range(2,data.shape[0]):
            #User Singular value decomposition to aquire the singular values sigma_i
            #http://docs.scipy.org/doc/numpy/reference/generated/numpy.linalg.svd.html
            #print numpy.array([X[:,0],data[:, 1]])
                U, s, V = numpy.linalg.svd(numpy.array([X[0:i,0],data[0:i, 1]]), full_matrices=True)
                singularValues[i]= s[1]
            dy = singularValues
            print "{0:0.1f} minutes to compute singular values of the time series.".format((time.time() - startTime)/60.0)
            """

            """
            fig = plt.figure()
            plt.title("$\sigma_i$")
            plt.plot(X[:, 0].T,dy)
            plt.xlabel('Minutes')
            plt.ylabel('$\sigma_i$')
            plt.grid(True)
            plt.savefig("Images/GaussianProcess/MovingSTD.png")
            if showPlot:
                plt.show()
            """
        
        #dy =  numpy.ones(data.shape[0])*numpy.std(rolling_window(y, data.shape[0]), 1)
        
        startTime = time.time()
        # Mesh the input space for evaluations of the real function, the prediction and
        # its MSE
        x = np.atleast_2d(np.linspace(0, len(series), len(series)*1.0)).T
        
        if withError:
            # Instanciate a Gaussian Process model
            
            gp = GaussianProcess(corr='squared_exponential',
                                 theta0=1e-1,
                                 thetaL=1e-3,
                                 thetaU=10,
                                 nugget=(dy / y) ** 2,
                                 random_start=10)
            """
            yunbiased = y-np.mean(y)
            ynorm = np.sum(yunbiased**2)
            autoCorr = (np.correlate(yunbiased, yunbiased, mode='full')/ynorm)-(np.correlate(yunbiased, yunbiased, mode='full')/ynorm).min()
            #autoCorr[np.where(autoCorr == 0.)[0]] = 1e-9
            gp = GaussianProcess(corr='squared_exponential', theta0=autoCorr.mean(),
                                 nugget=(dy / y) ** 2,
                                 random_start=500)
            """
        else:
            gp = GaussianProcess(corr='cubic', theta0=1e-1, thetaL=1e-4, thetaU=1e-1,
                         random_start=100)

          
        # Fit to data using Maximum Likelihood Estimation of the parameters
        
        gp.fit(X, y)
        
        # Make the prediction on the meshed x-axis (ask for MSE as well)
        y_pred, MSE = gp.predict(x, eval_MSE=True)
        #score = gp.score(y, y_pred)
        #print score
        predictionSets.append(y_pred)
        sigma = np.sqrt(MSE)
        #print "{0:0.1f} minutes to compute Gaussian Process & Fit.".format((time.time() - startTime)/60.0)

        if genPlot:
            # Plot the function, the prediction and the 95% confidence interval based on
            # the MSE
            #fig = plt.figure()
            plt.plot(X, y, '%s.'%colors[count], label=labels[count], markersize=5, zorder=4)
            #plt.plot(dataPost[:,0], dataPost[:,1], 'g.', label=u'Live Data', markersize=5, zorder=4)
            if withError:
                plt.title("Gaussian Process Regression(Squared_Exponential)")
                #plt.errorbar(X.ravel(), y, dy, fmt='%s.'%colors[count], markersize=0, barsabove=False, ecolor=colors[count],  alpha=.75, zorder=1)#label=u'Moving STD $\sigma_i$',zorder=1)
            else:
                plt.title("Gaussian Process Regression(Cubic)")
            plt.plot(x, y_pred, '%s-'%colors[count], zorder=5)#label=u'Prediction',zorder=5)
            plt.fill(np.concatenate([x, x[::-1]]),
                    np.concatenate([y_pred - 1.9600 * sigma,
                                   (y_pred + 1.9600 * sigma)[::-1]]),
                    alpha=.25, fc=colors[count], ec='None', zorder=2)#)label='95% Confidence Interval',zorder=2)
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
