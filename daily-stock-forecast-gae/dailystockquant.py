#!/usr/bin/env python
#
# Copyright 2014 DMT SOURCE, LLC.
#
#     DMTSOURCE.COM | CONTACT: DEREK M TISLER lstrdean@gmail.com
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import os
import urllib

from google.appengine.api import users
from google.appengine.ext import ndb
from google.appengine.ext.ndb import stats

import jinja2
import webapp2

import numpy as np

from pytz import timezone
from datetime import datetime, time
import time as tt

from lib.Forecast import Forecast
from lib.StockList import StockList
from lib.UserProfile import UserProfile

JINJA_ENVIRONMENT = jinja2.Environment(
    loader=jinja2.FileSystemLoader(os.path.dirname(__file__)),
    extensions=['jinja2.ext.autoescape'],
    autoescape=True)

class MainPage(webapp2.RequestHandler):

    def get(self):
        
        #Get total nubmer of predictions
        #global_stat = stats.GlobalStat.query().get()
        #forecastCount = global_stat.count

        #Get the time, make a string of format:
        #Tue, Jan 6, 2014, 12:00AM EST - US MARKETS CLOSED
        now = datetime.now(tz=timezone('US/Eastern'))
        #Construct the EST time for the top of page
        if( (now.time() >= time(9,30) and now.time() <= time(16,30)) and (now.weekday() <= 4 ) ):
            timeString = "{0:s} EST  - US Markets Are Open".format(now.strftime("%a, %b %d %Y, %I:%M%p"))
        else:
            timeString = "{0:s} EST  - US Markets Are Closed".format(now.strftime("%a, %b %d %Y, %I:%M%p"))
        #

        stockList = StockList.query(StockList.rank.IN(list(np.arange(1,26))))

        computedCloseValues = np.zeros((stockList.count(), 3), float)

        #createa  binary list of stock list vs user favorite list
        user = users.get_current_user()
        if user:
            up = UserProfile.query(UserProfile.user_id == str(user.user_id()))            
            favBinary = []
            for u in up:
                for stock in stockList:
                    if stock.symbol in u.favorite_list:
                        favBinary.append(1)
                    else:
                        favBinary.append(0)
                break
            #case where user exists(like admin) but no profile yet
            if stockList.count() != len(favBinary):
                favBinary = np.zeros((stockList.count(),))
        else:
            favBinary = np.zeros((stockList.count(),))
        #

        i = 0
        for stock in stockList:
            computedCloseValues[i][0] = stock.forecastedPrice-stock.currentPrice
            computedCloseValues[i][1] = (stock.forecastedPrice-stock.currentPrice)/abs(stock.currentPrice)*100.0
            computedCloseValues[i][2] = favBinary[i]
            i += 1

        #Init items using info from forecast, just use the first item
        dayOfForecast = now.strftime("%A, %B %d %Y")
        dof = now
        for stock in stockList:
            dayOfForecast = stock.date.strftime("%A, %B %d %Y")
            #dof = forecast.date
            break
        
        if users.get_current_user():
            url = users.create_logout_url(self.request.uri)
            url_linktext = 'Logout'
        else:
            url = users.create_login_url(self.request.uri)
            url_linktext = 'Sign Up for Updates'

        template_values = {
            'stock_list':stockList,
            'computed_values':computedCloseValues,
            #'forecast_count':forecastCount,
            'timeStr':timeString,
            'dayOfForecast':dayOfForecast,
            'url': url,
            'url_linktext': url_linktext,
        }

        template = JINJA_ENVIRONMENT.get_template('index.html')
        self.response.write(template.render(template_values))


class Markets(webapp2.RequestHandler):

    def get(self):
        
        #Get total nubmer of predictions
        #global_stat = stats.GlobalStat.query().get()
        #forecastCount = global_stat.count

        #Get the time, make a string of format:
        #Tue, Jan 6, 2014, 12:00AM EST - US MARKETS CLOSED
        now = datetime.now(tz=timezone('US/Eastern'))
        #Construct the EST time for the top of page
        if( (now.time() >= time(9,30) and now.time() <= time(16,30)) and (now.weekday() <= 4 ) ):
            timeString = "{0:s} EST  - US Markets Are Open".format(now.strftime("%a, %b %d %Y, %I:%M%p"))
        else:
            timeString = "{0:s} EST  - US Markets Are Closed".format(now.strftime("%a, %b %d %Y, %I:%M%p"))
        #

        stockList = StockList.query(StockList.rank.IN([1]))
        
        #Init items using info from forecast, just use the first item
        dayOfForecast = now.strftime("%A, %B %d %Y")
        dof = now
        for forecast in stockList:
            dayOfForecast = forecast.date.strftime("%A, %B %d %Y")
            dof = forecast.date
            break
        
        if users.get_current_user():
            url = users.create_logout_url(self.request.uri)
            url_linktext = 'Logout'
        else:
            url = users.create_login_url(self.request.uri)
            url_linktext = 'Sign Up for Updates'

        template_values = {
            #'stock_list':stockList,
            #'forecast_data':forecastPlotData,
            #'validation_data':validationPlotData,
            #'computed_values':computedValued,
            #'forecast_count':forecastCount,
            'timeStr':timeString,
            'dayOfForecast':dayOfForecast,
            'url': url,
            'url_linktext': url_linktext,
        }

        template = JINJA_ENVIRONMENT.get_template('markets.html')
        self.response.write(template.render(template_values))


class SymbolHandler(webapp2.RequestHandler):

    def get(self, stock_symbol):

        #Request the market and prepare its data for plotting
        
        #Reqest the stock and prepare its data for plotting
        #Get total nubmer of predictions
        #global_stat = stats.GlobalStat.query().get()
        #forecastCount = global_stat.count

        if stock_symbol == '':
            symbol_search = self.request.get("symbol_search")


        #Get the time, make a string of format:
        #Tue, Jan 6, 2014, 12:00AM EST - US MARKETS CLOSED
        now = datetime.now(tz=timezone('US/Eastern'))
        #Construct the EST time for the top of page
        if( (now.time() >= time(9,30) and now.time() <= time(16,30)) and (now.weekday() <= 4 ) ):
            timeString = "{0:s} EST  - US Markets Are Open".format(now.strftime("%a, %b %d %Y, %I:%M%p"))
        else:
            timeString = "{0:s} EST  - US Markets Are Closed".format(now.strftime("%a, %b %d %Y, %I:%M%p"))
        #


        #query to get the top 10 stocks for newest forecast round
        #stockList = Forecast.query(Forecast.rank.IN(list(np.arange(1,11))))
        stockList = Forecast.query(Forecast.symbol == stock_symbol.upper())

        #createa  binary list of stock list vs user favorite list
        user = users.get_current_user()
        if user:
            up = UserProfile.query(UserProfile.user_id == str(user.user_id()))            
            favBinary = []
            for u in up:
                for stock in stockList:
                    if stock.symbol in u.favorite_list:
                        favBinary.append(1)
                    else:
                        favBinary.append(0)
                break
            #case where user exists(like admin) but no profile yet
            if stockList.count() != len(favBinary):
                favBinary = np.zeros((stockList.count(),))
        else:
            favBinary = np.zeros((stockList.count(),))
        #

        
        #3d array of the candlestick plots
        # stock, list of L, list of O, list of C, list of H, list of V
        #stocks, history, category
        forecastPlotData = np.zeros((stockList.count(), 10, 6), object)
        #3d array of the validation plots
        #Stocks, history, category
        validationPlotData = []
        #3d array of computed values nstock, 10
        computedValued = np.zeros((stockList.count(), 11), float)

        #Init items using info from forecast, just use the first item
        dayOfForecast = now.strftime("%A, %B %d %Y")
        dof = now
        for forecast in stockList:
            validationPlotData = np.zeros((stockList.count(), len(forecast.lowPriceHistory), 10), object)
            dayOfForecast = forecast.date.strftime("%A, %B %d %Y")
            dof = forecast.date
            break

        i = 0
        for forecast in stockList:
            forecastPlotData[i,:,0] = [x.encode('utf-8').replace("'","") for x in forecast.dayOfPred[-10:]]
            #forecastPlotData[i,-1,0]   = str(forecast.dayOfPred[-1]).replace("'","")#.encode('utf-8')#.replace('&#39;','')
            forecastPlotData[i,:-1,1] = forecast.lowPriceHistory[-9:]
            forecastPlotData[i,-1,1]   = forecast.lowPredPrice[-1]
            forecastPlotData[i,:-1,2] = forecast.openPriceHistory[-9:]
            forecastPlotData[i,-1,2]   = forecast.openPredPrice[-1]
            forecastPlotData[i,:-1,3] = forecast.closePriceHistory[-9:]
            forecastPlotData[i,-1,3]   = forecast.closePredPrice[-1]
            forecastPlotData[i,:-1,4] = forecast.highPriceHistory[-9:]
            forecastPlotData[i,-1,4]   = forecast.highPredPrice[-1]
            forecastPlotData[i,:-1,5] = forecast.volumeHistory[-9:]
            forecastPlotData[i,-1,5]   = forecast.volumePred[-1]

            validationPlotData[i,:,0] = forecast.openPriceHistory
            validationPlotData[i,:,1] = forecast.openPredPrice[:-1]
            validationPlotData[i,:,2] = forecast.closePriceHistory
            validationPlotData[i,:,3] = forecast.closePredPrice[:-1]
            validationPlotData[i,:,4] = forecast.highPriceHistory
            validationPlotData[i,:,5] = forecast.highPredPrice[:-1]
            validationPlotData[i,:,6] = forecast.lowPriceHistory
            validationPlotData[i,:,7] = forecast.lowPredPrice[:-1]
            validationPlotData[i,:,8] = forecast.volumeHistory
            validationPlotData[i,:,9] = forecast.volumePred[:-1]

            computedValued[i][0] = forecast.openPredPrice[-1]-forecast.openPriceHistory[-1]
            computedValued[i][1] = (forecast.openPredPrice[-1]-forecast.openPriceHistory[-1])/abs(forecast.openPriceHistory[-1])*100.0
            computedValued[i][2] = forecast.closePredPrice[-1]-forecast.closePriceHistory[-1]
            computedValued[i][3] =(forecast.closePredPrice[-1]-forecast.closePriceHistory[-1])/abs(forecast.closePriceHistory[-1])*100.0
            computedValued[i][4] = forecast.highPredPrice[-1]-forecast.highPriceHistory[-1]
            computedValued[i][5] =(forecast.highPredPrice[-1]-forecast.highPriceHistory[-1])/abs(forecast.highPriceHistory[-1])*100.0
            computedValued[i][6] = forecast.lowPredPrice[-1]-forecast.lowPriceHistory[-1]
            computedValued[i][7] =(forecast.lowPredPrice[-1]-forecast.lowPriceHistory[-1])/abs(forecast.lowPriceHistory[-1])*100.0
            computedValued[i][8] = forecast.volumePred[-1]-forecast.volumeHistory[-1]
            computedValued[i][9] = (forecast.volumePred[-1]-forecast.volumeHistory[-1])/abs(forecast.volumeHistory[-1])*100.0
            computedValued[i][10] = favBinary[i]
            #Count for filling arrays
            i += 1
        
        if users.get_current_user():
            url = users.create_logout_url(self.request.uri)
            url_linktext = 'Logout'
        else:
            url = users.create_login_url(self.request.uri)
            url_linktext = 'Sign Up for Updates'

        template_values = {
            'stock_list':stockList,
            'forecast_data':forecastPlotData,
            'validation_data':validationPlotData,
            'computed_values':computedValued,
            #'forecast_count':forecastCount,
            'timeStr':timeString,
            'dayOfForecast':dayOfForecast,
            'url': url,
            'url_linktext': url_linktext,
        }

        template = JINJA_ENVIRONMENT.get_template('symbol.html')
        self.response.write(template.render(template_values))

class FavoiteHandler(webapp2.RequestHandler):

    def get(self, stock_symbol):

        #make symbol all caps
        stock_symbol = stock_symbol.upper() 
        
        # Checks for active Google account session
        user = users.get_current_user()

        #Only commit to ndb if we need to
        need_to_commit = False
        
        #Do we have a loged in user
        if user:

            #Get the users list of favorite stocks
            up = UserProfile.query(UserProfile.user_id == str(user.user_id()))
            
            # If new user, make profile
            if up.count() == 0:
                UserProfile(nickname           = str(user.nickname()),
                                 email              = str(user.email()),
                                 user_id            = str(user.user_id()),
                                 federated_identity = str(user.federated_identity()),
                                 federated_provider = str(user.federated_provider()),
                                 favorite_list      = []).put()
                up = UserProfile.query(UserProfile.user_id == str(user.user_id()))
                need_to_commit = True
            else:
                for u in up:
                    #Update last login date
                    u.last_login_date = datetime.now()
                    #update user email if changed
                    if user.email != u.email:
                        u.email = str(user.email())
                        need_to_commit = True
                
            
            #optinal:Remove the stock if its in the list, otherwise add it
            for u in up:
                if stock_symbol != '':
                    need_to_commit = True
                    if stock_symbol in u.favorite_list:
                        u.favorite_list.remove(stock_symbol)
                    else:
                        u.favorite_list.append(stock_symbol)

            #commit the changes
            if need_to_commit:
                for u in up:
                    u.put()

            #return to where the user was
            if stock_symbol != '':
                try:
                    self.redirect(self.request.referer)
                except:
                    self.redirect('./')
        #no user
        else:
            #A non user tried to favorite something, lets log them in so they get the action still after login
            if stock_symbol != '':
                self.redirect(users.create_login_url(self.request.uri))

        #Form the symbol list to query
        queryList = []
        if user:
            for u in up:
                for item in u.favorite_list:
                    queryList.append(item)
            
                
        #Get the time, make a string of format:
        #Tue, Jan 6, 2014, 12:00AM EST - US MARKETS CLOSED
        now = datetime.now(tz=timezone('US/Eastern'))
        #Construct the EST time for the top of page
        if( (now.time() >= time(9,30) and now.time() <= time(16,30)) and (now.weekday() <= 4 ) ):
            timeString = "{0:s} EST  - US Markets Are Open".format(now.strftime("%a, %b %d %Y, %I:%M%p"))
        else:
            timeString = "{0:s} EST  - US Markets Are Closed".format(now.strftime("%a, %b %d %Y, %I:%M%p"))
        #

        stockList = StockList.query(StockList.symbol.IN(queryList)).order(StockList.rank)

        #prevent empty query from causing crashes
        if len(queryList)==0:
            stockList = []

        #Get computed values
        if len(queryList)!=0:
            computedCloseValues = np.zeros((stockList.count(), 2), float)
        else:
            computedCloseValues = np.zeros((0, 2), float)
        i = 0
        if len(queryList)!=0:
            for stock in stockList:
                computedCloseValues[i][0] = stock.forecastedPrice-stock.currentPrice
                computedCloseValues[i][1] = (stock.forecastedPrice-stock.currentPrice)/abs(stock.currentPrice)*100.0
                i += 1

        #Init items using info from forecast, just use the first item
        dayOfForecast = now.strftime("%A, %B %d %Y")
        dof = now
        if stock_symbol != '':
            for stock in stockList:
                dayOfForecast = stock.date.strftime("%A, %B %d %Y")
                #dof = forecast.date
                break
        #Form the login/logout url and a name to id the state in jinja2
        if user:
            url = users.create_logout_url(self.request.uri)
            url_linktext = 'Logout'
        else:
            url = users.create_login_url(self.request.uri)
            url_linktext = 'Login'

        template_values = {
            'stock_list':stockList,
            'computed_values':computedCloseValues,
            #'forecast_count':forecastCount,
            'timeStr':timeString,
            'dayOfForecast':dayOfForecast,
            'url': url,
            'url_linktext': url_linktext,
        }

        #Show mystock page
        template = JINJA_ENVIRONMENT.get_template('mystocks.html')
        self.response.write(template.render(template_values))

app = webapp2.WSGIApplication([
    ('/', MainPage),
    ('/markets', Markets),
    ('/symbol/(.*)', SymbolHandler),
    ('/favorite/(.*)', FavoiteHandler)
    
], debug=True)
