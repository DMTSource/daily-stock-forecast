#!/usr/bin/env python
#
# Copyright 2014 DMT SOURCE, LLC.
#
# This sytem is designed to perform the TishlerHarper tensor difference analysis
# on a dataset of gaussian outout files intended for structual analysis
#
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

JINJA_ENVIRONMENT = jinja2.Environment(
    loader=jinja2.FileSystemLoader(os.path.dirname(__file__)),
    extensions=['jinja2.ext.autoescape'],
    autoescape=True)

class MainPage(webapp2.RequestHandler):

    def get(self):
        
        """list_of_entities = Forecast.query(Forecast.rank < 6000)
        list_of_keys = ndb.put_multi(list_of_entities)
        list_of_entities = ndb.get_multi(list_of_keys)
        ndb.delete_multi(list_of_keys)"""
        #Get total nubmer of predictions
        global_stat = stats.GlobalStat.query().get()
        forecastCount = global_stat.count

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
        stockList = Forecast.query(Forecast.rank.IN(list(np.arange(1,11))))

        #3d array of the candlestick plots
        # stock, list of L, list of O, list of C, list of H, list of V
        #stocks, history, category
        forecastPlotData = np.zeros((stockList.count(), 10, 6), object)
        #3d array of the validation plots
        #Stocks, history, category
        validationPlotData = []
        #3d array of computed values nstock, 10
        computedValued = np.zeros((stockList.count(), 10), float)

        #Init items using info from forecast, just use the first item
        """dayOfForecast = now.strftime("%d/%m/%Y")
        dof = now
        for forecast in stockList:
            validationPlotData = np.zeros((stockList.count(), len(forecast.lowPriceHistory), 10), object)
            dayOfForecast = forecast.date.strftime("%m/%d/%Y")
            dof = forecast.date
            break
        if now.date() == dof.date():
            nameOfDayOfForecast = 'Today'
        else:
            nameOfDayOfForecast = dof.strftime("%a")"""
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
            #Count for filling arrays
            i += 1
        

        """
        guestbook_name = self.request.get('guestbook_name',
                                          DEFAULT_GUESTBOOK_NAME)
        greetings_query = Greeting.query(
            ancestor=guestbook_key(guestbook_name)).order(-Greeting.date)
        greetings = greetings_query.fetch(10)
        """
        
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
            'forecast_count':forecastCount,
            'timeStr':timeString,
            'dayOfForecast':dayOfForecast,
            #'nameOfDayOfForecast':nameOfDayOfForecast,
            'url': url,
            'url_linktext': url_linktext,
        }

        template = JINJA_ENVIRONMENT.get_template('index.html')
        self.response.write(template.render(template_values))


application = webapp2.WSGIApplication([
    ('/', MainPage),
    #('/sign', Guestbook),
], debug=True)
