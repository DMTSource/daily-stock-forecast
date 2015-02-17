#!/usr/bin/env python
#
# Copyright 2014 DMT SOURCE, LLC.
#
#
#     DMTSOURCE.COM | CONTACT: DEREK M TISLER lstrdean@gmail.com
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from lib.Forecast import Forecast
from lib.StockList import StockList

import numpy as np

import webapp2

def InjectTestData():
    #init a new forecast object, fill it with anything
    Forecast(
        rank                     = 1,
        symbol                   = "GOOG",
        company                  = "Google, Inc.",
        exchange                 = "NASDAQ",
        industry                 = "Technology",

        #predictions
        openPredPrice            = list(550.0+5.0*(np.random.ranf(11)-0.5)),
        closePredPrice           = list(550.0+5.0*(np.random.ranf(11)-0.5)),
        highPredPrice            = list(555.0+10.0*(np.random.ranf(11)-0.5)),
        lowPredPrice             = list(545.0+10.0*(np.random.ranf(11)-0.5)),
        volumePred               = list(np.random.randint(1850000,2150000, 11)),
        dayOfPred                = ['M','Tu','W','Th','F',
                                    'M','Tu','W','Th', 'F',
                                    'M','Tu','W','Th', 'F',
                                    'M','Tu','W','Th', 'F',
                                    'M','Tu','W','Th', 'F',
                                    'M','Tu','W','Th', 'F',
                                    'M'],

        #History lists
        openPriceHistory         = list(550.0+5.0*(np.random.ranf(10)-0.5)),
        closePriceHistory        = list(550.0+5.0*(np.random.ranf(10)-0.5)),
        highPriceHistory         = list(560.0+10.0*(np.random.ranf(10)-0.5)),
        lowPriceHistory          = list(540.0+10.0*(np.random.ranf(10)-0.5)),
        volumeHistory            = list(np.random.randint(1850000,2150000, 10)),

        #prediction validation value, R2
        openPredR2               = 1.0 + (np.random.ranf(1)[0] - 0.5)/10.0,
        closePredR2              = 1.0 + (np.random.ranf(1)[0] - 0.5)/10.0,
        highPredR2               = 1.0 + (np.random.ranf(1)[0] - 0.5)/10.0,
        lowPredR2                = 1.0 + (np.random.ranf(1)[0] - 0.5)/10.0,
        volumePredR2             = 1.0 + (np.random.ranf(1)[0] - 0.5)/10.0,

        openPredSlope            = 1.0 + (np.random.ranf(1)[0] - 0.5)/10.0,
        closePredSlope           = 1.0 + (np.random.ranf(1)[0] - 0.5)/10.0,
        highPredSlope            = 1.0 + (np.random.ranf(1)[0] - 0.5)/10.0,
        lowPredSlope             = 1.0 + (np.random.ranf(1)[0] - 0.5)/10.0,
        volumePredSlope          = 1.0 + (np.random.ranf(1)[0] - 0.5)/10.0,

        openModelAccuracy        = 1,
        closeModelAccuracy       = 2,
        highModelAccuracy        = 2,
        lowModelAccuracy         = 3
    ).put()
    
    StockList(
        rank                     = 1,
        symbol                   = "GOOG",
        company                  = "Google, Inc.",
        exchange                 = "NASDAQ",
        currentPrice             = 550.0+5.0*(np.random.ranf(1)[0]-0.5),
        forecastedPrice          = 550.0+5.0*(np.random.ranf(1)[0]-0.5),
        modelAccuracy            = np.random.randint(1,4,1)[0]
    ).put()
    #

class TestPage(webapp2.RequestHandler):
    def get(self):
        self.response.headers['Content-Type'] = 'text/plain'
        self.response.write('Injecting Test Data...')

        
        InjectTestData()
        
        self.redirect('./')

application = webapp2.WSGIApplication([
    ('/testData', TestPage),
], debug=True)
