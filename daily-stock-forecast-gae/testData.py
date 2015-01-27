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

from lib.Forecast import Forecast

import numpy as np

import webapp2

def InjectTestData():

    #init a new forecast object, fill it with anything
    newForcast = Forecast(
    rank                     = 1,
    symbol                   = "Goog",
    company                  = "Google, Inc.",
    exchange                 = "NASDAQ",
    industry                 = "Technology",

    #predictions
    openPredPrice            = list(550.0+5.0*(np.random.ranf(31)-0.5)),
    closePredPrice           = list(550.0+5.0*(np.random.ranf(31)-0.5)),
    highPredPrice            = list(555.0+10.0*(np.random.ranf(31)-0.5)),
    lowPredPrice             = list(545.0+10.0*(np.random.ranf(31)-0.5)),
    volumePred               = list(np.random.randint(1850000,2150000, 31)),
    dayOfPred                = ['M','Tu','W','Th','F',
                                'M','Tu','W','Th', 'F',
                                'M','Tu','W','Th', 'F',
                                'M','Tu','W','Th', 'F',
                                'M','Tu','W','Th', 'F',
                                'M','Tu','W','Th', 'F',
                                'M'],

    #History lists
    openPriceHistory         = list(550.0+5.0*(np.random.ranf(30)-0.5)),
    closePriceHistory        = list(550.0+5.0*(np.random.ranf(30)-0.5)),
    highPriceHistory         = list(560.0+10.0*(np.random.ranf(30)-0.5)),
    lowPriceHistory          = list(540.0+10.0*(np.random.ranf(30)-0.5)),
    volumeHistory            = list(np.random.randint(1850000,2150000, 30)),

    #prediction validation value, R2
    openPredR2               = 1.0 + (np.random.ranf(1)[0] - 0.5)/10.0,
    closePredR2              = 1.0 + (np.random.ranf(1)[0] - 0.5)/10.0,
    highPredR2               = 1.0 + (np.random.ranf(1)[0] - 0.5)/10.0,
    lowPredR2                = 1.0 + (np.random.ranf(1)[0] - 0.5)/10.0,
    volumePredR2             = 1.0 + (np.random.ranf(1)[0] - 0.5)/10.0,

    openPredSlope               = 1.0 + (np.random.ranf(1)[0] - 0.5)/10.0,
    closePredSlope              = 1.0 + (np.random.ranf(1)[0] - 0.5)/10.0,
    highPredSlope               = 1.0 + (np.random.ranf(1)[0] - 0.5)/10.0,
    lowPredSlope                = 1.0 + (np.random.ranf(1)[0] - 0.5)/10.0,
    volumePredSlope             = 1.0 + (np.random.ranf(1)[0] - 0.5)/10.0,

    )

    """#computed values(check docs for better cv)
    
    newForcast.openPriceChange          = newForcast.openPredPrice[-1] - newForcast.openPriceHistory[-1]
    newForcast.openPriceChangePercent   = (newForcast.openPredPrice[-1] - newForcast.openPriceHistory[-1])/abs(newForcast.openPriceHistory[-1])*100.0
    newForcast.closePriceChange         = newForcast.closePredPrice[-1] - newForcast.closePriceHistory[-1]
    newForcast.closePriceChangePercent  = (newForcast.closePredPrice[-1] - newForcast.closePriceHistory[-1])/abs(newForcast.closePriceHistory[-1])*100.0
    newForcast.highPriceChange          = newForcast.highPredPrice[-1] - newForcast.highPriceHistory[-1]
    newForcast.highPriceChangePercent   = (newForcast.highPredPrice[-1] - newForcast.highPriceHistory[-1])/abs(newForcast.highPriceHistory[-1])*100.0
    newForcast.lowPriceChange           = newForcast.lowPredPrice[-1] - newForcast.lowPriceHistory[-1]
    newForcast.lowPriceChangePercent    = (newForcast.lowPredPrice[-1] - newForcast.lowPriceHistory[-1])/abs(newForcast.lowPriceHistory[-1])*100.0
    newForcast.volumeChange             = newForcast.volumePred[-1] - newForcast.volumeHistory[-1]
    newForcast.volumeChangePercent      = (newForcast.volumePred[-1] - newForcast.volumeHistory[-1])/abs(newForcast.volumeHistory[-1])*100.0

    #Market snapshot
    newForcast.predOpen                 = 522.01
    newForcast.predClose                = 521.76
    newForcast.predHigh                 = 553.41
    newForcast.predLow                  = 515.64
    newForcast.predVolume               = 2345816"""
    """newForcast.marketCap                = "325.75B"
    newForcast.avgVolume                = 2125468
    newForcast.beta                     = 2.14
    newForcast.priceEarningRatio        = 27.17
    newForcast.earningsPerShare         = 19.00"""

    #enter the new data
    newForcast.put()

    #self.redirect('./')

class TestPage(webapp2.RequestHandler):
    def get(self):
        self.response.headers['Content-Type'] = 'text/plain'
        self.response.write('Injecting Test Data...')

        InjectTestData()

application = webapp2.WSGIApplication([
    ('/testData', TestPage),
], debug=True)
