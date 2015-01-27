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

from google.appengine.ext import ndb

class Forecast(ndb.Model):

    def roundValue(self, value):
        return round(value,3)

    """ The purpose of the Forecast class is to store each stocks forecast
        for each day. This offsets the request burden since we are only
        storing float data, estimated 65 million items per year."""

    #Indexed items, dat, symbol, and company name(for search)
    date                     = ndb.DateTimeProperty(auto_now_add=True)
    rank                     = ndb.IntegerProperty()
    symbol                   = ndb.StringProperty(indexed=True) 
    company                  = ndb.StringProperty(indexed=True)
    exchange                 = ndb.StringProperty(indexed=True, choices=set(["NASDAQ", "NYSE", "AMEX"]))
    sector                 = ndb.StringProperty(indexed=False)
    #industry                 = ndb.StringProperty(indexed=True, choices=set(["Basic Industries", "Capital Goods", "Consumer Durables", "Consumer Non-Durables", "Consumer Services", "Energy", "Finance", "Healthcare", "Miscellaneous", "Public Utilities", "Technology", "Transportation"]))
    industry                 = ndb.StringProperty(indexed=True)
    

    #predictions(list of past predictions, todays is last item)
    openPredPrice            = ndb.FloatProperty(repeated=True,validator=roundValue)
    closePredPrice           = ndb.FloatProperty(repeated=True,validator=roundValue)
    highPredPrice            = ndb.FloatProperty(repeated=True,validator=roundValue)
    lowPredPrice             = ndb.FloatProperty(repeated=True,validator=roundValue)
    volumePred               = ndb.FloatProperty(repeated=True,validator=roundValue)
    dayOfPred                = ndb.StringProperty(repeated=True)
    
    #History lists(remember pred lists are n+1 of history lists when plotting)
    openPriceHistory         = ndb.FloatProperty(repeated=True,validator=roundValue)
    closePriceHistory        = ndb.FloatProperty(repeated=True,validator=roundValue)
    highPriceHistory         = ndb.FloatProperty(repeated=True,validator=roundValue)
    lowPriceHistory          = ndb.FloatProperty(repeated=True,validator=roundValue)
    volumeHistory            = ndb.FloatProperty(repeated=True)
#    dayOfWeekHistory         = ndb.StringProperty(repeated=True)

    #prediction validation value, R2
    openPredR2               = ndb.FloatProperty(validator=roundValue)
    closePredR2              = ndb.FloatProperty(validator=roundValue)
    highPredR2               = ndb.FloatProperty(validator=roundValue)
    lowPredR2                = ndb.FloatProperty(validator=roundValue)
    volumePredR2             = ndb.FloatProperty(validator=roundValue)

    openPredSlope            = ndb.FloatProperty(validator=roundValue)
    closePredSlope           = ndb.FloatProperty(validator=roundValue)
    highPredSlope            = ndb.FloatProperty(validator=roundValue)
    lowPredSlope             = ndb.FloatProperty(validator=roundValue)
    volumePredSlope          = ndb.FloatProperty(validator=roundValue)

    """#computed values(check docs for better cv)
    
    openPriceChange          = ndb.FloatProperty(validator=roundValue)
    openPriceChangePercent   = ndb.FloatProperty(validator=roundValue)
    closePriceChange         = ndb.FloatProperty(validator=roundValue)
    closePriceChangePercent  = ndb.FloatProperty(validator=roundValue)
    highPriceChange          = ndb.FloatProperty(validator=roundValue)
    highPriceChangePercent   = ndb.FloatProperty(validator=roundValue)
    lowPriceChange           = ndb.FloatProperty(validator=roundValue)
    lowPriceChangePercent    = ndb.FloatProperty(validator=roundValue)
    volumeChange             = ndb.FloatProperty(validator=roundValue)
    volumeChangePercent      = ndb.FloatProperty(validator=roundValue)"""

