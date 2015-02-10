#!/usr/bin/env python
#
# Copyright 2014 DMT SOURCE, LLC.
#
#     DMTSOURCE.COM | CONTACT: DEREK M TISLER lstrdean@gmail.com
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

from google.appengine.ext import ndb

class StockList(ndb.Model):

    def roundValue(self, value):
        return round(value,3)

    #Indexed items, dat, symbol, and company name(for search)
    date                     = ndb.DateTimeProperty(auto_now_add=True)
    rank                     = ndb.IntegerProperty()
    symbol                   = ndb.StringProperty(indexed=True) 
    company                  = ndb.StringProperty(indexed=True)
    exchange                 = ndb.StringProperty(indexed=True, choices=set(["NASDAQ", "NYSE", "AMEX"]))
    
    currentPrice             = ndb.FloatProperty(validator=roundValue)

    forecastedPrice          = ndb.FloatProperty(validator=roundValue)

    modelAccuracy            = ndb.IntegerProperty()

