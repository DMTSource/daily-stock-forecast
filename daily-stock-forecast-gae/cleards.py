
import webapp2
from google.appengine.ext import ndb
import numpy as np

from lib.Forecast import Forecast

class MainPage (webapp2.RequestHandler):
    def get(self):
        self.response.headers['Content-Type'] = 'text/plain'
        #print np.arange(1,10001,1000)
        for i in np.arange(1,10001,1000):
            list_of_entities = Forecast.query(Forecast.rank < i)#Forecast.rank < 6000)
            list_of_keys = ndb.put_multi(list_of_entities)
            list_of_entities = ndb.get_multi(list_of_keys)
            ndb.delete_multi(list_of_keys)

        self.response.out.write('All Forecasts Deleted')

app = webapp2.WSGIApplication([
    ('/cleards', MainPage)
], debug=True)
