import httplib2, webapp2
from oauth2client.appengine import AppAssertionCredentials
from apiclient.discovery import build
import time

http = AppAssertionCredentials('https://www.googleapis.com/auth/prediction').authorize(httplib2.Http())
service = build('prediction', 'v1.6', http=http)
  
class MakePrediction(webapp2.RequestHandler):
  def get(self):


    #Insert training data
    result = service.trainedmodels().insert(project='daily-stock-forecast', 
                                            body={'id': 'goog_reg_close_1', 'storageDataLocation': 'auto-test/goog_reg_close.txt'}
                                            ).execute()

    
    #get the model, we need to wait for training to finish(exponential Request Retry Intervals)
    count = 0.0
    while service.trainedmodels().get(id='goog_reg_close_1')['trainingStatus'] == 'DONE':
      time.sleep (100.0*count / 1000.0);
      count += 1.0

    #Make a prediction on the trained model
    result = service.trainedmodels().predict(project='daily-stock-forecast', 
                                             id='goog_reg_close_1', 
                                             body={'input': {'csvInstance': ['550.00']}}
                                             ).execute()

    self.response.headers['Content-Type'] = 'text/plain'
    self.response.out.write('Result: ' + repr(result))

app = webapp2.WSGIApplication([
    ('/makePrediction', MakePrediction),
], debug=True)