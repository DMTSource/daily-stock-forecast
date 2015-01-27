import logging

import webapp2
import json
import gce
from apiclient.discovery import build
from oauth2client.appengine import OAuth2Decorator

import httplib2
from google.appengine.api import memcache
from oauth2client.appengine import AppAssertionCredentials

settings = json.loads(open(gce.SETTINGS_FILE, 'r').read())

"""decorator = OAuth2Decorator(
    client_id='446253548147-kaaii57iu0ki3jtq5atqgcvrp0qifste.apps.googleusercontent.com',
    client_secret='2QYM3_NbQH4LYVLMGuM6ZqME',
    scope=settings['compute_scope'])"""
credentials = AppAssertionCredentials(scope='https://www.googleapis.com/auth/compute')

bucket='auto-test'
image_url = 'http://storage.googleapis.com/gce-demo-input/photo.jpg'
image_text = 'Ready for dessert?'
INSTANCE_NAME = 'daily-forecast'
DISK_NAME = INSTANCE_NAME + '-disk'
INSERT_ERROR = 'Error inserting %(name)s.'
DELETE_ERROR = """
Error deleting %(name)s. %(name)s might still exist; You can use
the console (http://cloud.google.com/console) to delete %(name)s.
"""

def delete_resource(delete_method, *args):
  """Delete a Compute Engine resource using the supplied method and args.

  Args:
    delete_method: The gce.Gce method for deleting the resource.
  """

  resource_name = args[0]
  logging.info('Deleting %s' % resource_name)

  try:
    delete_method(*args)
  except (gce.ApiError, gce.ApiOperationError, ValueError) as e:
    logging.error(DELETE_ERROR, {'name': resource_name})
    logging.error(e)

class MainPage (webapp2.RequestHandler):
    #@decorator.oauth_required
    def get(self):
        #self.response.headers['Content-Type'] = 'text/plain'
        
        """# Get the authorized Http object created by the decorator.
        http = decorator.http()"""
        http = credentials.authorize(httplib2.Http(memcache))

        # Initialize gce.Gce.
        gce_helper = gce.Gce(http, project_id=settings['project'])

        # Create a Persistent Disk (PD), which is used as a boot disk.
        try:
          gce_helper.create_disk(DISK_NAME)
        except (gce.ApiError, gce.ApiOperationError, ValueError, Exception) as e:
          logging.error(INSERT_ERROR, {'name': DISK_NAME})
          logging.error(e)
          return

        # Start an instance with a local start-up script and boot disk.
        logging.info('Starting GCE instance')
        try:
          gce_helper.start_instance(
              INSTANCE_NAME,
              DISK_NAME,
              service_email=settings['compute']['service_email'],
              scopes=settings['compute']['scopes'],
              startup_script='startup.sh',
              metadata=[
                  {'key': 'url', 'value': image_url},
                  {'key': 'text', 'value': image_text},
                  {'key': 'cs-bucket', 'value': bucket}])
        except (gce.ApiError, gce.ApiOperationError, ValueError, Exception) as e:
          # Delete the disk in case the instance fails to start.
          delete_resource(gce_helper.delete_disk, DISK_NAME)
          logging.error(INSERT_ERROR, {'name': INSTANCE_NAME})
          logging.error(e)
          return
        except gce.DiskDoesNotExistError as e:
          logging.error(INSERT_ERROR, {'name': INSTANCE_NAME})
          logging.error(e)
          return
        """
        # List all running instances.
        logging.info('These are your running instances:')
        instances = gce_helper.list_instances()
        for instance in instances:
          logging.info(instance['name'])

        # Stop the instance.
        delete_resource(gce_helper.stop_instance, INSTANCE_NAME)

        # Delete the disk.
        delete_resource(gce_helper.delete_disk, DISK_NAME)
        """
        
        #self.response.out.write('Job Complete')

app = webapp2.WSGIApplication([
    ('/forecast', MainPage),
    #(decorator.callback_path, decorator.callback_handler())
], debug=True)

def main():

    try:
        #wsgiref.handlers.CGIHandler().run(app)
        run_wsgi_app(app)
    except:
        logging.error("Failed to run script forecast.py as cron job")

if __name__ == '__main__':
    main()

