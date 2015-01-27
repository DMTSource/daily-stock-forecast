# Copyright 2012 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Google Compute Engine helper class.

Use this class to:
- Start an instance
- List instances
- Delete an instance
"""

__author__ = 'kbrisbin@google.com (Kathryn Hurley)'

import logging
try:
  import simplejson as json
except:
  import json
import time
import traceback

from apiclient.discovery import build
from apiclient.errors import HttpError
from httplib2 import HttpLib2Error
from oauth2client.client import AccessTokenRefreshError

SETTINGS_FILE = 'settings.json'
DISK_TYPE = 'PERSISTENT'


class Gce(object):
  """Demonstrates some of the image and instance API functionality.

  Attributes:
    settings: A dictionary of application settings from the settings.json file.
    service: An apiclient.discovery.Resource object for Compute Engine.
    project_id: The string Compute Engine project ID.
    project_url: The string URL of the Compute Engine project.
  """

  def __init__(self, auth_http, project_id=None):
    """Initialize the Gce object.

    Args:
      auth_http: an authorized instance of httplib2.Http.
      project_id: the API console project name.
    """

    self.settings = json.loads(open(SETTINGS_FILE, 'r').read())

    self.service = build(
        'compute', self.settings['compute']['api_version'], http=auth_http)

    self.gce_url = 'https://www.googleapis.com/compute/%s/projects' % (
        self.settings['compute']['api_version'])

    self.project_id = None
    if not project_id:
      self.project_id = self.settings['project']
    else:
      self.project_id = project_id
    self.project_url = '%s/%s' % (self.gce_url, self.project_id)

  def start_instance(self,
                     instance_name,
                     disk_name,
                     zone=None,
                     machine_type=None,
                     network=None,
                     service_email=None,
                     scopes=None,
                     metadata=None,
                     startup_script=None,
                     startup_script_url=None,
                     blocking=True):
    """Start an instance with the given name and settings.

    Args:
      instance_name: String name for instance.
      disk_name: The string disk name.
      zone: The string zone name.
      machine_type: The string machine type.
      network: The string network.
      service_email: The string service email.
      scopes: List of string scopes.
      metadata: List of metadata dictionaries.
      startup_script: The filename of a startup script.
      startup_script_url: Url of a startup script.
      blocking: Whether the function will wait for the operation to complete.

    Returns:
      Dictionary response representing the operation.

    Raises:
      ApiOperationError: Operation contains an error message.
      DiskDoesNotExistError: Disk to be used for instance boot does not exist.
      ValueError: Either instance_name is None an empty string or disk_name
          is None or an empty string.
    """

    if not instance_name:
      raise ValueError('instance_name required.')

    if not disk_name:
      raise ValueError('disk_name required.')

    # Instance dictionary is sent in the body of the API request.
    instance = {}

    # Set required instance fields with defaults if not provided.
    instance['name'] = instance_name
    if not zone:
      zone = self.settings['compute']['zone']
    if not machine_type:
      machine_type = self.settings['compute']['machine_type']
    instance['machineType'] = '%s/zones/%s/machineTypes/%s' % (
        self.project_url, zone, machine_type)
    if not network:
      network = self.settings['compute']['network']
    instance['networkInterfaces'] = [{
        'accessConfigs': [{'type': 'ONE_TO_ONE_NAT', 'name': 'External NAT'}],
        'network': '%s/global/networks/%s' % (self.project_url, network)}]

    # Make sure the disk exists, and apply disk to instance resource.
    disk_exists = self.get_disk(disk_name)
    if not disk_exists:
      raise DiskDoesNotExistError(disk_name + ' disk must exist.')
    instance['disks'] = [{
        'source': '%s/zones/%s/disks/%s' % (self.project_url, zone, disk_name),
        'boot': True,
        'type': DISK_TYPE
    }]

    # Set optional fields with provided values.
    if service_email and scopes:
      instance['serviceAccounts'] = [{'email': service_email, 'scopes': scopes}]

    # Set the instance metadata if provided.
    instance['metadata'] = {}
    instance['metadata']['items'] = []
    if metadata:
      instance['metadata']['items'].extend(metadata)

    # Set the instance startup script if provided.
    if startup_script:
      startup_script_resource = {
          'key': 'startup-script', 'value': open(startup_script, 'r').read()}
      instance['metadata']['items'].append(startup_script_resource)

    # Set the instance startup script URL if provided.
    if startup_script_url:
      startup_script_url_resource = {
          'key': 'startup-script-url', 'value': startup_script_url}
      instance['metadata']['items'].append(startup_script_url_resource)

    # Send the request.
    request = self.service.instances().insert(
        project=self.project_id, zone=zone, body=instance)
    response = self._execute_request(request)
    if response and blocking:
      response = self._blocking_call(response)

    if response and 'error' in response:
      raise ApiOperationError(response['error']['errors'])

    return response

  def list_instances(self, zone=None, list_filter=None):
    """Lists project instances.

    Args:
      zone: The string zone name.
      list_filter: String filter for list query.

    Returns:
      List of instances matching given filter.
    """

    if not zone:
      zone = self.settings['compute']['zone']

    request = None
    if list_filter:
      request = self.service.instances().list(
          project=self.project_id, zone=zone, filter=list_filter)
    else:
      request = self.service.instances().list(
          project=self.project_id, zone=zone)
    response = self._execute_request(request)

    if response and 'items' in response:
      return response['items']
    return []

  def stop_instance(self,
                    instance_name,
                    zone=None,
                    blocking=True):
    """Stops an instance.

    Args:
      instance_name: String name for the instance.
      zone: The string zone name.
      blocking: Whether the function will wait for the operation to complete.

    Returns:
      Dictionary response representing the operation.

    Raises:
      ApiOperationError: Operation contains an error message.
      ValueError: instance_name is None or an empty string.
    """

    if not instance_name:
      raise ValueError('instance_name required.')

    if not zone:
      zone = self.settings['compute']['zone']

    # Delete the instance.
    request = self.service.instances().delete(
        project=self.project_id, zone=zone, instance=instance_name)
    response = self._execute_request(request)
    if response and blocking:
      response = self._blocking_call(response)

    if response and 'error' in response:
      raise ApiOperationError(response['error']['errors'])

    return response

  def create_disk(self,
                  disk_name,
                  image_project=None,
                  image=None,
                  zone=None,
                  blocking=True):
    """Creates a new persistent disk.

    Args:
      disk_name: String name for the disk.
      image_project: The string name for the project of the image.
      image: String name of the image to apply to the disk.
      zone: The string zone name.
      blocking: Whether the function will wait for the operation to complete.

    Returns:
      Dictionary response representing the operation.

    Raises:
      ApiOperationError: Operation contains an error message.
      ValueError: disk_name is None or an empty string.
    """

    if not disk_name:
      raise ValueError('disk_name required.')

    # Disk dictionary is sent in the body of the API request.
    disk = {}

    # Set required disk fields with defaults if not provided.
    disk['name'] = disk_name
    if not zone:
      zone = self.settings['compute']['zone']
    if not image_project:
      image_project = self.settings['compute']['image_project']
    if not image:
      image = self.settings['compute']['image']
    source_image = '%s/%s/global/images/%s' % (
        self.gce_url, image_project, image)

    request = self.service.disks().insert(
        project=self.project_id,
        zone=zone,
        sourceImage=source_image,
        body=disk)
    response = self._execute_request(request)
    if response and blocking:
      response = self._blocking_call(response)

    if response and 'error' in response:
      raise ApiOperationError(response['error']['errors'])

    return response

  def get_disk(self, disk_name, zone=None):
    """Gets the specified disk by name.

    Args:
      disk_name: The string name of the disk.
      zone: The string name of the zone.

    Returns:
      Dictionary response representing the disk or None if the disk
      does not exist.
    """

    if not zone:
      zone = self.settings['compute']['zone']

    request = self.service.disks().get(
        project=self.project_id, zone=zone, disk=disk_name)
    try:
      response = self._execute_request(request)
      return response
    except ApiError, e:
      return

  def delete_disk(self, disk_name, zone=None, blocking=True):
    """Deletes a disk.

    Args:
      disk_name: String name for the disk.
      zone: The string zone name.
      blocking: Whether the function will wait for the operation to complete.

    Returns:
      Dictionary response representing the operation.

    Raises:
      ApiOperationError: Operation contains an error message.
      ValueError: disk_name is None or an empty string.
    """

    if not disk_name:
      raise ValueError('disk_name required.')

    if not zone:
      zone = self.settings['compute']['zone']

    # Delete the disk.
    request = self.service.disks().delete(
        project=self.project_id, zone=zone, disk=disk_name)
    response = self._execute_request(request)
    if response and blocking:
      response = self._blocking_call(response)

    if response and 'error' in response:
      raise ApiOperationError(response['error']['errors'])

    return response

  def _blocking_call(self, response):
    """Blocks until the operation status is done for the given operation.

    Args:
      response: The response from the API call.

    Returns:
      Dictionary response representing the operation.
    """

    status = response['status']

    while status != 'DONE' and response:
      operation_id = response['name']
      if 'zone' in response:
        zone = response['zone'].rsplit('/', 1)[-1]
        request = self.service.zoneOperations().get(
            project=self.project_id, zone=zone, operation=operation_id)
      else:
        request = self.service.globalOperations().get(
            project=self.project_id, operation=operation_id)
      response = self._execute_request(request)
      if response:
        status = response['status']
        logging.info(
            'Waiting until operation is DONE. Current status: %s', status)
        if status != 'DONE':
          time.sleep(3)

    return response

  def _execute_request(self, request):
    """Helper method to execute API requests.

    Args:
      request: The API request to execute.

    Returns:
      Dictionary response representing the operation if successful.

    Raises:
      ApiError: Error occurred during API call.
    """

    try:
      response = request.execute()
    except AccessTokenRefreshError, e:
      logging.error('Access token is invalid.')
      raise ApiError(e)
    except HttpError, e:
      logging.error('Http response was not 2xx.')
      raise ApiError(e)
    except HttpLib2Error, e:
      logging.error('Transport error.')
      raise ApiError(e)
    except Exception, e:
      logging.error('Unexpected error occured.')
      traceback.print_stack()
      raise ApiError(e)

    return response


class Error(Exception):
  """Base class for exceptions in this module."""
  pass


class ApiError(Error):
  """Error occurred during API call."""
  pass


class ApiOperationError(Error):
  """Raised when an API operation contains an error."""

  def __init__(self, error_list):
    """Initialize the Error.

    Args:
      error_list: the list of errors from the operation.
    """

    super(ApiOperationError, self).__init__()
    self.error_list = error_list

  def __str__(self):
    """String representation of the error."""

    return repr(self.error_list)


class DiskDoesNotExistError(Error):
  """Disk to be used for instance boot does not exist."""
  pass
