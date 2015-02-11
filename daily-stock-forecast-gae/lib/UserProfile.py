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

class UserProfile(ndb.Model):

    def roundValue(self, value):
        return round(value,3)
    
    #https://cloud.google.com/appengine/docs/python/users/userclass#User_federated_provider
    create_date         = ndb.DateTimeProperty(auto_now_add=True)
    last_login_date     = ndb.DateTimeProperty(auto_now_add=True)

    """Returns the "nickname" of the user, a displayable name.
    For Google Accounts users, the nickname is either the
    "name" portion of the user's email address if the
    address is in the same domain as the application, or the
    user's full email address otherwise. For OpenID users,
    the nickname is the OpenID identifier."""
    nickname            = ndb.StringProperty()

    """
    Returns the email address of the user. If you use OpenID,
    you should not rely on this email address to be correct.
    Applications should use nickname for displayable names.
    """
    email               = ndb.StringProperty()

    """
    If the email address is associated with a Google account,
    user_id returns the unique permanent ID of the user, a str.
    This ID is always the same for the user regardless of whether
    the user changes her email address.

    If the email address is not associated with a Google account,
    user_id returns None.

    Note: If your application constructs the User instance, the API
    will not set a value for user_id and it returns None.
    """
    user_id             = ndb.StringProperty()

    """
    Returns the user's OpenID identifier.
    """
    federated_identity  = ndb.StringProperty()
    """
    Returns the URL of the user's OpenID provider.
    """
    federated_provider  = ndb.StringProperty()

    """
    String list of stock symbols the user has favorited
    """
    favorite_list       = ndb.StringProperty(repeated=True)


