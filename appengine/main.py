#!/usr/bin/env python
import os
import webapp2
from google.appengine.ext.webapp import template
from google.appengine.ext import db
import json
import logging 
import datetime

class ArtistsForUser(db.Model):
    user = db.StringProperty()
    artists = db.StringListProperty()
    created_at = db.DateTimeProperty(auto_now_add=True)
    updated_at = db.DateTimeProperty(auto_now=True)

class MainHandler(webapp2.RequestHandler):
    def get(self):
        v = dict()
        path = os.path.join(os.path.dirname(__file__), 'templates', 'index.html')
        result = template.render(path, v)
        self.response.out.write(result)

class GetArtistsForUsersHandler(webapp2.RequestHandler):
    def get(self):
        afus = ArtistsForUser.all().order('-created_at').fetch(10)
        afus_dict = []
        for afu in afus:
            afus_dict.append({'user_id': afu.user,
                         'artists': afu.artists,
                         'created_at': afu.created_at.isoformat()})

        self.response.out.write(json.dumps(afus_dict))#,default=lambda obj: obj.isoformat() if isinstance(obj, datetime.datetime) else obj))

class StoreArtistsForUserHandler(webapp2.RequestHandler):
    def post(self):

        # Store the shit
        artists = json.loads(self.request.get('artists'))
        user_id = self.request.get('user_id')

        if not (type(artists)==list):
            v = {'success': False}
        else:            

            logging.info(artists)
            logging.info(user_id)

            # Store the fukekr
            afu_obj = ArtistsForUser(user=user_id, artists=artists)
            afu_obj.put()

            v = {'success': True}
        
        self.response.out.write(json.dumps(v))

app = webapp2.WSGIApplication([
    ('/', MainHandler),
    ('/store_artists_for_user', StoreArtistsForUserHandler),
    ('/get_artists_for_users', GetArtistsForUsersHandler)
], debug=True)
