#!/usr/bin/env python
import os
import webapp2
from google.appengine.ext.webapp import template
from google.appengine.ext import db
import json
import logging 
import datetime

class ArtistsForUser(db.Model):

    lastfm_user = db.StringProperty()
    fb_user = db.StringProperty()

    full_name = db.StringProperty()
    artists = db.StringListProperty(required=True)

    created_at = db.DateTimeProperty(auto_now_add=True)
    updated_at = db.DateTimeProperty(auto_now=True)

class MainHandler(webapp2.RequestHandler):
    def get(self):
        params = dict()
        path = os.path.join(os.path.dirname(__file__), 'templates', 'index.html')
        result = template.render(path, params)
        self.response.out.write(result)

class GetArtistsForUsersHandler(webapp2.RequestHandler):
    def get(self):
        afus = ArtistsForUser.all().order('-created_at').fetch(10)
        afus_dict = []
        for afu in afus:
            name = afu.full_name if afu.full_name else afu.lastfm_user if afu.lastfm_user else "unknown"
            afus_dict.append({'name': name,
                         'artists': afu.artists,
                         'fb_user': afu.fb_user,
                         'created_at': afu.created_at.isoformat(),
                         'key': afu.key().id()})

        self.response.out.write(json.dumps(afus_dict))#,default=lambda obj: obj.isoformat() if isinstance(obj, datetime.datetime) else obj))

class StoreArtistsForUserHandler(webapp2.RequestHandler):
    def post(self):

        # Store the shit
        artists = json.loads(self.request.get('artists'))
        fb_user = self.request.get('fb_user')
        lastfm_user = self.request.get('lastfm_user')
        full_name = self.request.get('full_name')

        if not (type(artists)==list):
            v = {'success': False}
        else:            

            logging.info(artists)

            # Store the fukekr
            afu_obj = ArtistsForUser(fb_user=fb_user, 
                                     lastfm_user=lastfm_user,
                                     artists=artists, 
                                     full_name=full_name)
            afu_obj.put()

            v = {'success': True}
        
        self.response.out.write(json.dumps(v))

app = webapp2.WSGIApplication([
    ('/', MainHandler),
    ('/store_artists_for_user', StoreArtistsForUserHandler),
    ('/get_artists_for_users', GetArtistsForUsersHandler)
], debug=True)
