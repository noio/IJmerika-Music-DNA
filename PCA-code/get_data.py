#!/usr/bin/env python
# encoding: utf-8
"""
main.py

Created by Gilles de Hollander on 2012-09-30.
Copyright (c) 2012 __MyCompanyName__. All rights reserved.
"""

import sys
import os
import pylast
import numpy as np
import pickle as pkl
import random

# You have to have your own unique two values for API_KEY and API_SECRET
# Obtain yours from http://www.last.fm/api/account for Last.fm
API_KEY = "dfa3f081178624892ff5259be9bdc7c2" # this is a sample key
API_SECRET = "6d34c6ab655c33e4e46f1b59a946d6f3"

DB_FILE = 'db3.pkl'

base_artists = ['David Guetta', 'Rihanna', 'LMFAO', 'DJ Tiesto', 'David Guetta', 'Interpol', 'Alt-J', 'Franz Ferdinand', 'Muse']

if os.path.exists(DB_FILE):
    similarities = pkl.load(open(DB_FILE))
else:
    similarities = {}

# In order to perform a write operation you need to authenticate yourself
username = "jeboyG"
password_hash = pylast.md5("kutbum38")

network = pylast.LastFMNetwork(api_key = API_KEY, api_secret = 
    API_SECRET, username = username, password_hash = password_hash)


# result = np.zeros((len(base_artists), len(base_artists)))

# all_artists = [i.item.name for s in similarities.values() for i in s ]
# print 'Length before set: ' % len(all_artists)
# all_artists = list(set(all_artists))

# random.shuffle(all_artists)

for artist in base_artists:
    print artist
    if artist not in similarities.keys():
        print 'Fetching similarities of %s' % artist
        artist_object = network.get_artist(artist)
        similarities[artist] = artist_object.get_similar()    
        print 'Dumping %s' % similarities.keys()
    
        pkl.dump(similarities, open(DB_FILE, 'w'))
