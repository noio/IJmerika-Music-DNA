#!/usr/bin/env python
# encoding: utf-8

""" Main script for music-dna.
    Checks the website for new artist data,
    processes them (by projecting on the coeffiecients),
    saves an output image, and calls the printing function
"""

### IMPORTS ###

# Python
import urllib2
import json
from datetime import datetime, timedelta

# Libraries
from escpos import *

# Local
from musicdna import *
        
        
### SETTINGS ###

# Got this by using javascript: "$$('.tagList a').map(function(e) { return e.text })" on the last.fm Music page. 
LASTFM_FRONTPAGE_TAGS = ["ambient", "blues", "classical", "country", "electronic", "emo", "folk", "hardcore", "hip hop", "indie", "jazz", "latin", "metal", "pop", "pop punk", "punk", "reggae", "rnb", "rock", "soul", "world", "60s", "70s", "80s", "90s"]
SERVER_URL = 'http://localhost:8081/get_artists_for_users'


### MAIN PROCEDURE ###

if __name__ == '__main__':
    lastfm = LastFM('765932fff6d3d591f8635746fbaa1b7a')
    am = ArtistMatrix(lastfm)

    # CHECK IF THERE IS A COEFFICIENT MATRIX AVAILABLE
    try:
        am.load('_artistmatrix.pickle.gz')
        print "Found Artist Matrix data."
    # BUILD ONE OTHERWISE
    except IOError:
        print "Building new Artist Matrix."
        feature_artists = set([artist for tag in LASTFM_FRONTPAGE_TAGS for artist in lastfm.tag_gettopartists(tag, 50)])

        # PCA:
        sample_artists = lastfm.chart_gettopartists(100)
        am.set_projection_pca(feature_artists, sample_artists)
        am.find_distribution(sample_artists)
        
        am.dump('_artistmatrix.pickle.gz')
    
    # am.print_info()
    
    # INITIALIZE VIZUALIZER AND PRINTER OBJECT
    
    epson = printer.Usb(0x04b8,0x0202)
    viz = Visualizer(epson)
    
    while True:
        done = set()
        now = datetime.now()

        # CHECK FOR NEW USER DATA ONLINE
        data = json.loads(urllib2.urlopen(SERVER_URL).read())
        for entry in data:
            date = datetime.strptime(entry['created_at'], '%Y-%m-%dT%H:%M:%S.%f')
            if (entry['key'] not in done and 
                (now - date) < timedelta(hours=10)):
                        
                # CLEAN DATA
                artists = entry['artists']
                print artists
                
                # PROJECT ARTISTS ONTO FEATURES
                data = am.redistribute(am.project(artists))
                
                # SAVE AN IMAGE
                viz.printout(data, entry['name'])
        
                # CALL PRINTER
                done.add(entry['key'])
                time.sleep(10);
        
