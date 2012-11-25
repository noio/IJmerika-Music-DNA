#!/usr/bin/env python
# encoding: utf-8

""" Main script for music-dna.
    Checks the website for new artist data,
    processes them (by projecting on the coeffiecients),
    saves an output image, and calls the printing function
"""

### IMPORTS ###

# Python

# Local
from musicdna import *
        
        
### SETTINGS ###

# Got this by using javascript: "$$('.tagList a').map(function(e) { return e.text })" on the last.fm Music page. 
LASTFM_FRONTPAGE_TAGS = ["ambient", "blues", "classical", "country", "electronic", "emo", "folk", "hardcore", "hip hop", "indie", "jazz", "latin", "metal", "pop", "pop punk", "punk", "reggae", "rnb", "rock", "soul", "world", "60s", "70s", "80s", "90s"]

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
        
        am.dump('_artistmatrix.pickle.gz')
    
    am.print_info()
    
    viz = Visualizer()
    
    # TODO: Below
    # while True:
        # CHECK FOR NEW USER DATA ONLINE
        
        # CLEAN DATA
        
        # PROJECT ARTISTS ONTO FEATURES
        
        # SAVE AN IMAGE
        
        # CALL PRINTER
        # time.sleep(10);
        
