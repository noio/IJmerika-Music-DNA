#!/usr/bin/env python
# encoding: utf-8

""" Test script for music-dna.
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
        am.load('_artistmatrix_test.pickle.gz')
        print "Found Artist Matrix data."
    # BUILD ONE OTHERWISE
    except IOError:
        print "Building new Artist Matrix."
        feature_artists = set([artist for tag in LASTFM_FRONTPAGE_TAGS for artist in lastfm.tag_gettopartists(tag, 50)])
        
        # PCA
        sample_artists = lastfm.chart_gettopartists(100)
        print sample_artists
        am.set_projection_pca(feature_artists, sample_artists)
        
        # LDA:
        # am.set_projection_lda(feature_artists,
        #                       lastfm.tag_gettopartists("indie", 20), 
        #                       lastfm.tag_gettopartists("pop", 20))
        # 
        # am.set_projection_lda(feature_artists,
        #                       lastfm.tag_gettopartists("electronic", 20), 
        #                       lastfm.tag_gettopartists("metal", 20))
        #                       
        # am.set_projection_lda(feature_artists,
        #                       lastfm.tag_gettopartists("hip hop", 20), 
        #                       lastfm.tag_gettopartists("indie", 20))
    
        
        am.dump('_artistmatrix_test.pickle.gz')
    
    am.print_info()
    
    viz = Visualizer()
        
    # TEST
    for user in ['andr01d', 'jeboyG', 'tomaiz']:
        artists = lastfm.user_gettopartists(user, limit=20)
        viz.draw(am.project(artists), 'bar_' + user + '.png')

    