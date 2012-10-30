#!/usr/bin/env python
# encoding: utf-8

""" Main script for music-dna.
    Checks the website for new artist data,
    processes them (by projecting on the coeffiecients),
    saves an output image, and calls the printing function
"""

### IMPORTS ###

# Python
import os
import sys
import time
import json
import urllib
import urllib2
import pickle
import gzip
from collections import defaultdict

# Libraries
import numpy as np
from scipy.misc import imsave
from scipy.ndimage.interpolation import zoom
import matplotlib.pyplot as plt

# Shortcuts & library settings
np.seterr(all='raise')


### HELPER FUNCTIONS ###

def remap(n, (min_old, max_old), (min_new, max_new)):
    return (n - min_old) / (max_old - min_old) * (max_new - min_new) + min_new

def normalize(matrix):
    norms = np.sqrt((matrix ** 2).sum(axis=1))
    return matrix / norms[:, np.newaxis]


### CLASSES ### 

class LastFM(object):
    """ A mini lastfm api wrapper with built-in caching. 
        Can only deal with non-authenticated API requests.
    """
    
    API_URL = "http://ws.audioscrobbler.com/2.0/"
    CACHE_FILE = '_lastfmcache.pickle.gz'
    
    def __init__(self, api_key):
        self.api_key = api_key
        try:
            self.cache = pickle.load(gzip.open(self.CACHE_FILE))
        except IOError as e:
            print "LastFM: Error reading cache. %s" % e
            self.cache = {}
    
    def get(self, method, params):
        """ Makes the API call, retrieves from cache if needed, 
            and parses JSON
        """
        # Fetch the data from lastfm if not in cache
        if (method, frozenset(params.items())) not in self.cache:
            querystring = urllib.urlencode(params)
            url = '%s?method=%s&%s&api_key=%s&format=json' % (LastFM.API_URL, method, querystring, self.api_key)
            print "LastFM: %s %s" % (method, params)
            data = json.loads(urllib2.urlopen(url).read())
            self.cache[(method, frozenset(params.items()))] = data
            # Store the cache every 10 items.
            if len(self.cache) % 10 == 0:
                with gzip.open(self.CACHE_FILE, 'wb') as gz:
                    pickle.dump(self.cache, gz, protocol=-1)
        return self.cache[(method, frozenset(params.items()))]
        
    def artist_getsimilar(self, artist):
        data = self.get('artist.getsimilar', {'artist': artist.encode('utf-8')})
        return [(d['name'],float(d['match'])) for d in data['similarartists']['artist']]
        
    def tag_gettopartists(self, tag):
        data = self.get('tag.gettopartists', {'tag': tag})
        return [d['name'] for d in data['topartists']['artist']]
        
    def chart_gettopartists(self, limit=200):
        data = self.get('chart.gettopartists', {'limit': limit})
        return [d['name'] for d in data['artists']['artist']]
        
    def user_gettopartists(self, user, limit=50):
        data = self.get('user.gettopartists', {'user': user, 'limit': limit})
        return [d['name'] for d in data['topartists']['artist']]
        

class ArtistMatrix(object):
    """ Keeps a list of feature artists, and the coefficient matrix
        that projects new artists onto a number of principal components.
    """
    def __init__(self, lastfm):                
        self.lastfm = lastfm # An instance of a lastfm api
        
        self.feature_artists = None # The artists of which the similarity is used as
                                    # "features". The "columns" of the matrix
        self.coefficients    = None # The feature vector coefficients resulting from PCA
        self.center          = None # The data center (mean)
        
    def load(self, filename):
        """ Load this object's data from the given file. """
        fields = pickle.load(gzip.open(filename))
        # Set fields 
        self.feature_artists = fields['feature_artists']
        self.coefficients    = fields['coefficients']
        self.center          = fields['center']
        
    def dump(self, filename):
        """ Dumps this object's data to the given file. """
        fields = {'feature_artists': self.feature_artists,
                  'coefficients': self.coefficients,
                  'center': self.center}
        pickle.dump(fields, gzip.open(filename, 'wb'))
                
    def get_vector(self, artist):
        """ Returns an artist as a 'feature vector', that is,
            its similarities to each of the feature artists
        """
        similar_to = defaultdict(float, self.lastfm.artist_getsimilar(artist))
        return np.array([similar_to[f] for f in self.feature_artists])
        
    def do_pca(self, feature_artists, sample_artists, max_comps=100):
        """ Performs PCA on the feature_artists/sample_artist matrix
        """
        self.feature_artists = feature_artists
        # Build a data matrix, observations are rows
        matrix = np.vstack([self.get_vector(artist) for artist in sample_artists])
        # Subtract the mean
        self.center = matrix.mean(axis=0)
        matrix -= self.center
        print "ArtistMatrix: doing PCA on %s matrix." % (matrix.shape, )
        _, d, v = np.linalg.svd(np.cov(matrix.T))
        self.coefficients = v.T[:max_comps]
        print "ArtistMatrix: obtained %s coefficient matrix." % (self.coefficients.shape, )
        
    def project(self, artists):
        # Gather data
        data = np.vstack([self.get_vector(artist) for artist in artists])
        data -= self.center                          # Subtract center
        data = normalize(data)
        return np.dot(self.coefficients, data.T).T
        
    def print_info(self):
        """ Print some information about the projection this matrix
            performs
        """
        for i in range(3):
            print "=== Dimension %d ===" % i
            assert len(self.coefficients[i,:]) == len(self.feature_artists)
            influence = sorted(zip(self.coefficients[i,:], self.feature_artists), reverse=True)
            print '    More like: ' + ', '.join(artist for (_, artist) in influence[:5])
            print '    Less like: ' + ', '.join(artist for (_, artist) in influence[-5:])

 
class Visualizer(object):
    """ Simulates a gel-electrophoresis graphic, using the
        projected artists.
    """
    
    def __init__(self, resolution=(150, 400), bars=3, bar_height=20):
        # Vars
        self.res            = resolution
        self.bars           = bars
        # TODO: Set proper range for bars, this is just empirically found.
        self.bar_data_range = [(-0.01, 0.01) for _ in xrange(bars)]
        self.bar_height     = bar_height
        
    def draw(self, projection, filename):
        """ Projection is an n x p matrix, 
            individual artists are rows, their projected
            features are the columns
        """
        bars = np.vstack([self.drawbar(projection[:,i]) for i in xrange(3)])
        imsave(filename, bars)
        
    def drawbar(self, projection):
        """ Return image of a single electrophoresis bar """
        w = self.res[0]
        bar = np.ones((30, w))
        x = remap(projection, self.bar_data_range[0], (0, w))
        x = np.clip(x, 1, w-2).astype('int')
        for _x in x:
            bar[:, _x-1:_x+1] *= 0.5
            bar[1:-1, _x+1] *= 0.5
        
        return zoom(bar, 2, order=0)
        
        
        
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
        feature_artists = [artist for tag in LASTFM_FRONTPAGE_TAGS for artist in lastfm.tag_gettopartists(tag)]
        sample_artists = lastfm.chart_gettopartists(50)
        am.do_pca(feature_artists, sample_artists)
        am.dump('_artistmatrix.pickle.gz')
    
    am.print_info()
        
    viz = Visualizer()
        
    # TEST
    for user in ['andr01d', 'jeboyG', 'tomaiz']:
        artists = lastfm.user_gettopartists(user, limit=10)
        viz.draw(am.project(artists), 'bar_' + user + '.png')

    # TODO: Below
    # while True:
        # CHECK FOR NEW USER DATA ONLINE
        
        # CLEAN DATA
        
        # PROJECT ARTISTS ONTO FEATURES
        
        # SAVE AN IMAGE
        
        # CALL PRINTER
        # time.sleep(10);
        
