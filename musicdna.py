#!/usr/bin/env python
# encoding: utf-8

""" Library scripts for music-dna.
    Contains bookkeeping classes and other stuff.
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

import scipy
from scipy import stats

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
    
    def get(self, method, params, extractor=None):
        """ Makes the API call, retrieves from cache if needed, 
            and parses JSON
        """
        # Fetch the data from lastfm if not in cache
        if (method, frozenset(params.items())) not in self.cache:
            querystring = urllib.urlencode(params)
            url = '%s?method=%s&%s&api_key=%s&format=json' % (LastFM.API_URL, method, querystring, self.api_key)
            print "LastFM: %s %s" % (method, params)
            data = json.loads(urllib2.urlopen(url).read())
            
            try:
                # Extract only the desired data, saves cache size/time
                if callable(extractor):
                    data = extractor(data)
            except (TypeError, KeyError):
                data = []
            self.cache[(method, frozenset(params.items()))] = data
            # Store the cache every 20 items.

            if len(self.cache) % 20 == 0:
                self.dump_cache()
        return self.cache[(method, frozenset(params.items()))]
        
    def dump_cache(self):
        """ Dumps the cache now. """
        print "LastFM: dumping cache..."
        with gzip.open(self.CACHE_FILE, 'wb') as gz:
            pickle.dump(self.cache, gz, protocol=-1)
        
    def artist_getsimilar(self, artist):
        if artist == "[unknown]":
            return []

        extractor = lambda data: [(d['name'],float(d['match'])) for d in data['similarartists']['artist']]
        return self.get('artist.getsimilar', {'artist': artist.encode('utf-8')}, extractor)
        
    def tag_gettopartists(self, tag, limit=50):
        extractor = lambda data: [d['name'] for d in data['topartists']['artist']]
        return self.get('tag.gettopartists', {'tag': tag, 'limit':limit}, extractor)
        
    def chart_gettopartists(self, limit=200):
        extractor = lambda data: [d['name'] for d in data['artists']['artist']]
        return self.get('chart.gettopartists', {'limit': limit}, extractor)
        
    def user_gettopartists(self, user, limit=50, period='12month'):
        extractor = lambda data: [d['name'] for d in data['topartists']['artist']]
        return self.get('user.gettopartists', {'user': user, 'limit': limit}, extractor)
        

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
        self.distributions = [] # Distributions that resemble resulting projections for large population
        
    def load(self, filename):
        """ Load this object's data from the given file. """
        fields = pickle.load(gzip.open(filename))
        # Set fields 
        self.feature_artists = fields['feature_artists']
        self.coefficients    = fields['coefficients']
        self.center          = fields['center']
        self.distributions   = fields['distributions']
        
    def dump(self, filename):
        """ Dumps this object's data to the given file. """
        fields = {'feature_artists': self.feature_artists,
                  'coefficients': self.coefficients,
                  'center': self.center,
                  'distributions': self.distributions,
                  }
        pickle.dump(fields, gzip.open(filename, 'wb'))
                
    def get_vector(self, artist):
        """ Returns an artist as a 'feature vector', that is,
            its similarities to each of the feature artists
        """
        similar_to = defaultdict(float, self.lastfm.artist_getsimilar(artist))
        return np.array([similar_to[f] for f in self.feature_artists])
        
    def set_projection_pca(self, feature_artists, sample_artists, max_comps=100):
        """ Performs PCA on the feature_artists/sample_artist matrix
        """
        self.feature_artists = feature_artists
        # Build a data matrix, observations are rows
        matrix = np.vstack([self.get_vector(artist) for artist in sample_artists])
        matrix = matrix[(matrix ** 2).sum(axis=1) > 0]
        # Subtract the mean
        self.center = matrix.mean(axis=0)
        matrix -= self.center
        print "ArtistMatrix: doing PCA on %s matrix." % (matrix.shape, )
        _, d, v = np.linalg.svd(np.cov(matrix.T))
        self.coefficients = v.T[:max_comps]
        print "ArtistMatrix: obtained %s coefficient matrix." % (self.coefficients.shape, )

        
    def set_projection_lda(self, feature_artists, sample0, sample1, add=True):
        """ Performs linear disciminant analysis:
        """
        self.feature_artists = feature_artists
        # Build a data matrix, observations are rows
        matrix0 = np.vstack([self.get_vector(artist) for artist in sample0])
        matrix1 = np.vstack([self.get_vector(artist) for artist in sample1])
        # Compute the class means
        mu0 = matrix0.mean(axis=0)
        mu1 = matrix1.mean(axis=0)
        # ..and the shared covariance (LDA assumes that Sigma0 == Sigma1)
        matrix0 -= mu0
        matrix1 -= mu1
        self.center = (mu0 + mu1) / 2
        sigma = np.cov(np.vstack((matrix0, matrix1)).T)
        # Compute the vector w along which we project
        w = normalize(np.dot(np.linalg.pinv(sigma), (mu1 - mu0))[np.newaxis, :])
        # Either create a new coefficient matrix, or add w to an existing one
        if not add or self.coefficients is None:
            self.coefficients = w
        else:
            self.coefficients = np.vstack((self.coefficients, w))
        
    def project(self, artists, normalize_projection=True, normalization_method='uniformize'):
        # Gather data
        data = np.vstack([self.get_vector(artist) for artist in artists])
        valid = (data ** 2).sum(axis=1) > 0
        data -= self.center                          # Subtract center
        data = normalize(data)
        
        return np.dot(self.coefficients, data.T).T, valid
            
    def find_distribution(self, sample):
        """ Find the data distributions for the given sample """
        data, _ = self.project(sample)
        self.distributions = [scipy.stats.norm.fit(dim) for dim in data.T]
        
            
    def redistribute(self, projection, method='uniform'):
        """ Redistribute the projected coefficients
        """
        for i, (col, dist) in enumerate(zip(projection.T, self.distributions)):
            if method == 'uniform':
                dist = scipy.stats.norm(*dist)
                projection[:, i] = dist.cdf(col)
            else:
                projection[:, i] = (col - dist[0]) / dist[1]
        return projection
            
        
    def print_info(self):
        """ Print some information about the projection this matrix
            performs
        """
        for i in range(min(3, len(self.coefficients))):
            print "=== Dimension %d ===" % i
            assert len(self.coefficients[i,:]) == len(self.feature_artists)
            projected, _ = self.project(self.feature_artists)[:, i]
            influence = sorted(zip(project, self.feature_artists), reverse=True)
            print '    More like: ' + ', '.join(artist for (_, artist) in influence[:5])
            print '    Less like: ' + ', '.join(artist for (_, artist) in influence[-5:])
            

 
class Visualizer(object):
    """ Simulates a gel-electrophoresis graphic, using the
        projected artists.
    """
    

    def __init__(self, printer, resolution=512, 
                                bars=3, 
                                bar_width=10,
                                bar_height=30, 
                                bar_range=None,
                                dry_run=False):
        # Vars
        self.printer    = printer
        self.resolution = resolution
        self.bars       = bars
        self.bar_range  = bar_range
        self.bar_height = bar_height
        self.bar_width  = bar_width
        self.dry_run    = dry_run
        
        if self.bar_range == None:
            self.bar_range = [(0, 1) for _ in xrange(bars)]
        
        
    def printout(self, projection, username):
        """ Projection is an n x p matrix, 
            individual artists are rows, their projected
            features are the columns
        """
        for i in xrange(min(3, projection.shape[1])):
            bar = self.drawbar(projection[:,i])
            imsave('temp.png', bar)
            
            self.printcall('image','temp.png')
            self.printcall('control','LF')
        
        gibber = "MUSIC_DNA//USR:%s" % (username.upper())
        
        self.printcall('text',gibber)
        self.printcall('set', align='right')
        self.printcall('control','LF')
        self.printcall('image','256-noise-clean.gif')
        self.printcall('cut')
        
    def printcall(self, f, *args, **kwargs):
        if self.printer is not None and not self.dry_run:
            getattr(self.printer,f)(*args, **kwargs)
        
    def drawbar(self, projection):
        """ Return image of a single electrophoresis bar """
        w = self.resolution / 2
        bw = self.bar_width // 2
        bar = np.ones((self.bar_height, w))
        x = remap(projection, self.bar_range[0], (0, w))
        x = np.clip(x, bw, w-bw).astype('int')
        for _x in x:
            bar[:, _x-bw:_x+(bw-1)] *= 0.8
            bar[1:-1, _x+(bw-1)] *= 0.8
        
        return zoom(bar, 2, order=0)
        
        
        
