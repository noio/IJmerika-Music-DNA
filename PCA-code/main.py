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
# np.seterr(all='raise')
import pickle as pkl
from operator import itemgetter
from pprint import pprint
import scipy.io as sio
from scipy import stats

from pca import PCA

import matplotlib
import matplotlib.pyplot as plt

# You have to have your own unique two values for API_KEY and API_SECRET
# Obtain yours from http://www.last.fm/api/account for Last.fm
API_KEY = "dfa3f081178624892ff5259be9bdc7c2" # this is a sample key
API_SECRET = "6d34c6ab655c33e4e46f1b59a946d6f3"

PREFIX = 'data/v22_'

network = pylast.LastFMNetwork(api_key = API_KEY, api_secret = 
    API_SECRET)

class SimilarityGetter(object):
    
    
    def __init__(self, prefix):

        self.prefix = prefix
        self.base_artists = pkl.load(open(self.prefix+'base_artists.pkl'))
        self.all_artists = pkl.load(open(self.prefix+'all_artists.pkl'))
        self.pca_0 = pkl.load(open(self.prefix+'first_component.pkl'))
        self.similarity_matrix = pkl.load(open(prefix+'similarity_matrix.pkl'))
        self.network = None
        self.mean_features = pkl.load(open(prefix+'mean_features.pkl'))
        
        
        if os.path.exists(self.prefix+'distr.pkl'):
            print 'Distributie gevonden'
            self.distr = stats.distributions.norm(**pkl.load(open(self.prefix+'distr.pkl')))
        else:
            print 'Geen Distributie gevonden'
            self.distr = None
        

        
        if os.path.exists(self.prefix+'cache.pkl'):
            self.cache = pkl.load(open(self.prefix+'cache.pkl'))
            print self.cache
        else:
            self.cache = {}
            print 'Kut.'
        
    
    def get_value(self, artist_string):
        
        if artist_string in self.cache.keys():
            print 'In Cache'
            return self.cache[artist_string]
        
        
        elif artist_string in self.base_artists:
            print 'IN DB'
            vector = self.similarity_matrix[self.base_artists.index(artist_string), :]
            
        else:
            
            print 'NOT IN DB'
            
            if self.network == None:
                self.network = pylast.LastFMNetwork(api_key = API_KEY, api_secret = 
                    API_SECRET)
            
            artist = self.network.get_artist(artist_string)
            similarity_list = artist.get_similar()
            similar_artists = [s.item.name for s in similarity_list]
            similar_values = [s.match for s in similarity_list]
            
            vector = np.zeros((len(self.all_artists)))
            
            for other_artist, match in zip(similar_artists, similar_values):
                if other_artist in self.all_artists:
                    vector[self.all_artists.index(other_artist)] = match
            
            
        if np.sum(vector) > 0:
            result = np.dot(vector/np.dot(vector, vector) - self.mean_features, self.pca_0)
        else:
            result = np.nan
        
        if self.distr is not None:
            result = self.distr.cdf(result)
        
        self.cache[artist_string] = result
        
        pkl.dump(self.cache, open(self.prefix+'cache.pkl', 'w'))
        return result
            
            
    def get_values(self, artist_list, filter_nan=True):
        if filter_nan:
            values = [self.get_value(artist) for artist in artist_list]
            return [v for v in values if v != np.nan]
        else:
            return [self.get_value(artist) for artist in artist_list]


if __name__ == '__main__':
    print 'Creating similarity-getter'
    
    
    x = SimilarityGetter(PREFIX)
    # print x.get_value('The Beatles')
    # print x.get_value('The Doors')
    # print x.get_value('The Kinks')
    # print x.get_value('High the Moon')

    # plt.hist([x.get_values(['Metallica', 'Slayer', 'Anthrax', 'System of a Down']), x.get_values(['The Killers', 'Brandon Flowers', 'Alt-J', 'Franz Ferdinand']), x.get_values(['The Doors', 'Rolling Stones', 'Beatles', 'The Kinks', 'Jefferson Airplane'])], bins=100)
    # plt.hist([x.get_values(['Metallica', 'Slayer', 'Anthrax', 'System of a Down']), x.get_values(['The Killers', 'Brandon Flowers', 'Alt-J', 'Franz Ferdinand']), x.get_values(['The Doors', 'Rolling Stones', 'Beatles', 'The Kinks', 'Jefferson Airplane'])], bins=100)
    
    
    gilles_artists = network.get_user('jeboyG').get_top_artists()
    print 'Getting Gilles artists %s' % [a.item.name for a in gilles_artists]
    gilles = [a.item.name for a in gilles_artists]

    thomas_artists = network.get_user('andr01d').get_top_artists()
    print 'Getting Thomas artists % s' % [a.item.name for a in thomas_artists]
    tom = [a.item.name for a in thomas_artists]

    tom_artists = network.get_user('tomaiz').get_top_artists()
    print 'Getting Tom artists %s' % [a.item.name for a in tom_artists]
    thomas = [a.item.name for a in tom_artists]



    # values = [x.get_values(gilles[:5]), x.get_values(tom[:5]), x.get_values(thomas[:5])]
    values = [x.get_values(gilles), x.get_values(tom), x.get_values(thomas)]

    

    # plt.hist(values, bins=100)
    # plt.plot(np.tile([0, 1], (len(values), 1)), np.tile(values, (1,2)), linewidth=2.0)
    plt.hlines(y=values[0], xmin=0, xmax=1, color='red', linewidth=2.0)
    plt.hlines(y=values[1], xmin=0, xmax=1, color='blue', linewidth=2.0)
    plt.hlines(y=values[2], xmin=0, xmax=1, color='green', linewidth=2.0)
    plt.legend(['Gilles', 'Tom', 'Thomas'])
    plt.show()