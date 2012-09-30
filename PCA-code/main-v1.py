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

from pca import PCA

# You have to have your own unique two values for API_KEY and API_SECRET
# Obtain yours from http://www.last.fm/api/account for Last.fm
API_KEY = "dfa3f081178624892ff5259be9bdc7c2" # this is a sample key
API_SECRET = "6d34c6ab655c33e4e46f1b59a946d6f3"


similarities = pkl.load(open('db2.pkl'))

FIRST_ARTISTS = ['Alt-J', 'Iron Maiden', 'Metallica', 'David Guetta', 'The Killers', 'Franz Ferdinand', 'Radiohead', 'Kanye West', 'Blood Red Shoes', 'Anthrax', 'Slayer', 'The Beatles', 'The Rolling Stones', 'Dire Straits', 'Lil B', 'Daft Punk', 'LMFAO', 'Arctic Monkeys', 'Lykke Li', 'Mumford and Sons', 'Scissor Sisters', 'James Blake', 'David Guetta & Nicky Romero', 'Guano Apes', 'Brandon Flowers', u'Foreigner', 'Lana del Rey', 'the Doors', 'Jennifer Lopez']

similarity_measure = dict()
all_artists = []

base_artists = similarities.keys()
for base_artist, similarity in similarities.items():
    for s in similarity:
        similarity_measure[tuple(sorted([s.item.name, base_artist]))] = s.match
        all_artists.append(s.item.name)


all_artists = list(set(all_artists))

pkl.dump(all_artists, open('all_artists.pkl', 'w'))

similarity_matrix = np.zeros((len(base_artists), len(all_artists)))




for base_artist, similarity in similarities.items():
    for s in similarity:
        # similarity_measure[tuple(sorted([s.item.name, base_artist]))] = s.match
        similarity_matrix[base_artists.index(base_artist), all_artists.index(s.item.name)] = s.match

# for artist in base_artists:


base_ind = [base_artists.index(s) for s in FIRST_ARTISTS]

pc = PCA(similarity_matrix)
values = np.dot(similarity_matrix[base_ind, :], pc.Vt[0])
pprint(sorted(zip(FIRST_ARTISTS, values), key=itemgetter(1)))


