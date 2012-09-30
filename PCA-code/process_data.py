#!/usr/bin/env python
# encoding: utf-8
"""
process_data.py

Created by Gilles de Hollander on 2012-09-30.
Copyright (c) 2012 __MyCompanyName__. All rights reserved.
"""

import sys
import os
import pickle as pkl
import numpy as np
from pca import PCA

import scipy.io as sio
import gzip

DB_FILE = 'db3.pkl'
PREFIX = 'data/v2_'

result = 'result.pkl'

similarities = pkl.load(open(DB_FILE))

base_artists = sorted(similarities.keys())

similarity_measure = dict()
all_artists = []

for base_artist, similarity in similarities.items():
    for s in similarity:
        similarity_measure[tuple(sorted([s.item.name, base_artist]))] = s.match
        all_artists.append(s.item.name)


print 'n of artists: %d' % len(all_artists)
all_artists = sorted(list(set(all_artists)))
print 'n of unique artists: %d' % len(all_artists)

similarity_matrix = np.zeros((len(base_artists), len(all_artists)))

for base_artist, similarity in similarities.items():
    for s in similarity:
        similarity_matrix[base_artists.index(base_artist), all_artists.index(s.item.name)] = s.match

mean_features = np.mean(similarity_matrix, axis=0)
pc = PCA(similarity_matrix - mean_features)



# sio.savemat(open(PREFIX+'data', 'w'), dict(PCA=pc, similarity_matrix=similarity_matrix), do_compression=True)
# sio.savemat(open(PREFIX+'similarity_matix.mat', 'w'), similarity_matrix, do_compression=True)
# sio.savemat(similarity_matrix, open(PREFIX+'similarity_matrix.mat', 'w'), do_compression = True)
# sio.savemat(pc, open(PREFIX+'PCA.mat', 'w'))
pkl.dump(base_artists, open(PREFIX+'base_artists.pkl', 'w'))
pkl.dump(all_artists, open(PREFIX+'all_artists.pkl', 'w'))
pkl.dump(similarity_matrix, open(PREFIX+'similarity_matrix.pkl', 'w'))
pkl.dump(pc, open(PREFIX+'PCA.pkl', 'w'))
pkl.dump(pc.Vt[0], open(PREFIX+'first_component.pkl', 'w'))
pkl.dump(mean_features, open(PREFIX+'mean_features.pkl', 'w'))
