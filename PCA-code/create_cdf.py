#!/usr/bin/env python
# encoding: utf-8
"""
create_cdf.py

Created by Gilles de Hollander on 2012-09-30.
Copyright (c) 2012 __MyCompanyName__. All rights reserved.
"""

import sys
import os
from main import SimilarityGetter
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from operator import itemgetter
from pprint import pprint
from scipy import stats
import pickle as pkl

PREFIX = 'data/v22_'

x = SimilarityGetter(PREFIX)

print 'BLA'
all_values = x.get_values(x.base_artists)

pkl.dump(dict(mean=np.mean(all_values), std=np.std(all_values)), open(PREFIX+'distr.pkl', 'w'))

pprint(sorted(zip(x.base_artists, all_values), key=itemgetter(1)))

# pprint(sorted(zip(x.all_artists, x.pca_0), key=itemgetter(1)))[:10]
# pprint(sorted(zip(x.all_artists, x.pca_0), key=itemgetter(1)))[-10:]

print np.mean(all_values)
# plt.hist(all_values, bins=100)
# plt.show()