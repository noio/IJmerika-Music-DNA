#!/usr/bin/env python
# encoding: utf-8

""" Test script for music-dna.
"""

### IMPORTS ###

# Python

# Local
from musicdna import *
import os
import matplotlib
import matplotlib.pyplot as plt
from xml.etree import ElementTree as et
import pickle as pkl
        
### SETTINGS ###

# Got this by using javascript: "$$('.tagList a').map(function(e) { return e.text })" on the last.fm Music page. 
LASTFM_FRONTPAGE_TAGS = ["ambient", "blues", "classical", "country", "electronic", "emo", "folk", "hardcore", "hip hop", "indie", "jazz", "latin", "metal", "pop", "pop punk", "punk", "reggae", "rnb", "rock", "soul", "world", "60s", "70s", "80s", "90s"]


USERS = ['LianneKLV', 'GuidoTM', 'ReneeKD', 'Cmmie' ,'JoyceYB', 'andr01d', 'jeboyG', 'tomaiz']

N_FEATS_PER_TAG = 100

### MAIN PROCEDURE ###



def build_poster(sample_artists, am, filename, target_length=75):
    
    doc = et.Element('svg', width='1600', height='1200', version='1.1', xmlns='http://www.w3.org/2000/svg')
    
    projected_sample_artists, valid_artists = am.project(sample_artists)
    sample_artists = np.array(sample_artists)[valid_artists]
            
    sample_scores = am.redistribute(projected_sample_artists[valid_artists, :])
    
    
    
    if target_length != None:
        stepsize = len(sample_artists)/target_length
        sample_artists = sample_artists[0:-1:stepsize]
        sample_scores = sample_scores[0:-1:stepsize, :]
    
    
    # sample_artists = sample_artists[0:-1:3]
    # sample_scores sample_scores[
    
    for artist, score in zip(sample_artists, sample_scores):
        for i in range(3):
            text1 = et.Element('text', x='300', y=str(100 + score[0] * 1000), fill='black', style='font-family:Sans;font-size:20px;text-anchor:middle;dominant-baseline:top')
            text2 = et.Element('text', x='800', y=str(100 + score[1] * 1000), fill='black', style='font-family:Sans;font-size:20px;text-anchor:middle;dominant-baseline:top')
            text3 = et.Element('text', x='1300', y=str(100 + score[2] * 1000), fill='black', style='font-family:Sans;font-size:20px;text-anchor:middle;dominant-baseline:top')
            text1.text = text2.text = text3.text = artist
            doc.append(text1)
            doc.append(text2)
            doc.append(text3)
            
    f = open(filename, 'w')

    f.write('<?xml version=\"1.0\" standalone=\"no\"?>\n')
    f.write('<!DOCTYPE svg PUBLIC \"-//W3C//DTD SVG 1.1//EN\"\n')
    f.write('\"http://www.w3.org/Graphics/SVG/1.1/DTD/svg11.dtd\">\n')
    f.write(et.tostring(doc))
    f.close()
    
    
    

if __name__ == '__main__':
    lastfm = LastFM('765932fff6d3d591f8635746fbaa1b7a')
    
    lda_axes_set = [
                    [('indie', 'pop'), ('metal', 'pop'), ('indie', 'rap')],
                    [('60s', '80s'), ('70s', '90s'), ('80s', '90s')],
                    [('soul', 'indie'), ('rap', 'metal'), ('latin', 'metal')],
                   ]
    
    
    projection_methods = [
                          # {'method':'PCA', 'sample_method':'gilles_friends', 'normalize_projection':'uniformize'},
                          # {'method':'LDA', 'axes':lda_axes_set[0], 'sample_method':100, 'normalize_projection':'none'}, 
                          # {'method':'LDA', 'axes':lda_axes_set[0], 'sample_method':100, 'normalize_projection':'zscore'}, 
                          # {'method':'LDA', 'axes':lda_axes_set[0], 'sample_method':100, 'normalize_projection':'uniformize'}, 
                          # {'method':'LDA', 'axes':lda_axes_set[1], 'sample_method':100, 'normalize_projection':'uniformize'}, 
                          # {'method':'LDA', 'axes':lda_axes_set[2], 'sample_method':25, 'normalize_projection':'uniformize'}, 
                          # # {'method':'PCA', 'sample_method':100, 'normalize_projection':'none'}, 
                          # {'method':'PCA', 'sample_method':100, 'normalize_projection':'uniformize'}, 
                          # # {'method':'PCA', 'sample_method':100, 'normalize_projection':'zscore'}, 
                          # # {'method':'PCA', 'sample_method':250, 'normalize_projection':'none'}, 
                          # {'method':'PCA', 'sample_method':250, 'normalize_projection':'uniformize'}, 
                          {'method':'PCA', 'sample_method':500, 'normalize_projection':'uniformize'},
                          # {'method':'PCA', 'sample_method':750, 'normalize_projection':'uniformize'},
                          # {'method':'PCA', 'sample_method':600, 'normalize_projection':'uniformize'},
                          ]
    
    for projection_method in projection_methods:
        print "Building new Artist Matrix."
        am = ArtistMatrix(lastfm)



        feature_artists = set([artist for tag in LASTFM_FRONTPAGE_TAGS for artist in lastfm.tag_gettopartists(tag, N_FEATS_PER_TAG)])

        if projection_method['method'] == 'PCA':
            # PCA
            
            if projection_method['sample_method'] == 'gilles_friends':
                sample_artists = pickle.load(open('gillesvrienden.pkl')).values()
                sample_artists = [unicode(a, 'utf-8') for sl in sample_artists for a in sl ]
                sample_artists = sample_artists[:500]
                print 'Length sample artists before uniqueifing: ', len(sample_artists)
                sample_artists = np.unique(sample_artists)
                print 'Length sample artists after uniqueifing: ', len(sample_artists)
            else:
                sample_artists = lastfm.chart_gettopartists(projection_method['sample_method'])
                
            am.set_projection_pca(feature_artists, sample_artists)
        elif projection_method['method'] == 'LDA':
            sample_artists = []
            for axes in projection_method['axes']:
                print 'LDA on %s vs %s' % (axes[0], axes[1])
                am.set_projection_lda(feature_artists,
                                    lastfm.tag_gettopartists(axes[0], projection_method['sample_method']), 
                                    lastfm.tag_gettopartists(axes[1], projection_method['sample_method'])
                                    )
                sample_artists += lastfm.tag_gettopartists(axes[0], projection_method['sample_method']) + lastfm.tag_gettopartists(axes[1], projection_method['sample_method'])
                
        
        am.find_distribution(sample_artists)


        if projection_method['method'] == 'LDA':
            
            axes_string = '_' + '_'.join(['%s_vs_%s' % (a[0], a[1]) for a in projection_method['axes']]) + '_'            
            fn = 'sample_%s_%s_%s_%d_feats_tag.svg' % (projection_method['method'], projection_method['sample_method'], axes_string, N_FEATS_PER_TAG)
        else:
            fn = 'sample_%s_%s_%d_feats_tag.svg' % (projection_method['method'], projection_method['sample_method'], N_FEATS_PER_TAG)
        
        
        build_poster(sample_artists, am, os.path.join('posters', fn), target_length=None)
        

        pkl.dump(am, open('artist_matrices/%s.pkl' % ''.join(fn.split('.')[:-1]), 'w'))