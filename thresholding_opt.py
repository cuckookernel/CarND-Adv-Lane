# -*- coding: utf-8 -*-
"""
Created on Thu Dec  6 18:21:15 2018

@author: mrestrepo
"""

import numpy as np
import cv2

import util  as u

from skopt import gp_minimize
from skopt.utils import use_named_args
from skopt.space import Real
import matplotlib.pyplot as plt

DATA_DIR = "C:/_DATA/autonomous-driving-nd/Adv_Lane_finding"

orig = u.rgb_read( DATA_DIR + '/color-shadow-example.jpg')

#%% Preproc

sobel_kernel=7

hls = cv2.cvtColor(orig, cv2.COLOR_RGB2HLS)

l_channel = hls[:,:,1]
s_channel = hls[:,:,2]

gray = u.rgb2gray( orig )

sobelx = cv2.Sobel(l_channel, cv2.CV_32F, 1, 0, ksize=sobel_kernel)
sobely = cv2.Sobel(l_channel, cv2.CV_32F, 0, 1, ksize=sobel_kernel)

s_sobelx = cv2.Sobel(s_channel, cv2.CV_32F, 1, 0, ksize=sobel_kernel)
s_sobely = cv2.Sobel(s_channel, cv2.CV_32F, 0, 1, ksize=sobel_kernel)

#sobelx = cv2.Sobel(s_channel, cv2.CV_32F, 1, 0, ksize=sobel_kernel)
#sobely = cv2.Sobel(s_channel, cv2.CV_32F, 0, 1, ksize=sobel_kernel)

abs_sob_x = np.abs( sobelx )
abs_sob_y = np.abs( sobely )

scaled_x = np.uint8( 255 * abs_sob_x / abs_sob_x.max() )

abs_sob_x = np.abs( sobelx )
abs_sob_y = np.abs( sobely )

angle = np.arctan2(abs_sob_y, abs_sob_x)
s_angle = np.arctan2(abs_sob_y, abs_sob_x)

mag = np.sqrt( sobelx ** 2  + sobely ** 2 )
mag_scaled = np.uint8( mag / ( mag.max() / 255.0 ) )


s_mag = np.sqrt( s_sobelx ** 2  + s_sobely ** 2 )

#%%

target = u.rgb2gray ( u.rgb_read( DATA_DIR + '/binary-combo-img.jpg') )
target = np.uint8( target > 20  )
#%%

def ml() :
#%%
    def bh( img ) :
        return img[ 400 : , ...]

    from sklearn.tree import DecisionTreeClassifier
    import pandas as pd

    X = pd.DataFrame(  { "scaled_x"   : bh(scaled_x).ravel(),
                         "s_channel"  : bh(s_channel).ravel(),
                         #"s_mag"      : bh(s_mag).ravel(),
                         "mag_scaled" : bh(mag_scaled).ravel(),
                         "angle"      : bh(angle).ravel() } )

    y = np.float32(  bh(target).ravel() )

    tree = DecisionTreeClassifier( max_depth = 4, class_weight={ 0 : 1, 1 : 2.1 } )
    print( "fitting")
    tree.fit( X, y )

    y_pred = np.float32( tree.predict( X ) )

    print( "accu" , (y == y_pred).mean()  )

    binary = np.uint8( y_pred.reshape( bh(target).shape )  * 255 )
#%%
    print( "binary range:", binary.min(), binary.max(), binary.mean()  )
    fig, [[ax1,ax2],[ax3,ax4]] = plt.subplots(2 ,2,   figsize=(16,5) )
    ax1.imshow( binary * 255, cmap='gray'  )
    ax2.imshow( bh(target) * 255, cmap='gray' )

    binary2 = ( ( scaled_x > 15 ) & ( mag_scaled > 21.5 )
              | ( scaled_x > 9.5) & (s_channel > 128  )
              | ( s_channel > 153) & (mag_scaled > 2.5)
              #| ( s_channel > 57.5 ) #&  (mag_scaled )
              )
    #binary2 = ( ( scaled_x > 15.5 ) & ( (mag_scaled > 21.5)  | (s_channel > 128) ) |
    #            ( ( s_channel > 153 ) & (mag_scaled > 5.5) ) )

    ax4.imshow( bh(binary2) * 255, cmap='gray' )

    print_tree( make_tree( tree.tree_, 0 ), col_names = X.columns )

#%%

class Node :
    def __init__(self, idx) :
        self.idx = idx
        self.feature = None
        self.threshold = None
        self.value = None
        self.right = None
        self.left  = None


def make_tree( tree_,  i ) :

    ret = Node(i)

    if tree_.children_right[i] != -1  and tree_.children_left[i] != -1 :
        ret.feature = tree_.feature[i]
        ret.threshold = tree_.threshold[i]
    else :
        ret.value = tree_.value[i].squeeze()

    if tree_.children_right[i] != -1 :
        right_idx = tree_.children_right[i]
        ret.right = make_tree( tree_, right_idx)

    if tree_.children_left[i] != -1 :
        left_idx = tree_.children_left[i]
        ret.left = make_tree( tree_, left_idx)

    print( f"make_tree {i}: feat={ret.feature} feat={ret.threshold}")

    return ret

def print_tree( node, col_names, indent = 0, pref="" ) :

    if node.feature is not None :
        print( " "  * indent, pref, #f"({node.idx})",
               col_names[node.feature], ' > ', node.threshold, sep="")
    else :
        print( " "  * indent, pref, node.value, sep="")


    if node.right is not None :
        print_tree( node.right, col_names, indent + 4, pref= "Y: " )

    if node.left is not None :
        print_tree( node.left, col_names, indent + 4, pref= "N: " )
#%%
binary = np.zeros_like( target )

space = [ Real( 0, 255.0, name = "min_mag" ),
          Real( 0, 255.0, name = "min_sat" ),
          Real( 0, np.pi/2, name = "min_angle" ),
          Real( 0, 1.0, name = "frac_mag" ),
          Real( 0, 1.0, name = "frac_sat" ),
          Real( 0, 1.0, name = "frac_angle" )
          ]
#%%

def make_binary( **params ) :
    min_mag = params['min_mag']
    max_mag = min_mag + ( 255 - min_mag) * params['frac_mag']

    min_angle = params['min_angle']
    max_angle = min_angle + (np.pi/2 - min_angle) *  params['frac_angle']

    min_sat = params['min_sat']
    max_sat = min_sat + (255 - min_sat) *  params['frac_sat']

    min_x = params['min_x']
    max_x = min_x + (1100 - min_x) *  params['frac_sat']


    filt_mag   = ( mag_scaled >=  min_mag   ) & ( mag_scaled <= max_mag )
    filt_x     = ( scaled_x   >=  min_x ) & ( scaled_x      <= max_x )
    #filt_angle = ( angle      >=  min_angle ) & ( angle      <= max_angle )
    filt_sat   = ( s_channel  >=  min_sat ) & ( s_channel  <= max_sat )

    #return filt_mag & filt_angle | ( filt_x & filt_sat )
    return ( filt_x | filt_sat ) & filt_mag
#%%

def test() :
    #%%
    params = { 'min_mag' : 3, 'frac_mag' : 1.0,
               'min_sat' : 170,  'frac_sat' : 1.0,
               'min_angle' : 0, 'frac_angle': 0.1,
               'min_x' : 20, 'frac_x': 0.5 }
    binary = make_binary( **params )
    binary = np.uint8( y_pred.reshape( target.shape )  * 255 )

    print( "binary range:", binary.min(), binary.max(), binary.mean()  )
    fig, (ax1,ax2) = plt.subplots(1 ,2, figsize=(16,5) )
    ax1.imshow( binary * 255, cmap='gray'  )
    #ax1.imshow( mag_scaled, cmap='gray'  )
    ax2.imshow( target * 255, cmap='gray' )
    #%%


min_found = float("inf")
@use_named_args( space )
def agreement(**params ) :
    print( params )

    binary = make_binary( params )

    global min_found

    val = 1.0 - np.mean( binary == target )
    if val < min_found:
        min_found = val

    print( "val=%.3f min_found=%.3f\n" % (val, min_found) )
    return val

def run() :
    res_gp = gp_minimize(agreement, space, n_calls=500, random_state=0)