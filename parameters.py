# -*- coding: utf-8 -*-

import os

####---- please set these folder according to their locations
current_path    = os.path.dirname(__file__)

wav_path        = os.path.join(current_path, "dataset","wav")
textgrid_path   = os.path.join(current_path, "dataset","textgrid")
feature_path    = os.path.join(current_path, "output","feature")
eval_path       = os.path.join(current_path, "output","eval")
models_path     = os.path.join(current_path, "output","model")

framesize_t = 0.020     # in second
hopsize_t   = 0.010
# fs          = 16000
fs          = 44100

varin                = {}
varin['plot']        = True
varin['fs']          = 44100

varin['framesize']   = int(round(framesize_t*fs))
varin['hopsize']     = int(round(hopsize_t*fs))


####---- below parameters are not took effect in SVFPatternMethod.py

feature_set          = ['zcr','autoCorrelation','mfcc', 'dmfcc', 'mfccBands','gfcc','plpcc','plp','rasta-plpcc', 'rasta-plp', 'bark','mrcg']
varin['feature_select'] = 'mfccBands'    # default feature
varin['rasta']       = False             # no use
varin['vuvCorrection'] = True            # True add VUV classification
varin['tolerance']   = 0.04              # evaluation tolerance, in second

####---- parameters of aversano
varin['a']           = 20
varin['b']           = 0.1
varin['c']           = 7

####---- parameters of hoang
varin['max_band']    = 8
varin['l']           = 2                 # segmentation interval
varin['h0']          = 0.6               # a0 peak threshold
varin['h1']          = 0.08              # a1 peak threshold
varin['h2']          = 0.0725
varin['th_phone']    = 0.2
varin['q']           = 3
varin['k']           = 3
varin['delta']       = 2.0
varin['step2']       = False             # if running the step 2

####---- parameters of Estevan
varin['N_win']  = 18            # sliding window
varin['gamma']  = 10**(-1)      # RBF width

####---- parameters of winebarger
varin['p_lambda']    = 0.35
varin['mode_bic']    = 1                 # 0: bic, 1: bicc
varin['h2']          = 0.0
varin['alpha']       = 0.5
varin['winmax']      = 0.35              # max dynamic window size
