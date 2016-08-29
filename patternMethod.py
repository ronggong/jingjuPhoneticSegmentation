# -*- coding: utf-8 -*-
'''
 * Copyright (C) 2016  Music Technology Group - Universitat Pompeu Fabra
 *
 * This file is part of jingjuPhoneticSegmentation
 *
 * pypYIN is free software: you can redistribute it and/or modify it under
 * the terms of the GNU Affero General Public License as published by the Free
 * Software Foundation (FSF), either version 3 of the License, or (at your
 * option) any later version.
 *
 * This program is distributed in the hope that it will be useful, but WITHOUT
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
 * FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more
 * details.
 *
 * You should have received a copy of the Affero GNU General Public License
 * version 3 along with this program.  If not, see http://www.gnu.org/licenses/
 *
 * If you have any problem about this python version code, please contact: Rong Gong
 * rong.gong@upf.edu
 *
 *
 * If you want to refer this code, please use this article:
 *
'''

##--instruction start--##

# not used in paper
# only pattern method used, no VUV, no SVF

##--end--##

import sys,os,time,csv
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), "speechSegmentation"))
sys.path.append(os.path.join(os.path.dirname(__file__), "public"))
sys.path.append(os.path.join(os.path.dirname(__file__), "VUV"))
sys.path.append(os.path.join(os.path.dirname(__file__), "eval"))

import metrics
import essentia.standard as ess
import features

from textgridParser import syllableTextgridExtraction
from patternManip import scalerPattern,featureSyllableSegmentation,voicedChangePatternCollection,voicedUnchangePatternCollection
from patternManip import svm_cv,svm_model,report_cv
from patternManip import getIntervalVoiced,testPatternClassification,mergeConsecutiveZeroIndex,boundaryFrame,boundaryReduction
from sklearn.externals import joblib
from trainTestSeparation import getRecordings

from random import randint,sample
import matplotlib.pyplot as plt

from parameters import *

####---- initialization

varin           = {}

# wav_path        = '/Users/gong/Documents/MTG document/Jingju arias/phonemeSeg/wav'
# textgrid_path   = '/Users/gong/Documents/MTG document/Jingju arias/phonemeSeg/textgrid'
# feature_path    = '/Users/gong/Documents/MTG document/Jingju arias/phonemeSeg/feature'

svm_model_filename    = models_path+'/phoneme_segmentation/svm_model_pattern.pkl'

# varin['tolerance']   = 0.04
# framesize_t = 0.020
# hopsize_t   = 0.010
# fs          = 44100
#
# varin['plot']        = False
# varin['fs']          = 44100
#
# varin['framesize']   = int(round(framesize_t*fs))
# varin['hopsize']     = int(round(hopsize_t*fs))

varin['N_pattern']   = 11
varin['N_feature']   = 36

recordings = getRecordings(wav_path)

####---- collect all features and phonemes
f_s_all             = []
f_vuv_s_all         = []
spec_all            = []
pho_s_all           = []
for recording in recordings:
    nestedPhonemeLists, numSyllables, numPhonemes = syllableTextgridExtraction(textgrid_path,recording,'pinyin','details')
    feature_syllables, feature_vuv_syllables, spec_syllables, phoneme_syllables = featureSyllableSegmentation(feature_path, recording, nestedPhonemeLists,varin)
    f_s_all.append(feature_syllables)
    f_vuv_s_all.append(feature_vuv_syllables)
    spec_all.append(spec_syllables)
    pho_s_all.append(phoneme_syllables)

f_s_all     = np.hstack(f_s_all)
f_vuv_s_all = np.hstack(f_vuv_s_all)
spec_all    = np.hstack(spec_all)
pho_s_all   = np.hstack(pho_s_all)

####--- patterns
patterns_voiced_change,_        = voicedChangePatternCollection(f_s_all, pho_s_all, varin)
patterns_voiced_unchange        = voicedUnchangePatternCollection(f_s_all, pho_s_all, varin)

for ii in range(len(patterns_voiced_change)):
    patterns_voiced_change[ii]  = scalerPattern(patterns_voiced_change[ii])
    patterns_voiced_change[ii]  = np.reshape(patterns_voiced_change[ii], varin['N_pattern']*varin['N_feature'])
patterns_voiced_change          = np.vstack(patterns_voiced_change)

for ii in range(len(patterns_voiced_unchange)):
    patterns_voiced_unchange[ii]  = scalerPattern(patterns_voiced_unchange[ii])
    patterns_voiced_unchange[ii]  = np.reshape(patterns_voiced_unchange[ii], varin['N_pattern']*varin['N_feature'])
patterns_voiced_unchange          = np.vstack(patterns_voiced_unchange)


n_pvc   = patterns_voiced_change.shape[0]
n_pvnc  = patterns_voiced_unchange.shape[0]

print patterns_voiced_change.shape, patterns_voiced_unchange.shape

pvc_test_set_index      = np.array(sample(range(n_pvc), int(n_pvc*0.25)))
pvc_train_set_index     = np.delete(np.array(range(n_pvc)), pvc_test_set_index)

pvnc_test_set_index      = np.array(sample(range(n_pvnc), int(n_pvnc*0.25)))
pvnc_train_set_index     = np.delete(np.array(range(n_pvnc)), pvnc_test_set_index)

pvc_train               = patterns_voiced_change[pvc_train_set_index,:]
pvc_test                = patterns_voiced_change[pvc_test_set_index,:]

pvnc_train              = patterns_voiced_unchange[pvnc_train_set_index,:]
pvnc_test               = patterns_voiced_unchange[pvnc_test_set_index,:]

# f_s_test        = f_s_all[test_set_index]
# f_vuv_s_test    = f_vuv_s_all[test_set_index]
# spec_test       = spec_all[test_set_index]
# pho_s_test      = pho_s_all[test_set_index]

####--- scale feature
# f_s_train_stacked   = np.vstack(f_s_train)
#
# scaler          = preprocessing.StandardScaler().fit(f_s_train_stacked)
#
# for ii in range(len(f_s_train)):
#     f_s_train[ii]   = scaler.transform(f_s_train[ii])
#
# for ii in range(len(f_s_test)):
#     f_s_test[ii]   = scaler.transform(f_s_test[ii])
print pvc_train.shape, pvnc_train.shape

fv_train        = np.vstack((pvc_train,pvnc_train))
target_train    = [0]*pvc_train.shape[0] + [1]*pvnc_train.shape[0]
# svm_cv(fv_train,target_train)


# svm_model(fv_train,target_train,100,0.001,svm_model_filename)

svm_model_object= joblib.load(svm_model_filename)

sumNumGroundtruthBoundaries, sumNumDetectedBoundaries, sumNumCorrect = 0,0,0

f_s_test        = f_s_all
f_vuv_s_test    = f_vuv_s_all
pho_s_test      = pho_s_all

for ii in range(len(f_vuv_s_test)):
    interval_voiced, boundary_detected_unvoiced           = getIntervalVoiced(f_vuv_s_test[ii],varin)
    x_frame, target_test      = testPatternClassification(f_s_test[ii], interval_voiced, svm_model_object, varin)
    boundary_detected_voiced  = mergeConsecutiveZeroIndex(x_frame,target_test)
    boundary_gt               = boundaryFrame(pho_s_test[ii],varin)
    boundary_detected         = boundary_detected_unvoiced + boundary_detected_voiced
    boundary_detected         = boundaryReduction(boundary_detected)
    print 'index test syllable ', ii, 'in total ', len(f_vuv_s_test)

    if varin['plot']:
        N               = 2 * varin['framesize']
        mX              = spec_test[ii]
        mX              = np.transpose(mX)
        maxplotfreq     = 6001.0
        eps             = np.finfo(np.float).eps
        mXPlot          = mX[:int(N*(maxplotfreq/fs))+1,:]
        binFreqs        = np.arange(mXPlot.shape[0])*fs/float(N)
        # timestamps_spec = np.arange(mXPlot.shape[1]) * (varin['hopsize']/float(fs))
        timestamps_spec = np.arange(mXPlot.shape[1])

        # print std_patterns
        # print mean_patterns

        f, axarr = plt.subplots(2, sharex=True)
        axarr[0].pcolormesh(timestamps_spec, binFreqs, 20*np.log10(mXPlot+eps))
        for b_gt in boundary_gt:
            axarr[0].axvline(b_gt)

        for zf in boundary_detected:
            axarr[1].axvline(zf)
        # axarr[1].plot(x_frame,target_test)
        plt.show()

    boundary_gt             = np.array(boundary_gt)*(varin['hopsize']/float(fs))
    boundary_detected       = np.array(boundary_detected)*(varin['hopsize']/float(fs))

    print boundary_gt, boundary_detected

    numDetectedBoundaries, numGroundtruthBoundaries, numCorrect = \
    metrics.boundaryDetection(groundtruthBoundaries=boundary_gt,
                          detectedBoundaries=boundary_detected,
                          tolerance=varin['tolerance'])
    print numDetectedBoundaries, numGroundtruthBoundaries, numCorrect

    sumNumGroundtruthBoundaries += numGroundtruthBoundaries
    sumNumDetectedBoundaries    += numDetectedBoundaries
    sumNumCorrect               += numCorrect

HR, OS, FAR, F, R, deletion, insertion = \
                    metrics.metrics(sumNumDetectedBoundaries, sumNumGroundtruthBoundaries, sumNumCorrect)

print HR, OS, FAR, F, R, deletion, insertion, sumNumDetectedBoundaries, sumNumGroundtruthBoundaries, sumNumCorrect
