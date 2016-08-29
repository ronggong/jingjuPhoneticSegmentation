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

import SpeechSegmentation   # uncomment this

####--- 0. extract and save features ----####

##--instruction start--##
# necessary step, you have to do this to save the feature files for each audio

# download the audio dataset in https://github.com/CompMusic/jingju-lyrics-annotation/tree/master/annotations/3folds/
# for fold1, fold2 and fold3.
# put all audios (without directories) into ./dataset/wav
##--end--##

# SpeechSegmentation.featureExtraction()



####---- 1. VUV classification ----####

##--instruction start--##
# to train the VUV SVM model and test the classification performance, please look at VUVclassification.py
##--end--##



####---- 2. speech segmentation method ----####

##--instruction start--##
# please read this at first

# feature_set: 'mfcc', 'dmfcc', 'mfccBands','gfcc','plpcc','plp','rasta-plpcc', 'rasta-plp', 'bark'
# feature string can be selected from the feature_set

# train_test_string is 'TRAIN' or 'TEST', depending on which data set want to use

# set parameters in parameters.py

# uncomment the method to run
##--end--##


# aversano method
# SpeechSegmentation.ave_eval(feature_string='bark',train_test_string='TEST')

# hoang method
# SpeechSegmentation.hoa_eval(feature_string='mfccBands',train_test_string='TEST')

# winebarger method
# SpeechSegmentation.win_eval(feature_string='mfccBands',train_test_string='TEST')




####---- 3. pattern classification method ----####
# from SVFPatternMethod import *        # uncomment this

# find the best parameters, you don't have to do this, there is a pre-trained model
# fv_train, target_train = parameterSVM()

# training the model by the C and gamma, you don't have to do this, there is a pre-trained model
# modelTraining(fv_train,target_train,C=10,gamma=0.001)

# calculate test set variable, you must do this, because the test data is not included in the git folder
# getTestVariables()

# predict on test set
# predict()