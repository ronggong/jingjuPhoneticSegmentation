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

import numpy as np
import sys,os

sys.path.append(os.path.join(os.path.dirname(__file__), "speechSegmentation"))
sys.path.append(os.path.join(os.path.dirname(__file__), "public"))
sys.path.append(os.path.join(os.path.dirname(__file__), "VUV"))
sys.path.append(os.path.join(os.path.dirname(__file__), "eval"))

from SpeechSegmentation import detectedBoundariesOutput
from trainTestSeparation import getRecordings,getRecordingNumber
from patternManip import featureSyllableSegmentation,voicedChangePatternCollection,scalerPattern,getIntervalVoiced,boundaryReduction,patternPadding,plotDetection
from textgridParser import syllableTextgridExtraction
from random import randint,sample
from patternManip import svm_cv,svm_model,report_cv
from sklearn.externals import joblib
import Winebarger as Win
import metrics

from parameters import *

svm_model_filename    = models_path+'/phoneme_segmentation/svm_model_biccPattern.pkl'

icdPatterns_filename    = os.path.join(eval_path,'rong','biccPattern','icdPatterns.npy')
voicedPatterns_filename = os.path.join(eval_path,'rong','biccPattern','voicedPatterns.npy')
index_vp_filename       = os.path.join(eval_path,'rong','biccPattern','index_vp.npy')
f_s_filename            = os.path.join(eval_path,'rong','biccPattern','f_s.npy')
f_vuv_s_filename        = os.path.join(eval_path,'rong','biccPattern','f_vuv_s.npy')
spec_filename           = os.path.join(eval_path,'rong','biccPattern','spec.npy')
pho_s_filename          = os.path.join(eval_path,'rong','biccPattern','pho_s.npy')
gtb_filename            = os.path.join(eval_path,'rong','biccPattern','gtb.npy')
db_filename             = os.path.join(eval_path,'rong','biccPattern','db.npy')
gtbv_filename           = os.path.join(eval_path,'rong','biccPattern','gtbv.npy')
dbv_filename            = os.path.join(eval_path,'rong','biccPattern','dbv.npy')

var_names       = {}
var_names['icdPatterns']    = icdPatterns_filename
var_names['voicedPatterns'] = voicedPatterns_filename
var_names['index_vp']       = index_vp_filename
var_names['f_s']            = f_s_filename
var_names['f_vuv']          = f_vuv_s_filename
var_names['spec']           = spec_filename
var_names['pho_s']          = pho_s_filename
var_names['gtb']            = gtb_filename
var_names['db']             = db_filename
var_names['gtbv']           = gtbv_filename
var_names['dbv']            = dbv_filename

# varin           = {}

# framesize_t = 0.020
# hopsize_t   = 0.010
# fs          = 44100

varin['phonemeSegFunction'] = Win.mainFunction
# varin['plot']        = False
# varin['fs']          = 44100

# varin['framesize']   = int(round(framesize_t*fs))
# varin['hopsize']     = int(round(hopsize_t*fs))

varin['feature_select'] = 'mfccBands'

varin['vuvCorrection'] = True

h2          = 0.02
alpha       = 0.2
p_lambda    = -0.2

varin['h2']             = h2
varin['alpha']          = alpha
varin['p_lambda']       = p_lambda

varin['mode_bic']    = 0
varin['alpha']       = 0.5
varin['winmax']      = 0.35              # max dynamic window size

varin['tolerance']   = 0.04
varin['N_feature']   = 40
varin['N_pattern']   = 21                # adjust this param, l in paper

recordings          = getRecordings(wav_path)
number_recording    = getRecordingNumber('TRAIN')
recordings_train    = []
for ii in number_recording:
    recordings_train.append(recordings[ii])

number_recording    = getRecordingNumber('TEST')
recordings_test     = []
for ii in number_recording:
    recordings_test.append(recordings[ii])

def incorrectDetection(groundtruthBoundariesVoiced, detectedBoundariesVoiced, tolerance):

    correctDetection = []
    for db in detectedBoundariesVoiced:
        for gtb in groundtruthBoundariesVoiced:
            if abs(db-gtb) < tolerance:
                correctDetection.append(db)
    icd         = np.delete(detectedBoundariesVoiced, np.array(correctDetection))
    return icd

def icdPatternCollection(feature, icd, varin):

    icdPatterns = []
    index       = []        # pattern index in icd
    N = varin['N_pattern']
    for ii in range(len(icd)):
        icd_frame   = int(round(icd[ii]*varin['fs']/varin['hopsize']))

        # if icd_frame-int(N/2)<0 or icd_frame+int(N/2)+1>feature.shape[0]:
        #     continue
        #
        # icdPattern      = feature[icd_frame-int(N/2):icd_frame+int(N/2)+1,:]

        icdPattern      = patternPadding(feature,icd_frame,N)
        icdPattern      = scalerPattern(icdPattern)
        icdPattern      = np.reshape(icdPattern, varin['N_feature']*varin['N_pattern'])
        icdPatterns.append(icdPattern)
        index.append(ii)

    return icdPatterns, index

def getDataAll(textgrid_path,recordings,varin):

    icdPatterns_all     = []    # voiced incorrect patterns
    voicedPatterns_all  = []    # voiced patterns, including incorrect patterns
    index_vp_all        = []    # index of detected boundaries who has voiced patterns
    f_s_all             = []
    f_vuv_s_all         = []
    spec_all            = []
    pho_s_all           = []
    gtb_all,db_all,gtbv_all,dbv_all     = [],[],[],[]

    # recordings level
    for recording in recordings:

        nestedPhonemeLists, numSyllables, numPhonemes   \
            = syllableTextgridExtraction(textgrid_path,recording,'pinyin','details')

        feature_syllables, feature_vuv_syllables, spec_syllables, phoneme_syllables \
            = featureSyllableSegmentation(feature_path, recording, nestedPhonemeLists,varin)

        groundtruthBoundariesSong, detectedBoundariesSong, \
        groundtruthBoundariesVoicedSong, detectedBoundariesVoicedSong \
            = detectedBoundariesOutput(recording,varin)

        f_s_all.append(feature_syllables)
        f_vuv_s_all.append(feature_vuv_syllables)
        spec_all.append(spec_syllables)
        pho_s_all.append(phoneme_syllables)

        gtb_all.append(groundtruthBoundariesSong)
        db_all.append(detectedBoundariesSong)
        gtbv_all.append(groundtruthBoundariesVoicedSong)
        dbv_all.append(detectedBoundariesVoicedSong)

        icdPatterns_song     = []
        voicedPatterns_song  = []
        index_vp_song        = []

        # syllable level
        for ii in range(len(groundtruthBoundariesVoicedSong)):

            groundtruthBoundariesVoiced     = groundtruthBoundariesVoicedSong[ii]
            detectedBoundariesVoiced        = detectedBoundariesVoicedSong[ii]
            icd                             = incorrectDetection(groundtruthBoundariesVoiced,
                                                                 detectedBoundariesVoiced,
                                                                 varin['tolerance'])

            feature                         = feature_syllables[ii]
            icdPatterns,_                   = icdPatternCollection(feature,icd,varin)
            if len(icdPatterns):
                icdPatterns                 = np.vstack(icdPatterns)
                icdPatterns_song.append(icdPatterns)

            voicedPatterns,index_vp         = icdPatternCollection(feature,detectedBoundariesVoiced,varin)
            if len(voicedPatterns):
                voicedPatterns              = np.vstack(voicedPatterns)
            # voicedPatterns is possible to be empty
            voicedPatterns_song.append(voicedPatterns)
            index_vp_song.append(index_vp)

        icdPatterns_song        = np.vstack(icdPatterns_song)
        icdPatterns_all.append(icdPatterns_song)

        voicedPatterns_all.append(voicedPatterns_song)
        index_vp_all.append(index_vp_song)

    icdPatterns_all             = np.vstack(icdPatterns_all)

    return icdPatterns_all,voicedPatterns_all,index_vp_all,f_s_all,f_vuv_s_all,spec_all,pho_s_all,gtb_all,db_all,gtbv_all,dbv_all

def saveDataAll(varNames,icdPatterns_all,voicedPatterns_all,index_vp_all,f_s_all,f_vuv_s_all,spec_all,pho_s_all,gtb_all,db_all,gtbv_all,dbv_all):

    np.save(varNames['icdPatterns'],icdPatterns_all)
    np.save(varNames['voicedPatterns'],voicedPatterns_all)
    np.save(varNames['index_vp'],index_vp_all)
    np.save(varNames['f_s'],f_s_all)
    np.save(varNames['f_vuv'],f_vuv_s_all)
    np.save(varNames['spec'],spec_all)
    np.save(varNames['pho_s'],pho_s_all)
    np.save(varNames['gtb'],gtb_all)
    np.save(varNames['db'],db_all)
    np.save(varNames['gtbv'],gtbv_all)
    np.save(varNames['dbv'],dbv_all)

def loadDataAll(varNames):

    icdPatterns_all     = np.load(varNames['icdPatterns'])
    voicedPatterns_all  = np.load(varNames['voicedPatterns'])
    index_vp_all        = np.load(varNames['index_vp'])
    f_s_all             = np.load(varNames['f_s'])
    f_vuv_s_all         = np.load(varNames['f_vuv'])
    spec_all            = np.load(varNames['spec'])
    pho_s_all           = np.load(varNames['pho_s'])
    gtb_all             = np.load(varNames['gtb'])
    db_all              = np.load(varNames['db'])
    gtbv_all            = np.load(varNames['gtbv'])
    dbv_all             = np.load(varNames['dbv'])

    return icdPatterns_all,voicedPatterns_all,index_vp_all,f_s_all,f_vuv_s_all,spec_all,pho_s_all,gtb_all,db_all,gtbv_all,dbv_all


def parameterSVM():

    # find the best parameters

    icdPatterns_train,voicedPatterns_train,index_vp_train,\
    f_s_train,f_vuv_s_train,spec_train,pho_s_train,\
    gtb_train,db_train,gtbv_train,dbv_train = getDataAll(textgrid_path,recordings_train,varin)

    f_s_train     = np.hstack(f_s_train)
    f_vuv_s_train = np.hstack(f_vuv_s_train)
    spec_train    = np.hstack(spec_train)
    pho_s_train   = np.hstack(pho_s_train)

    # incorrect detection patterns
    patterns_voiced_unchange    = icdPatterns_train

    # voiced change patterns taken from the ground truth annotation
    patterns_voiced_change,patterns_voiced_change_shifted        = voicedChangePatternCollection(f_s_train, pho_s_train, varin)
    patterns_voiced_change          = patterns_voiced_change+patterns_voiced_change_shifted

    for ii in range(len(patterns_voiced_change)):
        patterns_voiced_change[ii]  = scalerPattern(patterns_voiced_change[ii])
        patterns_voiced_change[ii]  = np.reshape(patterns_voiced_change[ii], varin['N_pattern']*varin['N_feature'])
    patterns_voiced_change          = np.vstack(patterns_voiced_change)

    n_pvc   = patterns_voiced_change.shape[0]
    n_pvnc  = patterns_voiced_unchange.shape[0]

    print patterns_voiced_change.shape, patterns_voiced_unchange.shape

    fv_train        = np.vstack((patterns_voiced_change,patterns_voiced_unchange))
    target_train    = [0]*n_pvc + [1]*n_pvnc

    svm_cv(fv_train,target_train)

    return fv_train, target_train

def modelTraining(fv_train,target_train,C=10,gamma=0.001):
    # rbf 10, 0.001, recall_weighted
    # linear 0.1, balanced
    svm_model(fv_train,target_train,C,gamma,svm_model_filename)


def getTestVariables():
    icdPatterns_test,voicedPatterns_test,index_vp_test,\
    f_s_test,f_vuv_s_test,spec_test,pho_s_test,\
    gtb_test,db_test,gtbv_test,dbv_test = getDataAll(textgrid_path,recordings_test,varin)

    saveDataAll(var_names,icdPatterns_test,voicedPatterns_test,
                index_vp_test,f_s_test,f_vuv_s_test,spec_test,
                pho_s_test,gtb_test,db_test,gtbv_test,dbv_test)

def predict():

    icdPatterns_test,voicedPatterns_test,index_vp_test,\
    f_s_test,f_vuv_s_test,spec_test,pho_s_test,\
    gtb_test,db_test,gtbv_test,dbv_test = loadDataAll(var_names)

    svm_model_object= joblib.load(svm_model_filename)

    sumNumGroundtruthBoundaries, sumNumDetectedBoundaries, sumNumCorrect = 0,0,0

    for ii in range(len(gtbv_test)):
        gtb_song         = gtb_test[ii]
        db_song          = db_test[ii]
        dbv_song         = dbv_test[ii]
        voicedPatterns_song     = voicedPatterns_test[ii]
        index_vp_song           = index_vp_test[ii]
        f_vuv_s_song            = f_vuv_s_test[ii]
        spec_song               = spec_test[ii]
        pho_s_song              = pho_s_test[ii]

        for jj in range(len(gtb_song)):
            print 'index test syllable ', jj, 'in total ', len(gtb_song)

            spec        = spec_song[jj]
            pho_s       = pho_s_song[jj]
            gtb         = gtb_song[jj]
            db          = db_song[jj]
            dbv         = dbv_song[jj]
            dbuv        = np.setdiff1d(db,dbv)

            voicedPatterns     = voicedPatterns_song[jj]
            if len(voicedPatterns):
                index_vp           = np.array(index_vp_song[jj])

                target             = svm_model_object.predict(voicedPatterns)
                dbvc               = dbv[index_vp[np.nonzero(1-target)]]    # detected boundaries voiced correct
                dbc                = np.hstack((dbuv,dbvc))
            else:
                dbc                = dbuv

            dbc        = dbc*fs/varin['hopsize']

            dbc        = np.array(boundaryReduction(dbc.tolist()))*(varin['hopsize']/float(fs))

            if varin['plot']:
                print pho_s

                plotDetection(spec,gtb,dbc,varin)

            numDetectedBoundaries, numGroundtruthBoundaries, numCorrect = \
                        metrics.boundaryDetection(groundtruthBoundaries=gtb,
                                              detectedBoundaries=dbc,
                                              tolerance=varin['tolerance'])

            sumNumGroundtruthBoundaries += numGroundtruthBoundaries
            sumNumDetectedBoundaries    += numDetectedBoundaries
            sumNumCorrect               += numCorrect

    HR, OS, FAR, F, R, deletion, insertion = \
            metrics.metrics(sumNumDetectedBoundaries, sumNumGroundtruthBoundaries, sumNumCorrect)

    print ('HR %.3f, OS %.3f, FAR %.3f, F %.3f, R %.3f, deletion %i, insertion %i' %
                   (HR, OS, FAR, F, R, deletion, insertion))
    print ('ground truth %i, detected %i, correct %i' % (sumNumGroundtruthBoundaries,sumNumDetectedBoundaries,sumNumCorrect))
