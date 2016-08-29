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

import sys,os,time,csv
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), "speechSegmentation"))
sys.path.append(os.path.join(os.path.dirname(__file__), "public"))
sys.path.append(os.path.join(os.path.dirname(__file__), "VUV"))
sys.path.append(os.path.join(os.path.dirname(__file__), "eval"))


import Aversano as Ave
import Hoang    as Hoa
import Estevan  as Est
import Winebarger as Win

import metrics
import essentia.standard as ess
import features

from textgridParser import syllableTextgridExtraction
from vuvAlgos import featureVUV,consonantInterval
from trainTestSeparation import getRecordings,getRecordingNumber
from sklearn.externals import joblib

from parameters import *

def featureExtraction():

    '''
    Extract features and save
    :param recordings:
    :param varin:
    :return:
    '''
    recordings              = getRecordings(wav_path)

    for recording in recordings:
        wav_file   = os.path.join(wav_path,recording+'.wav')
        energy_filename         = os.path.join(feature_path,'energy'+'_'+recording+'_'
                                                   +str(varin['framesize'])+'_'+str(varin['hopsize'])+'.npy')
        spec_filename           = os.path.join(feature_path,'spec'+'_'+recording+'_'
                                                   +str(varin['framesize'])+'_'+str(varin['hopsize'])+'.npy')
        for featurename in feature_set:
            print 'saving feature for ', recording, ', feature ', featurename
            feature_filename        = os.path.join(feature_path,featurename+'_'+recording+'_'
                                                   +str(varin['framesize'])+'_'+str(varin['hopsize'])+'.npy')
            varin['feature_select'] = featurename
            feature, energy, spec = features.features(wav_file,varin)

            np.save(feature_filename,feature)

            if featurename == feature_set[-1]:
                np.save(energy_filename,energy)
                np.save(spec_filename,spec)

def featureSyllableSegmentation(feature_path, recording, nestedPhonemeLists,varin):

    '''
    obtain the features of each syllable
    :param wav_path:
    :param recording:
    :param fs:
    :param nestedPhonemeLists:
    :return:
    feature_syllables is a list containing the feature for each syllable, len(audio_syllables) = len(nestedPhonemeLists)
    '''
    hopsize         = varin['hopsize']
    fs              = varin['fs']

    energy_filename         = os.path.join(feature_path,'energy'+'_'+recording+'_'
                                               +str(varin['framesize'])+'_'+str(varin['hopsize'])+'.npy')
    spec_filename           = os.path.join(feature_path,'spec'+'_'+recording+'_'
                                               +str(varin['framesize'])+'_'+str(varin['hopsize'])+'.npy')
    feature_filename        = os.path.join(feature_path,varin['feature_select']+'_'+recording+'_'
                                               +str(varin['framesize'])+'_'+str(varin['hopsize'])+'.npy')

    feature1                  = np.load(feature_filename)
    spec                      = np.load(spec_filename)
    # vuv feature
    feature_vuv               = featureVUV(feature_path,recording,varin)

    if varin['phonemeSegFunction'] == Win.mainFunction:
        mfcc_feature_filename     = os.path.join(feature_path,'mfcc'+'_'+recording+'_'
                                               +str(varin['framesize'])+'_'+str(varin['hopsize'])+'.npy')
        feature2                  = np.load(mfcc_feature_filename)
    else:
        feature2                  = np.load(energy_filename)

    feature1_syllables      = []
    feature2_syllables      = []
    spec_syllables          = []
    feature_vuv_syllables   = []
    for nestedPho in nestedPhonemeLists:
        syllable_start_frame    = int(round(nestedPho[0][0]*fs/hopsize))
        syllable_end_frame      = int(round(nestedPho[0][1]*fs/hopsize))
        feature_syllable        = feature1[syllable_start_frame:syllable_end_frame,:]
        feature_vuv_syllable    = feature_vuv[syllable_start_frame:syllable_end_frame,:]
        feature1_syllables.append(feature_syllable)
        feature_vuv_syllables.append(feature_vuv_syllable)

        if varin['phonemeSegFunction'] == Win.mainFunction:
            # mfcc
            feature2_syllable       = feature2[syllable_start_frame:syllable_end_frame,:]
        else:
            # energy
            feature2_syllable       = feature2[syllable_start_frame:syllable_end_frame,]

        feature2_syllables.append(feature2_syllable)

        spec_syllable           = spec[syllable_start_frame:syllable_end_frame,:]
        spec_syllables.append(spec_syllable)

    return feature1_syllables, feature2_syllables, spec_syllables, feature_vuv_syllables

def detectedBoundariesReduction(detectedBoundaries):

    '''
    merge the boundaries if they are too close
    :param detectedBoundaries:
    :param tolerance:
    :return:
    '''
    for ii in np.arange(1,len(detectedBoundaries))[::-1]:
        if detectedBoundaries[ii]-detectedBoundaries[ii-1] <= 0.02*2:
            detectedBoundaries[ii-1] = (detectedBoundaries[ii-1]+detectedBoundaries[ii])/2.0
            detectedBoundaries       = np.delete(detectedBoundaries,ii)
    return detectedBoundaries

def eval4oneSong(feature1_syllables, feature2_syllables, spec_syllables, feature_vuv_syllables, nestedPhonemeLists,
                phonemeSegfunction, varin):
    '''
    feature1_syllables: an array containing the feature vectors for each syllable
    '''

    groundtruthBoundariesSong,detectedBoundariesSong,groundtruthBoundariesVoicedSong,detectedBoundariesVoicedSong \
        = [],[],[],[]

    for ii, feature in enumerate(feature1_syllables):

        print 'evaluate syllable ', ii+1, ' in', len(feature1_syllables), ' feature: ', varin['feature_select']

        # phonemes of syllable
        phoList                     = nestedPhonemeLists[ii][1]
        syllable_start_time         = phoList[0][0]
        groundtruthBoundaries       = []
        groundtruthBoundariesVoiced = []

        for pho in phoList:
            groundtruthBoundaries.append(pho[0] - syllable_start_time)

        # syllable end time
        groundtruthBoundaries.append(phoList[-1][1] - syllable_start_time)

        # groundtruth voiced doesn't contain syllable boundary
        for jj in range(len(phoList)-1):
            pho0    = phoList[jj]
            pho1    = phoList[jj+1]
            if pho0[2] not in ['c','k','f','x'] and pho1[2] not in ['c','k','f','x']:
                groundtruthBoundariesVoiced.append(pho0[1] - syllable_start_time)

        if varin['phonemeSegFunction'] == Hoa.mainFunction:
            varin['energy']      = feature2_syllables[ii]
        elif varin['phonemeSegFunction'] == Win.mainFunction and varin['feature_select'] != 'mfcc':
            varin['mfcc']        = feature2_syllables[ii]

        if varin['feature_select'] == 'dmfcc':
            # print varin['mfcc'].shape, feature.shape
            feature              = np.hstack((varin['mfcc'],feature))

        print 'feature shape: ', feature.shape

        detectedBoundaries       = phonemeSegfunction(feature,spec_syllables[ii],varin)
        detectedBoundariesVoiced = detectedBoundaries

        # detect consonant, erase boundaries from this region, then add consonant boundaries
        if varin['vuvCorrection']:
            models_path         = '/Users/gong/Documents/MTG document/Jingju arias/phonemeSeg/models'
            scaler_filename     = os.path.join(models_path,'VUV_classification','standardization','scaler.pkl')
            svm_model_filename  = os.path.join(models_path,'VUV_classification','svm','svm_model_mfcc_undersampling.pkl')
            scaler              = joblib.load(scaler_filename)
            svm_model_object    = joblib.load(svm_model_filename)

            detectedinterval_vuv = consonantInterval(feature_vuv_syllables[ii],scaler,svm_model_object,varin)
            if len(detectedinterval_vuv):
                detectedBoundaries_vuv = np.hstack(detectedinterval_vuv)
                for ii_db in np.arange(len(detectedBoundaries))[::-1]:
                    for di_vuv in detectedinterval_vuv:
                        if di_vuv[0]-0.02 < detectedBoundaries[ii_db] < di_vuv[1]+0.02:
                            detectedBoundaries = np.delete(detectedBoundaries,ii_db)
                            break
            else:
                detectedBoundaries_vuv  = np.array([])

            detectedBoundariesVoiced    = detectedBoundaries
            detectedBoundaries          = np.hstack((detectedBoundaries,detectedBoundaries_vuv))

        # detectedBoundaries = detectedBoundariesReduction(detectedBoundaries)

        groundtruthBoundariesSong.append(groundtruthBoundaries)
        detectedBoundariesSong.append(detectedBoundaries)
        groundtruthBoundariesVoicedSong.append(groundtruthBoundariesVoiced)
        detectedBoundariesVoicedSong.append(detectedBoundariesVoiced)

    return groundtruthBoundariesSong,detectedBoundariesSong,groundtruthBoundariesVoicedSong,detectedBoundariesVoicedSong

def boundariesStat(groundtruthBoundariesAll,detectedBoundariesAll,tolerance):

    '''
    statistics of detected boundaries
    '''

    sumNumGroundtruthBoundaries, sumNumDetectedBoundaries, sumNumCorrect = 0,0,0

    for ii in range(len(groundtruthBoundariesAll)):
        groundtruthBoundariesSong   = groundtruthBoundariesAll[ii]
        detectedBoundariesSong      = detectedBoundariesAll[ii]

        for jj in range(len(groundtruthBoundariesSong)):
            groundtruthBoundaries   = groundtruthBoundariesSong[jj]
            detectedBoundaries      = detectedBoundariesSong[jj]

            # evaluation
            numDetectedBoundaries, numGroundtruthBoundaries, numCorrect = \
            metrics.boundaryDetection(groundtruthBoundaries=groundtruthBoundaries,
                                  detectedBoundaries=detectedBoundaries,
                                  tolerance=tolerance)

            sumNumGroundtruthBoundaries += numGroundtruthBoundaries
            sumNumDetectedBoundaries    += numDetectedBoundaries
            sumNumCorrect               += numCorrect

    return sumNumGroundtruthBoundaries, sumNumDetectedBoundaries, sumNumCorrect

####---- output detected boundaries

def detectedBoundariesOutput(recording,varin):

    # varin['feature_select'] = feature_string
    #
    # varin['vuvCorrection'] = True
    #
    # h2          = 0.02
    # alpha       = 0.2
    # p_lambda    = 0.1
    #
    # varin['h2']             = h2
    # varin['alpha']          = alpha
    # varin['p_lambda']       = p_lambda

    # print 'evaluate ', recording, ' l,h1,h2', varin['h2'], varin['alpha'], varin['p_lambda']

    nestedPhonemeLists, numSyllables, numPhonemes = syllableTextgridExtraction(textgrid_path,recording,'pinyin','details')
    feature_Syllables, mfcc_syllables, spec_syllables, feature_vuv_syllables = featureSyllableSegmentation(feature_path, recording,
                                                                                nestedPhonemeLists, varin)
    groundtruthBoundariesSong, detectedBoundariesSong, groundtruthBoundariesVoicedSong, detectedBoundariesVoicedSong = \
        eval4oneSong(feature_Syllables, mfcc_syllables, spec_syllables, feature_vuv_syllables, nestedPhonemeLists,
                     varin['phonemeSegFunction'], varin)

    return groundtruthBoundariesSong, detectedBoundariesSong, groundtruthBoundariesVoicedSong,detectedBoundariesVoicedSong

def csvWriter(recordings,csv_writer,params):

    groundtruthBoundariesAll, detectedBoundariesAll = [], []

    for ii in params[3]:
        recording   = recordings[ii]

        groundtruthBoundariesSong, detectedBoundariesSong, _,_ = detectedBoundariesOutput(recording,varin)

        groundtruthBoundariesAll.append(groundtruthBoundariesSong)
        detectedBoundariesAll.append(detectedBoundariesSong)

    # for tolerance in [0.02,0.04,0.06,0.08,0.10,0.20]:
    # for tolerance in [0.04]:

    sumNumGroundtruthBoundaries, sumNumDetectedBoundaries, sumNumCorrect = \
        boundariesStat(groundtruthBoundariesAll,detectedBoundariesAll,varin['tolerance'])

    # csv_writer.writerow([tolerance,sumNumDetectedBoundaries, sumNumGroundtruthBoundaries, sumNumCorrect])

    HR, OS, FAR, F, R, deletion, insertion = \
        metrics.metrics(sumNumDetectedBoundaries, sumNumGroundtruthBoundaries, sumNumCorrect)

    print ('HR %.3f, OS %.3f, FAR %.3f, F %.3f, R %.3f, deletion %i, insertion %i' %
               (HR, OS, FAR, F, R, deletion, insertion))
    print ('ground truth %i, detected %i, correct %i' % (sumNumGroundtruthBoundaries,sumNumDetectedBoundaries,sumNumCorrect))

    # csv_writer.writerow([tolerance, params[0], params[1], params[2], HR, OS, FAR, F, R, deletion, insertion, sumNumDetectedBoundaries, sumNumCorrect, sumNumGroundtruthBoundaries])



# framesize_t = 0.020
# hopsize_t   = 0.010
# # fs          = 16000
# fs          = 44100
#
#
# varin                = {}
# varin['plot']        = False
# varin['fs']          = 44100
#
# varin['framesize']   = int(round(framesize_t*fs))
# varin['hopsize']     = int(round(hopsize_t*fs))
#
# feature_set          = ['zcr','autoCorrelation','mfcc', 'dmfcc', 'mfccBands','gfcc','plpcc','plp','rasta-plpcc', 'rasta-plp', 'bark','mrcg']
# varin['feature_select'] = 'mfccBands'    # default feature
# varin['rasta']       = False
# varin['vuvCorrection'] = True
# varin['tolerance']   = 0.04              # evaluation tolerance
#
# ####---- parameters of aversano
# varin['a']           = 20
# varin['b']           = 0.1
# varin['c']           = 7
#
# ####---- parameters of hoang
# varin['max_band']    = 8
# varin['l']           = 2                 # segmentation interval
# varin['h0']          = 0.6               # a0 peak threshold
# varin['h1']          = 0.08              # a1 peak threshold
# varin['h2']          = 0.0725
# varin['th_phone']    = 0.2
# varin['q']           = 3
# varin['k']           = 3
# varin['delta']       = 2.0
# varin['step2']       = False
#
# ####---- parameters of Estevan
# varin['N_win']  = 18            # sliding window
# varin['gamma']  = 10**(-1)      # RBF width
#
# ####---- parameters of winebarger
# varin['p_lambda']    = 0.35
# varin['mode_bic']    = 1                 # 0: bic, 1: bicc
# varin['h2']          = 0.0
# varin['alpha']       = 0.5
# varin['winmax']      = 0.35              # max dynamic window size


####---- hopsize for mrcg is fixed into 0.01 s
# for feature in feature_set:
#     varin['feature_select']     = feature
#     if feature == 'mrcg':
#         varin['hopsize']        = 0.01


####---- Hoang's method

def hoa_eval(feature_string, train_test_string):

    varin['phonemeSegFunction'] = Hoa.mainFunction
    varin['feature_select']     = feature_string
    varin['vuvCorrection']      = False

    eval_result_file_name           = eval_path + '/hoa/hoa_mfccBands2_'+train_test_string+'.csv'
    eval_result_sorted_file_name    = eval_path + '/hoa/hoe_mfccBands2_'+train_test_string+'_sorted.csv'

    recordings              = getRecordings(wav_path)
    number_recording        = getRecordingNumber(train_test_string)

    ####---- for searching best parameters

    # with open(eval_result_file_name, 'w') as testfile:
    #     csv_writer = csv.writer(testfile)

        # best params for f-measure, l-2, h1-0.4, h2-0.1
        # for l in [2]:#[2,4,6,8]:
        #     for h1 in [0.4]:#[0.0,0.2,0.4,0.6,0.8,1.0]:
        #         for h2 in [0.1]:#[0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]:
        #
        #             varin['l']          = l
        #             varin['h1']         = h1
        #             varin['h2']         = h2

    params              = [varin['l'],varin['h1'],varin['h2'],number_recording]

    csvWriter(recordings,None,params)

    # metrics.sortResultByCol(eval_result_file_name, eval_result_sorted_file_name,col=7)

def ave_eval(feature_string, train_test_string):

    varin['phonemeSegFunction'] = Ave.mainFunction
    varin['feature_select']     = feature_string
    varin['vuvCorrection']      = False

    eval_result_file_name   = eval_path + '/ave/ave_bark_'+train_test_string+'.csv'
    eval_result_sorted_file_name   = eval_path + '/ave/ave_bark_sorted_'+train_test_string+'.csv'

    recordings              = getRecordings(wav_path)
    number_recording        = getRecordingNumber(train_test_string)

    ####---- for searching best parameters

    # with open(eval_result_file_name, 'w') as testfile:
    #     csv_writer = csv.writer(testfile)

        # a = 2, b = 0.25, c = 9 is the f-measure best param
        # for a in [1,2,3,4,5]:
        #     for b in [0.05,0.1,0.15,0.2,0.25]:
        #         for c in [3,5,7,9]:
        #
        #             varin['a']          = a
        #             varin['b']          = b
        #             varin['c']          = c

    params              = [varin['a'],varin['b'],varin['c'],number_recording]

    csvWriter(recordings,None,params)

    # metrics.sortResultByCol(eval_result_file_name, eval_result_sorted_file_name,col=7)

####---- Winebarger's method, BICc

def win_eval(feature_string, train_test_string):

    varin['phonemeSegFunction'] = Win.mainFunction

    varin['feature_select'] = feature_string

    recordings              = getRecordings(wav_path)
    number_recording        = getRecordingNumber(train_test_string)

    if varin['mode_bic'] == 0:
        path_bic = 'bic'
    elif varin['mode_bic'] == 1:
        path_bic = 'bicc'
    else:
        raise ValueError('Not a valid mode_bic value.')

    # for vuv in [True,False]:

        # varin['vuvCorrection'] = vuv

    if varin['vuvCorrection']:
        eval_result_file_name = eval_path +'/win/' + path_bic \
                                + '/win_vuv_dmfcc_'+path_bic+'_'+feature_string+'_'+train_test_string + '.csv'

        eval_result_sorted_file_name = eval_path + '/win/' + path_bic \
                                + '/win_vuv_dmfcc_'+path_bic+'_'+feature_string+'_'+train_test_string+'sorted' + '.csv'
    else:
        eval_result_file_name = eval_path + '/win/' + path_bic \
                                + '/win_'+path_bic+'_'+feature_string+'_'+train_test_string + '.csv'

        eval_result_sorted_file_name = eval_path + '/win/'+path_bic \
                                + '/win_'+path_bic+'_'+feature_string+'_'+train_test_string+'sorted' + '.csv'

    ####---- for searching best parameters
    # with open(eval_result_file_name, 'w') as testfile:
    #     csv_writer = csv.writer(testfile)

        # h2          = 0.02
        # alpha       = 0.2

        # p_lambda = 2.0 best param for bicc non vuv
        # p_lambda = 2.2 best param for bicc vuv
        # p_lambda = 0.3 best param for bic non vuv
        # p_lambda = 0.4 best param for bic vuv

        # for p_lambda in [2.2]:#np.arange(-1.0,0.1,0.1):

            # varin['h2']             = h2
            # varin['alpha']          = alpha
            # varin['p_lambda']       = p_lambda

    params                  = [varin['h2'],varin['alpha'],varin['p_lambda'],number_recording]

    csvWriter(recordings,None,params)

    # metrics.sortResultByCol(eval_result_file_name, eval_result_sorted_file_name,col=7)

# if __name__ == '__main__':
#
#     start = time.time()
#
#     # win eval
#     win_eval('mfccBands', wav_path,'TEST')
#
#     # hoang eval
#     # hoa_eval('mfccBands', wav_path,'TEST')
#
#     # aversano eval
#     # ave_eval('bark',wav_path,'TEST')
#
#     end = time.time()
#     print 'elapse time: ', end-start







