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

import os
import numpy as np


def featureVUV(feature_path, recording,varin):

    '''
    Collect the feature vectors for voiced, unvoiced and silence classes
    Training purpose

    :param feature_path:
    :param recording:
    :param nestedPhonemeLists:
    :return:
    '''

    energy_filename         = os.path.join(feature_path,'energy'+'_'+recording+'_'
                                               +str(varin['framesize'])+'_'+str(varin['hopsize'])+'.npy')
    zcr_filename            = os.path.join(feature_path,'zcr'+'_'+recording+'_'
                                               +str(varin['framesize'])+'_'+str(varin['hopsize'])+'.npy')
    autoCorrelation_filename= os.path.join(feature_path,'autoCorrelation'+'_'+recording+'_'
                                               +str(varin['framesize'])+'_'+str(varin['hopsize'])+'.npy')
    mfcc_filename           = os.path.join(feature_path,'mfcc'+'_'+recording+'_'
                                               +str(varin['framesize'])+'_'+str(varin['hopsize'])+'.npy')
    dmfcc_filename          = os.path.join(feature_path,'dmfcc'+'_'+recording+'_'
                                               +str(varin['framesize'])+'_'+str(varin['hopsize'])+'.npy')
    # mfccBands_filename      = os.path.join(feature_path,'mfccBands'+'_'+recording+'_'
    #                                            +str(varin['framesize'])+'_'+str(varin['hopsize'])+'.npy')

    #
    mfcc                    = np.load(mfcc_filename)
    dmfcc                   = np.load(dmfcc_filename)
    # mfccBands               = np.load(mfccBands_filename)
    zcr                     = np.zeros(shape=(mfcc.shape[0],1))
    autoCorrelation         = np.zeros(shape=(mfcc.shape[0],1))
    energy                  = np.zeros(shape=(mfcc.shape[0],1))

    zcr[:,0]                = np.load(zcr_filename)
    autoCorrelation[:,0]    = np.load(autoCorrelation_filename)
    energy[:,0]             = np.load(energy_filename)
    energy                  = np.log(np.finfo(np.float).eps+energy)

    feature                 = np.hstack((zcr,autoCorrelation,energy,mfcc,dmfcc))

    return feature

def consonantInterval(syllable_feature,scaler,svm_model_object,varin):

    '''
    detect consonant interval
    :param syllable_feature:
    :param scaler:
    :param svm_model_object:
    :param hopsize:
    :param fs:
    :return: [[consonant begin time0, consonant end time0],[begin time1, end time1], ...]
    '''
    hopsize         = varin['hopsize']
    fs              = varin['fs']
    syllableLen     = syllable_feature.shape[0] * (hopsize/float(fs))

    # standardization
    pho_feature_transformed = scaler.transform(syllable_feature)
    target_predict          = svm_model_object.predict(pho_feature_transformed)

    # assign silence to voiced
    target_predict[target_predict==2] = 0

    # detected consonant boundaries interval for each syllable
    detectedBoundaries_interval = []
    consonantInterval  = []
    if target_predict[0] and target_predict[1]:
        consonantInterval = [0.5]
    for ii_tp in range(1,len(target_predict)):
        if target_predict[ii_tp]-target_predict[ii_tp-1]>0:
            # interval start
            if detectedBoundaries_interval and ii_tp-0.5 - detectedBoundaries_interval[-1][-1] <= 2:
                # the interval start is too near to the last interval end
                consonantInterval = [detectedBoundaries_interval[-1][0]]
                detectedBoundaries_interval = detectedBoundaries_interval[:-1]
            else:
                consonantInterval = [ii_tp-0.5]
        if target_predict[ii_tp]-target_predict[ii_tp-1]<0 and consonantInterval:
            # interval end
            consonantInterval.append(ii_tp-0.5)
            detectedBoundaries_interval.append(consonantInterval)
            consonantInterval = []
    if target_predict[-1] and target_predict[-2] and consonantInterval:
        consonantInterval.append(len(target_predict)-0.5)
        detectedBoundaries_interval.append(consonantInterval)

    detectedBoundaries_interval = np.array(detectedBoundaries_interval)*hopsize/float(fs)

    # post processing of the interval
    for jj in range(len(detectedBoundaries_interval))[::-1]:
        if detectedBoundaries_interval[jj][0] > syllableLen/8.0 or \
                detectedBoundaries_interval[jj][1]-detectedBoundaries_interval[jj][0] < 0.02:
            detectedBoundaries_interval     = np.delete(detectedBoundaries_interval,jj,axis=0)

    return detectedBoundaries_interval