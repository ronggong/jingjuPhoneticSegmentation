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

####---- functions for voiced change pattern manipulation, clustering method (unused)
# collect voiced change patterns in improved Speech segmentation
# k-means clustering, mean pattern and std for each cluster
# test using moving window, measuring distance between test pattern and trained mean pattern

import os
import numpy as np
from sklearn.cluster import KMeans
from sklearn import preprocessing
from sklearn.externals import joblib
from sklearn.cross_validation import train_test_split, StratifiedKFold
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from vuvAlgos import featureVUV,consonantInterval
from scipy.spatial.distance import pdist,squareform

import matplotlib.pyplot as plt
import essentia.standard as ess

PEAK                 = ess.PeakDetection(maxPeaks=10)

def boundaryFrame(phoSeg,varin):

    # boundary frame from phoneme segmentation, [[pho_start,pho_end],[pho_s,pho_e],...]
    boundary        = [0.0]
    start_time_syllable     = phoSeg[0][0]
    if len(phoSeg) > 1:
        for ii in range(len(phoSeg)-1):
            end_frame = int(round((phoSeg[ii][1]-start_time_syllable)*varin['fs']/varin['hopsize']))
            boundary.append(end_frame)
    end_frame_syllable = int(round((phoSeg[-1][1]-start_time_syllable)*varin['fs']/varin['hopsize']))
    boundary.append(end_frame_syllable)

    return boundary

def scalerPattern(pattern):

    scaler          = preprocessing.StandardScaler().fit(pattern)
    pattern_scaled   = scaler.transform(pattern)
    return pattern_scaled

def reductionPatterns(patterns):

    patterns_stacked    = np.vstack(patterns)
    mX_dist             = squareform(pdist(patterns_stacked,'euclidean'))

    return mX_dist

def patternMeanStd(patterns_voiced_change,varin):

    '''

    :param patterns_voiced_change: contains pattern element, one sample, N_pattern * N_feature dimenstion
    :param varin:
    :return:
    '''

    patterns_voiced_change  = np.vstack(patterns_voiced_change)

    est     = KMeans(n_clusters=varin['n_clusters'], n_jobs=-1)
    est.fit(patterns_voiced_change)
    labels  = est.labels_

    patterns_mean           = []
    patterns_std            = []

    for ii in range(varin['n_clusters']):
        set_patterns    = patterns_voiced_change[labels==ii,:]
        pattern_mean    = np.mean(set_patterns,axis=0)
        pattern_std     = patternStd(set_patterns,pattern_mean)
        patterns_mean.append(pattern_mean)
        patterns_std.append(pattern_std)

    return patterns_mean, patterns_std

def patternStd(set_patterns, pattern_mean):

    '''
    :param set_patterns:
    :param pattern_mean:
    :return: standard deviation of multidimensional array
    '''
    print set_patterns.shape

    ed_array = []
    for ii in range(set_patterns.shape[0]):
        ed_array.append(np.linalg.norm(set_patterns[ii,:] - pattern_mean)**2)

    return np.sqrt(np.mean(ed_array))

def edDistancePatterns(feature_syllable, interval_voiced, mean_patterns, std_patterns, varin):

    # measuring distance between test pattern and trained mean pattern

    N   = varin['N_pattern']
    x_frame_all     = []
    ed_frame_all    = []
    for interval in interval_voiced:
        x_frame     = []
        ed_frame    = []
        start_frame = interval[0]
        end_frame   = interval[1]
        feature_syllable_interval   = feature_syllable[start_frame:end_frame,:]
        if feature_syllable_interval.shape[0] < N:
            continue
        for ii in range(int(N/2),feature_syllable_interval.shape[0]-int(N/2)):
            pattern_test            = feature_syllable_interval[ii-int(N/2):ii+int(N/2)+1,:]
            pattern_test            = scalerPattern(pattern_test)
            pattern_test            = np.reshape(pattern_test, varin['N_feature']*varin['N_pattern'])
            ed_patterns             = []
            for pattern_template in mean_patterns:
                ed_patterns.append(np.linalg.norm(pattern_test-pattern_template))

            x_frame.append(ii+start_frame)
            ed_frame.append(ed_patterns)
        ed_frame    = np.vstack(ed_frame)

        x_frame_all.append(x_frame)
        print ed_frame.shape
        ed_frame_all.append(ed_frame)

    # print len(x_frame_all), len(ed_frame_all)

    if len(x_frame_all):
        x_frame_all     = np.hstack(x_frame_all)
        ed_frame_all    = np.vstack(ed_frame_all)

    # print len(x_frame_all), ed_frame_all.shape

    return x_frame_all, ed_frame_all

def testPatternClassification(feature_syllable, interval_voiced, clf, varin):

    # moving windowing test pattern, classify by using clf

    N   = varin['N_pattern']
    x_frame_all     = []
    target_test_all    = []
    for interval in interval_voiced:
        x_frame     = []
        start_frame = interval[0]
        end_frame   = interval[1]
        feature_syllable_interval   = feature_syllable[start_frame:end_frame,:]
        if feature_syllable_interval.shape[0] < N:
            continue
        patterns_test               = []
        for ii in range(int(N/2),feature_syllable_interval.shape[0]-int(N/2)):
            pattern_test            = feature_syllable_interval[ii-int(N/2):ii+int(N/2)+1,:]
            pattern_test            = scalerPattern(pattern_test)
            pattern_test            = np.reshape(pattern_test, varin['N_feature']*varin['N_pattern'])
            x_frame.append(ii+start_frame)
            patterns_test.append(pattern_test)
        patterns_test    = np.vstack(patterns_test)
        target_test      = clf.predict(patterns_test)

        x_frame_all.append(x_frame)
        target_test_all.append(target_test)

    # print len(x_frame_all), len(ed_frame_all)

    # if len(x_frame_all):
    #     x_frame_all     = np.hstack(x_frame_all)
    #     target_test_all = np.hstack(target_test_all)

    # print len(x_frame_all), ed_frame_all.shape
    print target_test_all

    return x_frame_all, target_test_all

def mergeConsecutiveZeroIndex(x_frame_all,target_test_all):
    # find the consecutive index of zeros, do mean
    zero_frame_means_all    = []
    for ii in range(len(x_frame_all)):
        x_frame     = x_frame_all[ii]
        target_test = target_test_all[ii]
        print target_test

        zero_frame_means    = []
        jj = 0
        while jj < len(target_test):
            x_zeros     = []
            if target_test[jj] == 0:
                x_zeros.append(x_frame[jj])
                kk  = jj+1
                while kk < len(target_test) and target_test[kk] == 0:
                    x_zeros.append(kk)
                    kk  += 1
                jj  = kk
                zero_frame_means.append(np.mean(x_zeros))
            else:
                jj  += 1
        zero_frame_means_all.append(zero_frame_means)

    if len(zero_frame_means_all):
        zero_frame_means_all    = np.hstack(zero_frame_means_all).tolist()

    return zero_frame_means_all

def boundaryReduction(boundary_detected):
    # reduce boundary, if neighbor is <= threshold
    threshold   = 2
    boundary_detected.sort()
    for ii in range(len(boundary_detected))[1:][::-1]:
        if boundary_detected[ii] - boundary_detected[ii-1] <= threshold:
            boundary_detected[ii-1]     = (boundary_detected[ii-1]+boundary_detected[ii])/2.0
            boundary_detected.pop(ii)
    return boundary_detected

def getBoundaryPoint(x_frame, ed_frame, threshold):

    ed_frame[ed_frame>threshold] = threshold

    if len(ed_frame) < 2:
        return []

    pos,_           = PEAK(max(ed_frame) - ed_frame)

    if len(pos):
        pos     = np.array(pos*len(ed_frame), dtype=np.int)
        return x_frame[pos]
    else:
        return []


def featureSyllableSegmentation(feature_path, recording, nestedPhonemeLists,varin):

    '''
    Segment the audio for each syllable
    :param wav_path:
    :param recording:
    :param fs:
    :param nestedPhonemeLists:
    :return:
    feature_syllables is a list containing the feature for each syllable, len(audio_syllables) = len(nestedPhonemeLists)
    '''
    hopsize         = varin['hopsize']
    fs              = varin['fs']

    # mfcc_filename           = os.path.join(feature_path,'mfcc'+'_'+recording+'_'
    #                                            +str(varin['framesize'])+'_'+str(varin['hopsize'])+'.npy')
    # dmfcc_filename          = os.path.join(feature_path,'dmfcc'+'_'+recording+'_'
    #                                            +str(varin['framesize'])+'_'+str(varin['hopsize'])+'.npy')
    spec_filename           = os.path.join(feature_path,'spec'+'_'+recording+'_'
                                               +str(varin['framesize'])+'_'+str(varin['hopsize'])+'.npy')
    mfccBands_filename      = os.path.join(feature_path,'mfccBands'+'_'+recording+'_'
                                               +str(varin['framesize'])+'_'+str(varin['hopsize'])+'.npy')

    # mfcc                    = np.load(mfcc_filename)
    # dmfcc                   = np.load(dmfcc_filename)

    # feature1                = mfcc
    # feature1                = np.hstack((mfcc,dmfcc))
    feature1                = np.load(mfccBands_filename)
    spec                    = np.load(spec_filename)

    # vuv feature
    feature_vuv             = featureVUV(feature_path,recording,varin)

    feature1_syllables      = []
    feature_vuv_syllables   = []
    phoneme_syllables       = []
    spec_syllables          = []
    for nestedPho in nestedPhonemeLists:
        syllable_start_frame    = int(round(nestedPho[0][0]*fs/hopsize))
        syllable_end_frame      = int(round(nestedPho[0][1]*fs/hopsize))


        feature_syllable        = feature1[syllable_start_frame:syllable_end_frame,:]
        feature_vuv_syllable    = feature_vuv[syllable_start_frame:syllable_end_frame,:]
        spec_syllable           = spec[syllable_start_frame:syllable_end_frame,:]
        feature1_syllables.append(feature_syllable)
        feature_vuv_syllables.append(feature_vuv_syllable)
        spec_syllables.append(spec_syllable)
        phoneme_syllables.append(nestedPho[1])

    return feature1_syllables, feature_vuv_syllables, spec_syllables, phoneme_syllables

def patternPadding(feature,boundary_frame,N):

    len_feature     = feature.shape[0]
    if boundary_frame-int(N/2) < 0 and boundary_frame+int(N/2)+1 <= len_feature:
        # left padding
        p           = abs(boundary_frame-int(N/2))
        pad_left    = np.repeat(np.array([feature[0,:]]),p,axis=0)
        pattern     = np.vstack((pad_left,feature[0:boundary_frame+int(N/2)+1,:]))
    elif boundary_frame-int(N/2) >= 0 and boundary_frame+int(N/2)+1 > len_feature:
        # right padding
        q           = abs(boundary_frame+int(N/2)+1 - len_feature)
        pad_right   = np.repeat(np.array([feature[-1,:]]),q,axis=0)
        print 'feature shape' ,feature[boundary_frame-int(N/2):,:].shape
        print 'pad_right shape', pad_right.shape
        pattern     = np.vstack((feature[boundary_frame-int(N/2):,:],pad_right))
    elif boundary_frame-int(N/2) < 0 and boundary_frame+int(N/2)+1 > len_feature:
        # left and right padding
        p           = abs(boundary_frame-int(N/2))
        pad_left    = np.repeat(np.array([feature[0,:]]),p,axis=0)
        q           = abs(boundary_frame+int(N/2)+1 - len_feature)
        pad_right   = np.repeat(np.array([feature[-1,:]]),q,axis=0)
        pattern     = np.vstack((pad_left,feature,pad_right))
    else:
        pattern             = feature[boundary_frame-int(N/2):boundary_frame+int(N/2)+1,:]

    # print pattern.shape

    return pattern

def voicedChangePatternCollection(feature_syllables, phoneme_syllables, varin):

    '''
    :param feature_syllables:
    :param phoneme_syllables:
    :param varin:
    :return: voiced change feature pattern array
    '''

    hopsize         = varin['hopsize']
    fs              = varin['fs']
    N               = varin['N_pattern']

    n_syllable      = len(feature_syllables)

    patterns                = []
    patterns_shifted        = []

    for ii in range(n_syllable):
        phonemeList = phoneme_syllables[ii]
        feature     = feature_syllables[ii]
        len_feature = feature.shape[0]

        if len(phonemeList) < 2:
            continue

        start_time_syllable = phonemeList[0][0]

        for jj in range(len(phonemeList)-1):
            pho0    = phonemeList[jj]
            pho1    = phonemeList[jj+1]

            if pho0[2] in ['c','k','f','x'] or pho1[2] in ['c','k','f','x']:
                continue

            boundary_time_pho   = pho0[1] - start_time_syllable
            boundary_frame_pho  = int(round(boundary_time_pho*fs/hopsize))

            # if boundary_frame_pho-int(N/2) < 0 or boundary_frame_pho+int(N/2)+1 > len_feature:
            #     continue
            #
            # pattern             = feature[boundary_frame_pho-int(N/2):boundary_frame_pho+int(N/2)+1,:]

            pattern             = patternPadding(feature,boundary_frame_pho,N)

            patterns.append(pattern)

            for kk in [-2,-1,1,2]:
                # if boundary_frame_pho-kk-int(N/2) < 0 or boundary_frame_pho-kk+int(N/2)+1 > len_feature:
                #     continue
                #
                # pattern_shifted             = feature[boundary_frame_pho-kk-int(N/2):boundary_frame_pho-kk+int(N/2)+1,:]

                pattern_shifted    = patternPadding(feature,boundary_frame_pho-kk,N)

                patterns_shifted.append(pattern_shifted)

    return patterns, patterns_shifted

def voicedUnchangePatternCollection(feature_syllables, phoneme_syllables, varin):

    hopsize         = varin['hopsize']
    fs              = varin['fs']
    N               = varin['N_pattern']

    n_syllable      = len(feature_syllables)

    patterns        = []

    for ii in range(n_syllable):
        phonemeList = phoneme_syllables[ii]
        feature     = feature_syllables[ii]

        if len(phonemeList) < 2:
            continue

        start_time_syllable = phonemeList[0][0]

        for jj in range(len(phonemeList)):
            pho    = phonemeList[jj]

            if pho[2] in ['c','k','f','x']:
                continue

            start_frame_pho   = int((pho[0] - start_time_syllable)*fs/hopsize)
            end_frame_pho     = int((pho[1] - start_time_syllable)*fs/hopsize)

            # check previous phoneme, if it's not a consonant, push forward the start frame
            if jj > 0 and phonemeList[jj-1][2] not in ['c','k','f','x']:
                start_frame_pho     += int(N/2) + 1

            # check next pho, if it's not a consonant, push back the end frame
            if jj < len(phonemeList)-1 and phonemeList[jj+1][2] not in ['c','k','f','x']:
                end_frame_pho       -= int(N/2)

            if 0 < end_frame_pho - start_frame_pho < N:
                continue

            while start_frame_pho+N <= end_frame_pho:
                pattern     = feature[start_frame_pho:start_frame_pho+N,:]
                patterns.append(pattern)
                start_frame_pho     += N

    return patterns

def getIntervalVoiced(feature_vuv_syllable,varin):

    '''
    delete the unvoiced part, get the voiced interval in frame
    :param feature_vuv_syllable:
    :param varin:
    :return:
    '''

    hopsize         = varin['hopsize']
    fs              = varin['fs']

    models_path         = '/Users/gong/Documents/MTG document/Jingju arias/phonemeSeg/models'
    scaler_filename     = os.path.join(models_path,'VUV_classification','standardization','scaler.pkl')
    svm_model_filename  = os.path.join(models_path,'VUV_classification','svm','svm_model_mfcc_01balanced.pkl')
    scaler              = joblib.load(scaler_filename)
    svm_model_object    = joblib.load(svm_model_filename)

    detectedinterval_vuv = consonantInterval(feature_vuv_syllable,scaler,svm_model_object,varin)

    interval_voiced                 = []
    boundary_frame_unvoiced         = []
    start_frame_interval_voiced     = 0
    if len(detectedinterval_vuv):
        for interval in detectedinterval_vuv:
            start_frame_interval_unvoiced    = int(round(interval[0]*fs/hopsize))
            end_frame_interval_unvoiced      = int(round(interval[1]*fs/hopsize))
            boundary_frame_unvoiced.append(start_frame_interval_unvoiced)
            boundary_frame_unvoiced.append(end_frame_interval_unvoiced)
            if start_frame_interval_unvoiced - start_frame_interval_voiced >= varin['N_pattern']:
                interval_voiced.append([start_frame_interval_voiced,start_frame_interval_unvoiced])
            start_frame_interval_voiced      = end_frame_interval_unvoiced
    if feature_vuv_syllable.shape[0] - start_frame_interval_voiced >= varin['N_pattern']:
        interval_voiced.append([start_frame_interval_voiced,feature_vuv_syllable.shape[0]])

    return interval_voiced, boundary_frame_unvoiced

def svm_cv(fv_train,target_train):

    ####---- cross validation of train dataset, gridsearch the best parameters for svm

    # Set the parameters by cross-validation
    tuned_parameters = [{'kernel': ['rbf'], 'gamma': [0.001, 0.01, 0.1, 1, 10, 100, 1000],
                         'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]}]
    # tuned_parameters = [{'C': [0.01, 0.1, 1, 10, 100, 1000],
    #                      'class_weight': [None, 'balanced']}]

    scores = ['recall_weighted']

    for score in scores:
        print("# Tuning hyper-parameters for %s" % score)
        print()

        mycv = StratifiedKFold(target_train, n_folds = 5)

        # clf = GridSearchCV(LinearSVC(C=1), tuned_parameters, cv=mycv,
        #                    scoring='%s' % score)

        clf = GridSearchCV(SVC(C=1), tuned_parameters, cv=mycv, n_jobs=-1,
                           scoring= score)
        clf.fit(fv_train, target_train)

        report_cv(clf, fv_train, target_train)

def report_cv(clf,fv_test,target_test):

    print("Best parameters set found on development set:")
    print()
    print(clf.best_params_)
    print()
    print("Grid scores on development set:")
    print()
    for params, mean_score, scores in clf.grid_scores_:
        print("%0.3f (+/-%0.03f) for %r"
              % (mean_score, scores.std() * 2, params))
    print()

    print("Detailed classification report:")
    print()
    print("The model is trained on the full development set.")
    print("The scores are computed on the full evaluation set.")
    print()
    target_true, target_pred = target_test, clf.predict(fv_test)
    print(classification_report(target_true, target_pred))
    print()

def svm_model(fv_train,target_train,C,gamma,svm_model_filename):
    # generate the model
    # clf     = LinearSVC(C=C,class_weight=class_weight)
    clf = SVC(C=C,gamma=gamma)
    clf.fit(fv_train,target_train)
    joblib.dump(clf,svm_model_filename)

def plotDetection(spec,boundary_gt,boundary_detected,varin):
    '''
    plot for the predict results for pattern classification method
    boundary_gt:        ground truth boundary in sample
    boundary_detected:  detected boundary
    '''
    boundary_offset = 250
    markersize      = 15
    labelFontsize   = 15
    linewidth       = 3
    N               = 2 * varin['framesize']
    fs              = varin['fs']
    mX              = spec
    mX              = np.transpose(mX)
    maxplotfreq     = 16000.0
    eps             = np.finfo(np.float).eps
    mXPlot          = mX[:int(N*(maxplotfreq/fs))+1,:]
    binFreqs        = np.arange(mXPlot.shape[0])*fs/float(N)
    timestamps_spec = np.arange(mXPlot.shape[1]) * (varin['hopsize']/float(fs))
    # timestamps_spec = np.arange(mXPlot.shape[1])

    # print std_patterns
    # print mean_patterns

    boundary_gt         = boundary_gt[1:-1]
    boundary_detected   = boundary_detected[1:]

    # this plot is for paper, we want more boundary detected
    if len(boundary_gt) > 0 and len(boundary_detected) > 1:
        f, axarr = plt.subplots(1, sharex=True, figsize=(10,4))
        axarr.pcolormesh(timestamps_spec, binFreqs, 20*np.log10(mXPlot+eps))
        '''
        # only plot ground truth boundary
        for bg in boundary_gt:
            axarr.axvline(bg,linewidth=linewidth,color='black')
        '''
        # plot both ground truth and detected

        y = [maxplotfreq-boundary_offset]*len(boundary_gt)
        (markerLines, stemLines, baseLines) =axarr.stem(boundary_gt,y,markerfmt = 'o',bottom=maxplotfreq/2.0)
        plt.setp(markerLines,'markersize',markersize)
        plt.setp(baseLines, visible=False)

        y = [boundary_offset]*len(boundary_detected)
        (markerLines, stemLines, baseLines) = axarr.stem(boundary_detected[0:1],y[0:1],markerfmt = '*',bottom=maxplotfreq/2.0)
        plt.setp(markerLines,'markersize',markersize,'color','b')
        plt.setp(baseLines, visible=False)

        (markerLines, stemLines, baseLines) = axarr.stem(boundary_detected[1:],y[1:],markerfmt = 'D',bottom=maxplotfreq/2.0)
        plt.setp(markerLines,'markersize',markersize,'color','b')
        plt.setp(baseLines, visible=False)

        axarr.axis('tight')

        axarr.set_xlabel('time (s)',fontsize=labelFontsize)
        axarr.set_ylabel('frequency (Hz)',fontsize=labelFontsize)
        plt.tight_layout()

        # axarr[1].plot(x_frame,target_test)
        plt.show()

# if __name__ == '__main__':

    # x_frame_filename = '/Users/gong/Documents/MTG document/Jingju arias/phonemeSeg/eval/rong/x_frame.npy'
    # ed_frame_filename = '/Users/gong/Documents/MTG document/Jingju arias/phonemeSeg/eval/rong/ed_frame.npy'
    # boundary_gt_filename = '/Users/gong/Documents/MTG document/Jingju arias/phonemeSeg/eval/rong/boundary_gt.npy'
    #
    # varin                = {}
    #
    # varin['n_clusters']  = 1000
    #
    # varin['threshold']   = 11.0

    # mX_dist     = reductionPatterns(patterns_voiced_change)
    #
    # for ii in range(mX_dist.shape[0]-1):
    #     for jj in range(ii+1,mX_dist.shape[0]):
    #         if mX_dist[ii,jj] < 10:
    #             print ii,jj
    #
    # print mX_dist.shape

    # mean_patterns   = patterns_voiced_change
    # std_patterns    = [0.0]*len(mean_patterns)

    # for pvc in patterns_voiced_change:
    #     plt.figure()
    #     plt.pcolormesh(np.arange(varin['N_pattern']),np.arange(varin['N_feature']), pvc.T)
    #     plt.show()

    ####--- mean and std of patterns
    # mean_patterns, std_patterns     = patternMeanStd(patterns_voiced_change,varin)

    # for mp in mean_patterns:
    #     plt.figure()
    #     plt.pcolormesh(np.arange(varin['N_pattern']),np.arange(varin['N_feature']), np.transpose(np.reshape(mp,(varin['N_pattern'],varin['N_feature']))))
    #     plt.show()

    '''
    ####---- distance to each pattern, save to .npy
    x_frame_all     = []
    ed_frame_all    = []    # distance
    boundary_gt_all = []
    for ii in range(len(f_vuv_s_test)):
        interval_voiced           = getIntervalVoiced(f_vuv_s_test[ii],varin)
        x_frame, ed_frame         = edDistancePatterns(f_s_test[ii], interval_voiced, mean_patterns, std_patterns, varin)
        boundary_gt               = boundaryFrame(pho_s_test[ii])
        print 'index test syllable ', ii, 'in total ', len(f_vuv_s_test)

        x_frame_all.append(x_frame)
        ed_frame_all.append(ed_frame)
        boundary_gt_all.append(boundary_gt)

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

            for jj in range(varin['n_clusters']):

                f, axarr = plt.subplots(2, sharex=True)
                axarr[0].pcolormesh(timestamps_spec, binFreqs, 20*np.log10(mXPlot+eps))
                for b_gt in boundary_gt:
                    axarr[0].axvline(b_gt)

                axarr[1].plot(x_frame,ed_frame[:,jj])
                plt.title(std_patterns[jj])
                plt.show()

    np.save(x_frame_filename, x_frame_all)
    np.save(ed_frame_filename, ed_frame_all)
    np.save(boundary_gt_filename, boundary_gt_all)
    '''

    '''
    x_frame_all     = np.load(x_frame_filename)
    ed_frame_all    = np.load(ed_frame_filename)
    boundary_gt     = np.load(boundary_gt_filename)

    for ii in range(len(x_frame_all)):
        x_frame     = x_frame_all[ii]
        ed_frame    = ed_frame_all[ii]
        boundary_detected = []
        if len(x_frame):
            for jj in range(ed_frame.shape[1]):
                bd              = getBoundaryPoint(x_frame, ed_frame[:,jj], varin['threshold'])
                boundary_detected.append(bd)
            boundary_detected   = np.hstack(boundary_detected)

        print 'syllable ',ii, boundary_detected, boundary_gt[ii]
    '''