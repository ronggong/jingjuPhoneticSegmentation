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

# the schedule to train and test the VUV SVM model

# 1.find the best parameters C and gamma
# command line: python VUVclassification.py 'svm_cv' 'undersampling'

# 2.using C and gamma of last step, get C and gamma, fv_retrain and target_retrain
# command line: retrain python VUVclassification.py 'svm_cv_retrain' 'undersampling'

# 3.use these C, gamma, fv_retrain, fv_train to train the model:
# command line: python VUVclassification.py 'svm_model' 'undersampling'

# 4.to test the model on the test set:
# command line: python VUVclassification.py 'svm_predict'

##--end--##

import sys,os

sys.path.append(os.path.join(os.path.dirname(__file__), "eval"))
sys.path.append(os.path.join(os.path.dirname(__file__), "public"))
sys.path.append(os.path.join(os.path.dirname(__file__), "VUV"))

from parameters import *

import numpy as np
import metrics
import matplotlib.pyplot as plt
from random import randint,sample

from vuvAlgos import featureVUV, consonantInterval
from textgridParser import syllableTextgridExtraction
from trainTestSeparation import getRecordings,getRecordingNumber
from sklearn.cross_validation import train_test_split, StratifiedKFold
from sklearn.grid_search import GridSearchCV,RandomizedSearchCV
from sklearn.metrics import classification_report
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn import preprocessing
from sklearn.externals import joblib
from sklearn import mixture
from sklearn.ensemble import RandomForestClassifier

def featureClassCollection(feature, nestedPhonemeLists,varin):

    '''
    Collect the feature vectors for voiced, unvoiced and silence classes
    Training purpose

    :param wav_path:
    :param recording:
    :param fs:
    :param nestedPhonemeLists:
    :return:
    '''
    hopsize         = varin['hopsize']
    fs              = varin['fs']

    fv_voiced       = []
    fv_unvoiced     = []
    fv_silence      = []

    for nestedPho in nestedPhonemeLists:
        for nestedNestedPho in nestedPho[1]:
            pho_start_frame    = int(round(nestedNestedPho[0]*fs/hopsize))
            pho_end_frame      = int(round(nestedNestedPho[1]*fs/hopsize))
            pho_feature        = feature[pho_start_frame:pho_end_frame,:]

            if nestedNestedPho[2] in ['c','k','f','x']:
                # unvoiced feature
                fv_unvoiced.append(pho_feature)
            elif not len(nestedNestedPho[2]):
                # silence feature
                fv_silence.append(pho_feature)
            else:
                # voiced feature
                fv_voiced.append(pho_feature)

    return np.vstack(fv_voiced), np.vstack(fv_unvoiced), np.vstack(fv_silence)

def report_cv(clf,fv_test,target_test):

    '''
    generate report for grid search
    :param clf: grid search results
    :param fv_test:
    :param target_test:
    :return:
    '''

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

def svm_cv(fv_train,target_train,fv_test,target_test):

    ####---- cross validation of train dataset, gridsearch the best parameters for svm

    # Set the parameters by cross-validation
    tuned_parameters = [{'kernel': ['rbf'], 'gamma': [0.001, 0.01, 0.1, 1, 10, 100, 1000],
                         'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]}]
    # tuned_parameters = [{'C': [0.1, 1, 10, 100],
    #                      'class_weight': [None, 'balanced']}]

    scores = ['recall_macro']

    for score in scores:
        print("# Tuning hyper-parameters for %s" % score)
        print()

        mycv = StratifiedKFold(target_train, n_folds = 5)

        clf = GridSearchCV(SVC(C=1), tuned_parameters, cv=mycv, n_jobs=-1,
                           scoring='%s' % score)

        # clf = GridSearchCV(LinearSVC(C=1), tuned_parameters, cv=mycv, n_jobs=-1,
        #                    scoring= score)

        clf.fit(fv_train, target_train)

        report_cv(clf,fv_test,target_test)

        # # dump the model into pkl
        # svm_model_filename = os.path.join(models_path,'VUV_classification','svm','best_%s.pkl' % score)
        # joblib.dump(clf,svm_model_filename)

def svm_cv_retrain(fv_train_undersampling,target_train_undersampling,fv_train_all,target_train_all,fv_test,target_test,C,gamma):

    # use best parameters C and gamma train the model
    clf = SVC(C=C,gamma=gamma,kernel='rbf')
    # fit only on undersampling samples
    clf.fit(fv_train_undersampling,target_train_undersampling)

    # predict on all major class feature voiced
    predict_target_train_all    = clf.predict(fv_train_all)

    # pick the samples which is predicted wrongly
    # 0: voiced, 1: unvoiced, 2: silence
    index_fv_voiced_retrain     = np.logical_and(target_train_all == 0, predict_target_train_all != 0)
    # index_fv_silence_retrain    = np.logical_and(target_train_all == 2, predict_target_train_all != 2)
    index_fv_silence_retrain    = target_train_all == 2
    index_fv_unvoiced_retrain   = target_train_all == 1

    # collect the voiced, unvoiced and silence classes samples
    fv_voiced_retrain     = fv_train_all[index_fv_voiced_retrain,:]
    fv_silence_retrain    = fv_train_all[index_fv_silence_retrain,:]
    fv_unvoiced_retrain   = fv_train_all[index_fv_unvoiced_retrain,:]

    fv_retrain, target_retrain  = splitSet(fv_voiced_retrain,fv_unvoiced_retrain,fv_silence_retrain)

    print 'fv_voiced_retrain.shape:', fv_voiced_retrain.shape, 'fv_unvoiced_retrain.shape:', fv_unvoiced_retrain.shape, 'fv_silence_retrain.shape', fv_silence_retrain.shape

    # grid search on retrain training samples
    svm_cv(fv_retrain,target_retrain,fv_test,target_test)

    return fv_retrain,target_retrain

def svm_model(fv_train,target_train,C,gamma,svm_model_filename):
    # generate the model
    clf = SVC(C=C,gamma=gamma,kernel='rbf')
    clf.fit(fv_train,target_train)
    joblib.dump(clf,svm_model_filename)

def predict(textgrid_path,feature_path,scaler_filename,svm_model_filename,recording,varin):

    hopsize         = varin['hopsize']
    fs              = varin['fs']
    framesize       = varin['framesize']
    N               = 2*framesize

    scaler          = joblib.load(scaler_filename)
    svm_model_object= joblib.load(svm_model_filename)

    sumNumGroundtruthIntervals,sumNumDetectedIntervals,sumNumCorrect = 0,0,0

    nestedPhonemeLists, numSyllables, numPhonemes \
            = syllableTextgridExtraction(textgrid_path, recording, 'pinyin', 'details')

    # classification feature
    feature                 = featureVUV(feature_path,recording,varin)

    for ii, nestedPho in enumerate(nestedPhonemeLists):

        print 'evaluate syllable ', ii+1, ' in', len(nestedPhonemeLists)

        syllable_start_frame    = int(round(nestedPho[0][0]*fs/hopsize))
        syllable_end_frame      = int(round(nestedPho[0][1]*fs/hopsize))
        syllable_feature        = feature[syllable_start_frame:syllable_end_frame,:]

        detectedBoundaries_interval = consonantInterval(syllable_feature,scaler,svm_model_object,varin)

        ####---- merge interval into boundaries
        # if detectedBoundaries_interval:
        #     detectedBoundaries = np.hstack(detectedBoundaries_interval)
        # else:
        #     detectedBoundaries = np.array([])
        #
        # detectedBoundaries = detectedBoundaries*hopsize/float(fs)

        # phonemes of syllable
        phoList                 = nestedPhonemeLists[ii][1]
        syllable_start_time     = phoList[0][0]
        groundtruthBoundaries_interval   = []

        for pho in phoList:
            if pho[2] in ['c','k','f','x']:
                groundtruthBoundaries_interval.append([pho[0]-syllable_start_time,pho[1]-syllable_start_time])


        # # evaluate the consonant boundaries
        # numDetectedBoundaries, numGroundtruthBoundaries, numCorrect = \
        #     metrics.boundaryDetection(groundtruthBoundaries=groundtruthBoundaries,
        #                           detectedBoundaries=detectedBoundaries,
        #                           tolerance=varin['tolerance'])

        numDetectedIntervals, numGroundtruthIntervals, numCorrect = \
        metrics.intervalDetection(groundtruthBoundaries_interval,detectedBoundaries_interval,varin['tolerance'])

        # print numGroundtruthBoundaries, numDetectedBoundaries,numCorrect

        sumNumGroundtruthIntervals += numGroundtruthIntervals
        sumNumDetectedIntervals    += numDetectedIntervals
        sumNumCorrect              += numCorrect


        if varin['plot']:
            # load spectrogram
            spec_filename   = os.path.join(feature_path,'spec'+'_'+recording+'_'
                                    +str(varin['framesize'])+'_'+str(varin['hopsize'])+'.npy')
            spec            = np.load(spec_filename)
            syllable_spec   = spec[syllable_start_frame:syllable_end_frame,:]
            binFreqs        = np.arange(syllable_spec.shape[1])*fs/float(N)
            timestamps_spec = np.arange(syllable_spec.shape[0]) * (hopsize/float(fs))

            f, axarr = plt.subplots(2, sharex=True)
            axarr[0].pcolormesh(timestamps_spec,binFreqs,20*np.log10(syllable_spec.T+np.finfo(np.float).eps))
            for interval in detectedBoundaries_interval:
                axarr[0].axvspan(interval[0], interval[1], alpha=0.5, color='red')
            for interval in groundtruthBoundaries_interval:
                axarr[1].axvspan(interval[0], interval[1], alpha=0.5, color='red')
            plt.show()

    return sumNumGroundtruthIntervals,sumNumDetectedIntervals,sumNumCorrect

def gmm_cv(fv_train,target_train,fv_test,target_test):

    ####---- cross validation of train dataset, gridsearch the best parameters for gmm

    n_classes           = len(np.unique(target_train))
    n_components_max    = 7

    n_components_range  = range(1, n_components_max)
    cv_types            = ['spherical', 'tied', 'diag', 'full']
    for cv_type in cv_types:
        for n_components in n_components_range:
            # Fit a mixture of Gaussians with EM
            gmm         = mixture.GMM(n_components=n_components, covariance_type=cv_type)
            gmm.means_  = np.array([fv_train[target_train == i].mean(axis=0)
                                  for i in xrange(n_classes)])
            gmm.fit(fv_train_transformed)

            target_train_pred   = gmm.predict(fv_train_transformed)
            train_accuracy      = np.mean(target_train_pred == target_train) * 100

            print cv_type, n_components, ' Train accuracy: %.1f' % train_accuracy

def rf_cv(fv_train,target_train,fv_test,target_test):

    ####---- cross validation of train dataset, gridsearch the best parameters for random forest

    # Set the parameters by cross-validation
    tuned_parameters = {'n_estimators': [1000, 2000],
                        "max_depth": [3, 6, 9, None],
                        "max_features": ["auto","log2",None],
                        "class_weight": [None, 'balanced']}

    scores = ['recall_macro']

    n_iter_search   = 20

    for score in scores:
        print("# Tuning hyper-parameters for %s" % score)
        print()

        mycv = StratifiedKFold(target_train, n_folds = 5)

        clf = RandomizedSearchCV(RandomForestClassifier(n_jobs=-1), tuned_parameters, cv=mycv, n_iter=n_iter_search,
                           scoring='%s' % score)

        clf.fit(fv_train, target_train)

        report_cv(clf,fv_test,target_test)

def rf_model(fv_train,target_train,n_estimators,max_depth,max_features,class_weight,rf_model_filename):
    # generate the model
    clf = RandomForestClassifier(n_estimators=n_estimators,
                                 max_depth=max_depth,
                                 max_features=max_features,
                                 class_weight=class_weight,
                                 n_jobs=-1)
    clf.fit(fv_train,target_train)
    joblib.dump(clf,rf_model_filename)

def splitSet(fv_voiced_all,fv_unvoiced_all,fv_silence_all):
    # class target number, 0 voiced, 1 unvoiced, 2 silence
    target_all      = np.array([0]*fv_voiced_all.shape[0]+[1]*fv_unvoiced_all.shape[0]+[2]*fv_silence_all.shape[0])
    fv_all          = np.vstack((fv_voiced_all,fv_unvoiced_all,fv_silence_all))
    # fv_train,fv_test,target_train,target_test = train_test_split(fv_all,target_all,test_size=0.25,stratify=target_all)
    # return fv_train,fv_test,target_train,target_test
    return fv_all, target_all

if __name__ == '__main__':

    scaler_filename     = os.path.join(models_path,'VUV_classification','standardization','scaler.pkl')
    # svm_model_filename = os.path.join(models_path,'VUV_classification','svm','svm_model_mfcc_01balanced.pkl')
    svm_model_filename  = os.path.join(models_path,'VUV_classification','svm','svm_model_mfcc_undersampling.pkl')
    rf_model_filename   = os.path.join(models_path,'VUV_classification','rf','rf_model.pkl')

    fv_retrain_filename     = os.path.join(models_path,'VUV_classification','svm','fv_retrain.npy')
    target_retrain_filename = os.path.join(models_path,'VUV_classification','svm','target_retrain.npy')

    # framesize_t = 0.020
    # hopsize_t   = 0.010
    # # fs          = 16000
    # fs          = 44100
    #
    # varin                = {}
    # varin['plot']        = False
    # varin['fs']          = 44100
    # varin['tolerance']   = 0.04
    #
    # varin['feature_select'] = 'mfccBands'   # no use
    # varin['framesize']   = int(round(framesize_t*fs))
    # varin['hopsize']     = int(round(hopsize_t*fs))

    ####---- collect recording names
    recordings           = getRecordings(wav_path)
    number_recording     = getRecordingNumber('TRAIN')

    ####---- collect the feature vector class
    fv_voiced_all       = []
    fv_unvoiced_all     = []
    fv_silence_all      = []
    for ii in number_recording:
        recording       = recordings[ii]

        nestedPhonemeLists, numSyllables, numPhonemes \
            = syllableTextgridExtraction(textgrid_path, recording, 'pinyin', 'details')

        feature                            = featureVUV(feature_path,recording,varin)
        fv_voiced, fv_unvoiced, fv_silence = featureClassCollection(feature, nestedPhonemeLists,varin)

        fv_voiced_all.append(fv_voiced)
        fv_unvoiced_all.append(fv_unvoiced)
        fv_silence_all.append(fv_silence)


    fv_voiced_all   = np.vstack(fv_voiced_all)
    fv_unvoiced_all = np.vstack(fv_unvoiced_all)
    fv_silence_all  = np.vstack(fv_silence_all)

    fv_v_copy       = np.copy(fv_voiced_all)
    fv_uv_copy      = np.copy(fv_unvoiced_all)
    fv_s_copy       = np.copy(fv_silence_all)


    ####---- reduce the sample size for testing
    # fv_voiced_all   = fv_voiced_all[sample(xrange(len(fv_voiced_all)),int(len(fv_voiced_all)/10)),:]
    # fv_unvoiced_all = fv_unvoiced_all[sample(xrange(len(fv_unvoiced_all)),int(len(fv_unvoiced_all)/10)),:]
    # fv_silence_all  = fv_silence_all[sample(xrange(len(fv_silence_all)),int(len(fv_silence_all)/10)),:]

    print 'voiced shape: ',fv_voiced_all.shape,' unvoiced shape: ', fv_unvoiced_all.shape, ' silence shape: ', fv_silence_all.shape

    ####----balance class

    fv_train,fv_test,target_train,target_test = [],[],[],[]
    if sys.argv[2] == 'oversampling':
        # balance class
        ratio_vuv       = int(round(fv_voiced_all.shape[0]/fv_unvoiced_all.shape[0]))
        for ii in range(ratio_vuv-1):
            fv_unvoiced_all = np.vstack((fv_unvoiced_all,fv_uv_copy))
        ratio_vs        = int(round(fv_voiced_all.shape[0]/fv_silence_all.shape[0]))
        for ii in range(ratio_vs-1):
            fv_silence_all = np.vstack((fv_silence_all,fv_s_copy))

        fv_train,target_train = splitSet(fv_voiced_all,fv_unvoiced_all,fv_silence_all)

        print 'oversampling'


    elif sys.argv[2] == 'undersampling':

        # in this example, fv_unvoiced has the least samples
        # randomly select len(fv_unvoiced_all) samples from voiced and silence features

        index_voiced_train  = sample(range(len(fv_voiced_all)), len(fv_unvoiced_all))
        index_silence_train = sample(range(len(fv_silence_all)), len(fv_unvoiced_all))

        fv_unvoiced_train   = fv_unvoiced_all
        fv_voiced_train     = fv_voiced_all[index_voiced_train,:]
        fv_silence_train    = fv_silence_all[index_silence_train,:]

        fv_train_undersampling,target_train_undersampling   = splitSet(fv_voiced_train,fv_unvoiced_train,fv_silence_train)
        fv_train,target_train                               = splitSet(fv_voiced_all,fv_unvoiced_all,fv_silence_all)

        print 'undersampling'
        print fv_voiced_train.shape, fv_unvoiced_train.shape, fv_silence_train.shape

    else:
        ####---- split the dataset into train and test
        print 'normal split'
        fv_train,target_train = splitSet(fv_voiced_all,fv_unvoiced_all,fv_silence_all)


    ####---- standardization
    scaler          = preprocessing.StandardScaler().fit(fv_train)

    if sys.argv[1] != 'svm_predict':
        # dump scaler into pkl
        joblib.dump(scaler,scaler_filename)

    fv_train_transformed        = scaler.transform(fv_train)

    if sys.argv[2] == 'undersampling':
        fv_train_undersampling_transformed    = scaler.transform(fv_train_undersampling)

    if sys.argv[1] == 'svm_cv' and sys.argv[2] == 'undersampling':
        svm_cv(fv_train_undersampling_transformed,target_train_undersampling,fv_train_transformed,target_train)

    elif sys.argv[1] == 'svm_cv' and sys.argv[2] != 'undersampling':
        svm_cv(fv_train_transformed,target_train,fv_train_transformed,target_train)

    elif sys.argv[1] == 'svm_cv_retrain':
        fv_retrain,target_retrain = svm_cv_retrain(fv_train_undersampling_transformed,target_train_undersampling,fv_train_transformed,target_train,
                                    fv_train_transformed,target_train,C=10,gamma=0.01)
        np.save(fv_retrain_filename,fv_retrain)
        np.save(target_retrain_filename,target_retrain)

    elif sys.argv[1] == 'svm_model':
        # the best score of mfcc obtained by C=0.1, class_weight='balanced', linearSVC
        # the best score of mfcc obtained by C=1, gamma=0.01, svm model, SVC rbf kernel
        fv_retrain      = np.load(fv_retrain_filename)
        target_retrain  = np.load(target_retrain_filename)
        svm_model(fv_retrain,target_retrain,C=1,gamma=0.01,svm_model_filename=svm_model_filename)

    elif sys.argv[1] == 'svm_predict':
        model_filename = svm_model_filename
        # recording = recordings[randint(0,len(recordings)-1)]
        numGroundtruthIntervals, numDetectedIntervals, numCorrect = 0,0,0
        number_recording    = getRecordingNumber('TEST')
        for ii in number_recording:
            recording       = recordings[ii]
            print 'evaluate %s' % recording
            sumNumGroundtruthIntervals, sumNumDetectedIntervals, sumNumCorrect = \
                predict(textgrid_path,feature_path,scaler_filename,model_filename,recording,varin)
            numGroundtruthIntervals    += sumNumGroundtruthIntervals
            numDetectedIntervals       += sumNumDetectedIntervals
            numCorrect                 += sumNumCorrect
        HR, OS, FAR, F, R, deletion, insertion = \
                            metrics.metrics(numDetectedIntervals, numGroundtruthIntervals, numCorrect)
        print ('HR %.3f, OS %.3f, FAR %.3f, F %.3f, R %.3f, deletion %i, insertion %i, gt %i' %
               (HR, OS, FAR, F, R, deletion, insertion,numGroundtruthIntervals))
        print ('gt %i, detected %i, correct %i' % (numGroundtruthIntervals,numDetectedIntervals,numCorrect))

    # not used in the paper
    elif sys.argv[1] == 'gmm_cv':
        gmm_cv(fv_train,target_train,fv_train,target_train)
    elif sys.argv[1] == 'rf_cv':
        rf_cv(fv_train_transformed,target_train,fv_train_transformed,target_train)
    elif sys.argv[1] == 'rf_model':
        rf_model(fv_train,
                 target_train,
                 n_estimators=1000,
                 max_depth=None,
                 max_features='auto',
                 class_weight=None,
                 rf_model_filename=rf_model_filename)

    else:
        print '%s is not a valid classifier name.' % sys.argv[1]