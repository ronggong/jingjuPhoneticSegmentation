import sys,os

sys.path.append(os.path.join(os.path.dirname(__file__), "eval"))
sys.path.append(os.path.join(os.path.dirname(__file__), "public"))
sys.path.append(os.path.join(os.path.dirname(__file__), "VUV"))

import numpy as np
import metrics
import matplotlib.pyplot as plt
from random import randint

from vuvAlgos import featureVUV, consonantInterval
from textgridParser import syllableTextgridExtraction
from sklearn.cross_validation import train_test_split
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.svm import SVC
from sklearn import preprocessing
from sklearn.externals import joblib
from sklearn import mixture

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

def svm_cv(fv_train,target_train,fv_test,target_test):

    ####---- cross validation of train dataset, gridsearch the best parameters for svm

    # Set the parameters by cross-validation
    tuned_parameters = [{'kernel': ['rbf'], 'gamma': [0.001, 0.01, 0.1, 1, 10, 100, 1000],
                         'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]}]
    # tuned_parameters = [{'kernel': ['rbf'], 'gamma': [10],
    #                      'C': [0.001]}]

    scores = ['precision', 'recall']

    for score in scores:
        print("# Tuning hyper-parameters for %s" % score)
        print()

        clf = GridSearchCV(SVC(C=1), tuned_parameters, cv=5,
                           scoring='%s_weighted' % score)
        clf.fit(fv_train, target_train)

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

        # # dump the model into pkl
        # svm_model_filename = os.path.join(models_path,'VUV_classification','svm','best_%s.pkl' % score)
        # joblib.dump(clf,svm_model_filename)

def svm_model(fv_train,target_train,C,gamma,svm_model_filename):
    # generate the model
    clf = SVC(C=C,gamma=gamma,kernel='rbf')
    clf.fit(fv_train,target_train)
    joblib.dump(clf,svm_model_filename)

def svm_predict(textgrid_path,feature_path,scaler_filename,svm_model_filename,recording,varin):

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
            print timestamps_spec.shape, target_predict.shape
            axarr[1].plot(timestamps_spec,target_predict)
            plt.show()

    return sumNumGroundtruthIntervals,sumNumDetectedIntervals,sumNumCorrect

def gmm_cv(fv_train,target_train,fv_test,target_test):

    ####---- cross validation of train dataset, gridsearch the best parameters for gmm

    n_classes = len(np.unique(target_train))
    n_components_max = 7

    n_components_range = range(1, n_components_max)
    cv_types = ['spherical', 'tied', 'diag', 'full']
    for cv_type in cv_types:
        for n_components in n_components_range:
            # Fit a mixture of Gaussians with EM
            gmm = mixture.GMM(n_components=n_components, covariance_type=cv_type)
            gmm.means_ = np.array([fv_train[target_train == i].mean(axis=0)
                                  for i in xrange(n_classes)])
            gmm.fit(fv_train_transformed)

            target_train_pred = gmm.predict(fv_train_transformed)
            train_accuracy = np.mean(target_train_pred == target_train) * 100

            print cv_type, n_components, ' Train accuracy: %.1f' % train_accuracy


if __name__ == '__main__':


    wav_path        = '/Users/gong/Documents/MTG document/Jingju arias/phonemeSeg/wav'
    textgrid_path   = '/Users/gong/Documents/MTG document/Jingju arias/phonemeSeg/textgrid'
    feature_path    = '/Users/gong/Documents/MTG document/Jingju arias/phonemeSeg/feature'
    models_path     = '/Users/gong/Documents/MTG document/Jingju arias/phonemeSeg/models'
    scaler_filename = os.path.join(models_path,'VUV_classification','standardization','scaler.pkl')
    svm_model_filename = os.path.join(models_path,'VUV_classification','svm','svm_model.pkl')

    framesize_t = 0.020
    hopsize_t   = 0.010
    # fs          = 16000
    fs          = 44100

    varin       = {}
    varin['plot']        = False
    varin['fs']          = 44100
    varin['tolerance']   = 0.02

    varin['feature_select'] = 'mfccBands'
    varin['framesize']   = int(round(framesize_t*fs))
    varin['hopsize']     = int(round(hopsize_t*fs))

    ####---- collect recording names
    recordings      = []
    for root, subFolders, files in os.walk(wav_path):
            for f in files:
                file_prefix, file_extension = os.path.splitext(f)
                if file_prefix != '.DS_Store':
                    recordings.append(file_prefix)

    ####---- collect the feature vector class
    fv_voiced_all       = []
    fv_unvoiced_all     = []
    fv_silence_all      = []
    for recording in recordings[:1]:

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

    ####----balance class
    try:
        if sys.argv[2] == 'balance':
            # balance class
            ratio_vuv       = int(round(fv_voiced_all.shape[0]/fv_unvoiced_all.shape[0]))
            for ii in range(ratio_vuv-1):
                fv_unvoiced_all = np.vstack((fv_unvoiced_all,fv_uv_copy))
            ratio_vs        = int(round(fv_voiced_all.shape[0]/fv_silence_all.shape[0]))
            for ii in range(ratio_vs-1):
                fv_silence_all = np.vstack((fv_silence_all,fv_s_copy))

            print fv_voiced_all.shape, fv_unvoiced_all.shape, fv_silence_all.shape
    except:
        pass

    ####---- split the dataset into train and test
    # class target number, 0 voiced, 1 unvoiced, 2 silence
    target_all      = np.array([0]*fv_voiced_all.shape[0]+[1]*fv_unvoiced_all.shape[0]+[0]*fv_silence_all.shape[0])
    fv_all          = np.vstack((fv_voiced_all,fv_unvoiced_all,fv_silence_all))
    fv_train,fv_test,target_train,target_test = train_test_split(fv_all,target_all,test_size=0.25,stratify=target_all)

    ####---- standardization
    scaler          = preprocessing.StandardScaler().fit(fv_train)

    # dump scaler into pkl
    # joblib.dump(scaler,scaler_filename)

    fv_train_transformed    = scaler.transform(fv_train)
    fv_test_transformed     = scaler.transform(fv_test)

    if sys.argv[1] == 'svm_cv':
        svm_cv(fv_train_transformed,target_train,fv_test_transformed,target_test)
    elif sys.argv[1] == 'gmm_cv':
        gmm_cv(fv_train,target_train,fv_test,target_test)
    elif sys.argv[1] == 'svm_model':
        svm_model(fv_train_transformed,target_train,10,0.001,svm_model_filename)
    elif sys.argv[1] == 'svm_predict':
        # recording = recordings[randint(0,len(recordings)-1)]
        numGroundtruthIntervals, numDetectedIntervals, numCorrect = 0,0,0
        for recording in recordings:
            print 'evaluate %s' % recording
            sumNumGroundtruthIntervals, sumNumDetectedIntervals, sumNumCorrect = \
                svm_predict(textgrid_path,feature_path,scaler_filename,svm_model_filename,recording,varin)
            numGroundtruthIntervals    += sumNumGroundtruthIntervals
            numDetectedIntervals       += sumNumDetectedIntervals
            numCorrect                    += sumNumCorrect
        HR, OS, FAR, F, R, deletion, insertion = \
                            metrics.metrics(numDetectedIntervals, numGroundtruthIntervals, numCorrect)
        print ('HR %.3f, OS %.3f, FAR %.3f, F %.3f, R %.3f, deletion %i, insertion %i, gt %i' %
               (HR, OS, FAR, F, R, deletion, insertion,numGroundtruthIntervals))
        print ('gt %i, detected %i, correct %i' % (numGroundtruthIntervals,numDetectedIntervals,numCorrect))
    else:
        print '%s is not a valid classifier name.' % sys.argv[1]