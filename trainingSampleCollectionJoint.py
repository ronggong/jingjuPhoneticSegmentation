import os
import numpy as np
import h5py
import pickle
import cPickle
import gzip
from general.trainTestSeparation import getTestTrainRecordingsJoint
from general.textgridParser import textGrid2WordList, wordListsParseByLines
from lyricsRecognizer.audioPreprocessing import getMFCCBands2DMadmom
from lyricsRecognizer.audioPreprocessing import featureReshape
from sklearn import preprocessing
from parameters import *
from general.filePathJoint import *

def removeOutOfRange(frames, frame_start, frame_end):
    return frames[np.all([frames <= frame_end, frames >= frame_start], axis=0)]

def simpleSampleWeighting(mfcc, frames_onset_s_p, frames_onset_p, frame_start, frame_end):
    """
    simple weighing strategy used in Schluter's paper
    :param mfcc:
    :param frames_onset:
    :param frame_start:
    :param frame_end:
    :return:
    """

    frames_onset_s_p25 = np.hstack((frames_onset_s_p - 1, frames_onset_s_p + 1))
    frames_onset_s_p25 = removeOutOfRange(frames_onset_s_p25, frame_start, frame_end)

    frames_onset_p25 = np.hstack((frames_onset_p - 1, frames_onset_p + 1))
    frames_onset_p25 = removeOutOfRange(frames_onset_p25, frame_start, frame_end)
    # print(frames_onset_p75, frames_onset_p50, frames_onset_p25)

    # mfcc positive
    mfcc_s_p100 = mfcc[frames_onset_s_p, :]
    mfcc_s_p25 = mfcc[frames_onset_s_p25, :]

    mfcc_p100 = mfcc[frames_onset_p, :]
    mfcc_p25 = mfcc[frames_onset_p25, :]

    frames_all = np.arange(frame_start, frame_end)
    frames_n100 = np.setdiff1d(frames_all, np.hstack((frames_onset_s_p,
                                                      frames_onset_s_p25,
                                                      frames_onset_p,
                                                      frames_onset_p25)))
    # print(frames_n100.shape, frames_all.shape)
    mfcc_n100 = mfcc[frames_n100, :]

    mfcc_s_p = np.concatenate((mfcc_s_p100, mfcc_s_p25), axis=0)
    sample_weights_s_p = np.concatenate((np.ones((mfcc_s_p100.shape[0],)),
                                       np.ones((mfcc_s_p25.shape[0],)) * 0.25))

    mfcc_p = np.concatenate((mfcc_p100, mfcc_p25), axis=0)
    sample_weights_p_syllable = np.concatenate((np.ones((mfcc_p100.shape[0],)),
                                                np.ones((mfcc_p25.shape[0],))))

    sample_weights_p_phoneme = np.concatenate((np.ones((mfcc_p100.shape[0],)),
                                                np.ones((mfcc_p25.shape[0],)) * 0.25))

    mfcc_n = mfcc_n100
    sample_weights_n = np.ones((mfcc_n100.shape[0],))

    return mfcc_s_p, mfcc_p, mfcc_n, sample_weights_s_p, sample_weights_p_syllable, sample_weights_p_phoneme, sample_weights_n


def featureLabelOnsetH5py(filename_mfcc_s_p, filename_mfcc_p, filename_mfcc_n, scaling=True):
    '''
    organize the training feature and label
    :param
    :return:
    '''

    # feature_all = np.concatenate((mfcc_p, mfcc_n), axis=0)
    f_mfcc_s_p = h5py.File(filename_mfcc_s_p, 'a')
    f_mfcc_p = h5py.File(filename_mfcc_p, 'r')
    f_mfcc_n = h5py.File(filename_mfcc_n, 'r')

    dim_s_p_0 = f_mfcc_s_p['mfcc_s_p'].shape[0]
    dim_p_0 = f_mfcc_p['mfcc_p'].shape[0]
    dim_n_0 = f_mfcc_n['mfcc_n'].shape[0]
    dim_1 = f_mfcc_p['mfcc_p'].shape[1]

    label_s_p = [1] * dim_s_p_0
    label_p_syllable = [0] * dim_p_0 # phoneme onset label for syllable detection
    label_p_phoneme = [1] * dim_p_0 # phoneme onset label for phoneme detection
    label_n = [0] * dim_n_0

    label_all_syllable = label_s_p + label_p_syllable + label_n
    label_all_phoneme = label_s_p + label_p_phoneme + label_n

    label_all_syllable = np.array(label_all_syllable,dtype='int64')
    label_all_phoneme = np.array(label_all_phoneme,dtype='int64')

    feature_all = np.zeros((dim_s_p_0+dim_p_0+dim_n_0, dim_1), dtype='float32')

    print('concatenate features... ...')

    feature_all[:dim_s_p_0, :] = f_mfcc_s_p['mfcc_s_p']
    feature_all[dim_s_p_0:dim_s_p_0+dim_p_0, :] = f_mfcc_p['mfcc_p']
    feature_all[dim_s_p_0+dim_p_0:, :] = f_mfcc_n['mfcc_n']

    f_mfcc_s_p.flush()
    f_mfcc_s_p.close()
    f_mfcc_p.flush()
    f_mfcc_p.close()
    f_mfcc_n.flush()
    f_mfcc_n.close()

    print('scaling features... ... ')

    scaler = preprocessing.StandardScaler()
    scaler.fit(feature_all)
    if scaling:
        feature_all = scaler.transform(feature_all)

    return feature_all, label_all_syllable, label_all_phoneme, scaler


def dumpFeatureOnsetHelper(wav_path,
                           textgrid_path,
                           artist_name,
                           recording_name):

    groundtruth_textgrid_file = os.path.join(textgrid_path, artist_name, recording_name + '.TextGrid')
    print(groundtruth_textgrid_file)
    wav_file = os.path.join(wav_path, artist_name, recording_name + '.wav')

    lineList = textGrid2WordList(groundtruth_textgrid_file, whichTier='line')
    utteranceList = textGrid2WordList(groundtruth_textgrid_file, whichTier='dianSilence')
    phonemeList = textGrid2WordList(groundtruth_textgrid_file, whichTier='details')

    # parse lines of groundtruth
    nestedUtteranceLists, numLines, numUtterances = wordListsParseByLines(lineList, utteranceList)
    nestedPhonemeLists, _, _ = wordListsParseByLines(lineList, phonemeList)

    # load audio
    fs = 44100
    mfcc = getMFCCBands2DMadmom(wav_file, fs, hopsize_t, channel=1)

    return nestedUtteranceLists, nestedPhonemeLists, mfcc, phonemeList


def getFrameOnset(u_list):

    times_onset = [u[0] for u in u_list[1]]

    # syllable onset frames
    frames_onset = np.array(np.around(np.array(times_onset) / hopsize_t), dtype=int)

    # line start and end frames
    frame_start = frames_onset[0]
    frame_end = int(u_list[0][1] / hopsize_t)

    return frames_onset, frame_start, frame_end



def dumpFeatureOnset(wav_path,
                     textgrid_path,
                     recordings):
    '''
    :param recordings:
    :return:
    '''

    # p: position, n: negative, 75: 0.75 sample_weight
    mfcc_s_p_all = []
    mfcc_p_all = []
    mfcc_n_all = []
    sample_weights_s_p_all = []
    sample_weights_p_syllable_all = []
    sample_weights_p_phoneme_all = []
    sample_weights_n_all = []

    for artist_name, recording_name in recordings:

        nestedUtteranceLists, nestedPhonemeLists, mfcc, phonemeList = dumpFeatureOnsetHelper(wav_path,
                                                                                             textgrid_path,
                                                                                             artist_name,
                                                                                             recording_name)

        for ii_line, line in enumerate(nestedUtteranceLists):
            list_syllable = nestedUtteranceLists[ii_line]
            list_phoneme = nestedPhonemeLists[ii_line]

            onsets_syllable, frame_start_line_syllable, frame_end_line_syllable = getFrameOnset(list_syllable)
            onsets_phoneme, frame_start_line_phoneme, frame_end_line_phoneme = getFrameOnset(list_phoneme)

            if not set(onsets_syllable).issubset(onsets_phoneme) or frame_start_line_syllable != frame_start_line_phoneme \
                or frame_end_line_syllable != frame_end_line_phoneme:
                raise

            frames_onset_s_p = onsets_syllable # simultaneously syllable and phoneme onsets
            frames_onset_p = np.array([o for o in onsets_phoneme if o not in onsets_syllable], dtype=int) # only phoneme onsets

            mfcc_s_p, \
            mfcc_p, \
            mfcc_n, \
            sample_weights_s_p, \
            sample_weights_p_syllable, \
            sample_weights_p_phoneme, \
            sample_weights_n = \
                simpleSampleWeighting(mfcc, frames_onset_s_p, frames_onset_p, frame_start_line_syllable, frame_end_line_syllable)

            mfcc_s_p_all.append(mfcc_s_p)
            mfcc_p_all.append(mfcc_p)
            mfcc_n_all.append(mfcc_n)
            sample_weights_s_p_all.append(sample_weights_s_p)
            sample_weights_p_syllable_all.append(sample_weights_p_syllable)
            sample_weights_p_phoneme_all.append(sample_weights_p_phoneme)
            sample_weights_n_all.append(sample_weights_n)

            # print(len(mfcc_p_all), len(mfcc_n_all), len(sample_weights_p_all), len(sample_weights_n_all))

    return np.concatenate(mfcc_s_p_all), \
           np.concatenate(mfcc_p_all), \
           np.concatenate(mfcc_n_all), \
           np.concatenate(sample_weights_s_p_all), \
           np.concatenate(sample_weights_p_syllable_all), \
           np.concatenate(sample_weights_p_phoneme_all), \
           np.concatenate(sample_weights_n_all)


def dumpFeatureBatchOnset():
    """
    dump features for all the dataset for onset detection
    :return:
    """

    testPrimarySchool, trainNacta2017, trainNacta = getTestTrainRecordingsJoint()

    nacta_data = trainNacta
    nacta_data_2017 = trainNacta2017
    scaling = True

    mfcc_s_p_nacta2017, \
    mfcc_p_nacta2017,\
    mfcc_n_nacta2017, \
    sample_weights_s_p_nacta2017, \
    sample_weights_p_syllable_nacta2017, \
    sample_weights_p_phoneme_nacta2017, \
    sample_weights_n_nacta2017 = \
        dumpFeatureOnset(wav_path=nacta2017_wav_path,
                   textgrid_path=nacta2017_textgrid_path,
                   recordings=nacta_data_2017)

    mfcc_s_p_nacta, \
    mfcc_p_nacta, \
    mfcc_n_nacta, \
    sample_weights_s_p_nacta, \
    sample_weights_p_syllable_nacta, \
    sample_weights_p_phoneme_nacta, \
    sample_weights_n_nacta =    \
        dumpFeatureOnset(wav_path=nacta_wav_path,
                     textgrid_path=nacta_textgrid_path,
                     recordings=nacta_data)

    print('finished feature extraction.')

    mfcc_s_p = np.concatenate((mfcc_s_p_nacta2017, mfcc_s_p_nacta))
    mfcc_p = np.concatenate((mfcc_p_nacta2017, mfcc_p_nacta))
    mfcc_n = np.concatenate((mfcc_n_nacta2017, mfcc_n_nacta))

    sample_weights_s_p = np.concatenate((sample_weights_s_p_nacta2017, sample_weights_s_p_nacta))
    sample_weights_p_syllable = np.concatenate((sample_weights_p_syllable_nacta2017, sample_weights_p_syllable_nacta))
    sample_weights_p_phoneme = np.concatenate((sample_weights_p_phoneme_nacta2017, sample_weights_p_phoneme_nacta))
    sample_weights_n = np.concatenate((sample_weights_n_nacta2017, sample_weights_n_nacta))

    sample_weights_syllable = np.concatenate((sample_weights_s_p, sample_weights_p_syllable, sample_weights_n))
    sample_weights_phoneme = np.concatenate((sample_weights_s_p, sample_weights_p_phoneme, sample_weights_n))

    filename_mfcc_s_p = join(feature_data_path, 'mfcc_s_p_joint.h5')
    h5f = h5py.File(filename_mfcc_s_p, 'w')
    h5f.create_dataset('mfcc_s_p', data=mfcc_s_p)
    h5f.close()

    filename_mfcc_p = join(feature_data_path, 'mfcc_p_joint.h5')
    h5f = h5py.File(filename_mfcc_p, 'w')
    h5f.create_dataset('mfcc_p', data=mfcc_p)
    h5f.close()

    filename_mfcc_n = join(feature_data_path, 'mfcc_n_joint.h5')
    h5f = h5py.File(filename_mfcc_n, 'w')
    h5f.create_dataset('mfcc_n', data=mfcc_n)
    h5f.close()

    del mfcc_s_p
    del mfcc_p
    del mfcc_n

    feature_all, label_all_syllable, label_all_phoneme, scaler = \
        featureLabelOnsetH5py(filename_mfcc_s_p, filename_mfcc_p, filename_mfcc_n, scaling=scaling)

    os.remove(filename_mfcc_s_p)
    os.remove(filename_mfcc_p)
    os.remove(filename_mfcc_n)

    nlen = 7
    feature_all = featureReshape(feature_all, nlen=nlen)

    print('feature shape:', feature_all.shape)

    filename_feature_all = join(feature_data_path, 'feature_all_joint.h5')
    h5f = h5py.File(filename_feature_all, 'w')
    h5f.create_dataset('feature_all', data=feature_all)
    h5f.close()

    print('finished feature concatenation.')

    cPickle.dump(label_all_syllable,
                 gzip.open(
                     '../trainingData/labels_joint_syllable.pickle.gz',
                     'wb'), cPickle.HIGHEST_PROTOCOL)

    cPickle.dump(label_all_phoneme,
                 gzip.open(
                     '../trainingData/labels_joint_phoneme.pickle.gz',
                     'wb'), cPickle.HIGHEST_PROTOCOL)

    cPickle.dump(sample_weights_syllable,
                 gzip.open('../trainingData/sample_weights_joint_syllable.pickle.gz',
                           'wb'), cPickle.HIGHEST_PROTOCOL)

    cPickle.dump(sample_weights_phoneme,
                 gzip.open('../trainingData/sample_weights_joint_phoneme.pickle.gz',
                           'wb'), cPickle.HIGHEST_PROTOCOL)

    print(feature_all.shape)
    print(label_all_syllable.shape, label_all_phoneme.shape)
    print(sample_weights_syllable.shape, sample_weights_phoneme.shape)

    pickle.dump(scaler,
                open('../cnnModels/scaler_joint.pkl', 'wb'))

if __name__ == '__main__':
    dumpFeatureBatchOnset()