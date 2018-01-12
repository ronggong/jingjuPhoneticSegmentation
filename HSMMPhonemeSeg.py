import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches


import numpy as np
import pyximport
pyximport.install(reload_support=True,
                  setup_args={'include_dirs': np.get_include()})

from lyricsRecognizer.LRHSMM import LRHSMM
from lyricsRecognizer.makeHSMMNet import singleTransMatBuild
from lyricsRecognizer.audioPreprocessing import getMFCCBands2DMadmom, featureReshape
# from lyricsRecognizer.audioPreprocessing import mfccFeature_pipeline

from onsetSegmentEval.utilFunctions import removeSilence

from general.filePathHsmm import *
from general.parameters import *
from general.textgridParser import textGrid2WordList
from general.textgridParser import wordListsParseByLines
from general.trainTestSeparation import getTestTrainRecordingsJoint
from general.phonemeMap import dic_pho_map
from helperCode import results_aggregation_save_helper
from helperCode import gt_score_preparation_helper
import os
from onsetSegmentEval.runEval import run_eval_onset
from onsetSegmentEval.runEval import run_eval_segment


def textgridSyllablePhonemeParser(textgrid_file, tier1, tier2):
    """
    Parse the textgrid file,
    :param textgrid_file:
    :return: syllable and phoneme lists
    """
    lineList = textGrid2WordList(textgrid_file, whichTier='line')
    syllableList = textGrid2WordList(textgrid_file, whichTier=tier1)
    phonemeList = textGrid2WordList(textgrid_file, whichTier=tier2)

    # parse lines of groundtruth
    nestedSyllableLists, _, _ = wordListsParseByLines(lineList, syllableList)
    nestedPhonemeLists, _, _ = wordListsParseByLines(lineList, phonemeList)

    return nestedSyllableLists, nestedPhonemeLists


def figurePlot(mfcc_line,
               syllable_gt_onsets_0start,
               phoneme_gt_onsets_0start_without_syllable_onsets,
               hsmm,
               phoneme_score_labels,
               path,
               boundaries_phoneme_start_time,
               boundaries_syllable_start_time,
               syllable_score_durs,
               phoneme_score_durs):

    # plot Error analysis figures
    # plt.figure(figsize=(16, 6))
    tPlot, axes = plt.subplots(
        nrows=3, ncols=1, sharex=True, sharey=False,
        gridspec_kw={'height_ratios': [1, 2, 1]})
    ax1, ax2, ax3 = axes[0], axes[1], axes[2]
    # plt.figure(figsize=(8, 4))
    # class weight
    # ax1 = plt.subplot(311)
    y = np.arange(0, 80)
    x = np.arange(0, mfcc_line.shape[0]) * hopsize_t
    cax = ax1.pcolormesh(x, y, np.transpose(mfcc_line[:, 80 * 10:80 * 11]))
    for gso in syllable_gt_onsets_0start:
        ax1.axvline(gso, color='r', linewidth=2)
    for gpo in phoneme_gt_onsets_0start_without_syllable_onsets:
        ax1.axvline(gpo, color='k', linewidth=2)

    # cbar = fig.colorbar(cax)
    ax1.set_ylabel('Mel bands', fontsize=12)
    ax1.get_xaxis().set_visible(False)
    ax1.axis('tight')
    # plt.title('Calculating: '+rn+' phrase '+str(i_obs))


    # ax2 = plt.subplot(312, sharex=ax1)
    # plot observation proba matrix
    B_map = hsmm._getBmap()
    n_states = B_map.shape[0]
    n_frame = B_map.shape[1]
    y = np.arange(n_states + 1)
    x = np.arange(n_frame) * hopsize / float(fs)
    ax2.pcolormesh(x, y, B_map)
    ax2.set_yticks(y)
    ax2.set_yticklabels(phoneme_score_labels, fontdict={'fontsize': 6})
    # plot the decoding path
    ax2.plot(x, path, 'b', linewidth=1)
    for bpst in boundaries_phoneme_start_time:
        ax2.axvline(bpst, color='k', linewidth=1)
    for bsst in boundaries_syllable_start_time:
        ax2.axvline(bsst, color='r', linewidth=2)

    ax2.axis('tight')

    # ax3 = plt.subplot(313, sharex=ax1)
    # print(duration_score)
    time_start = 0
    # print(syllable_score_durs)
    for ii_ds, ds in enumerate(syllable_score_durs):
        ax3.add_patch(
            patches.Rectangle(
                (time_start, 0),  # (x,y)
                ds,  # width
                3,  # height
                alpha=0.5
            ))
        time_start += ds

    time_start = 0
    for psd in phoneme_score_durs:
        ax3.add_patch(
            patches.Rectangle(
                (time_start, 0.5),  # (x,y)
                psd,  # width
                2,  # height
                color='r',
                alpha=0.5
            ))
        time_start += psd
    ax3.set_ylim((0, 3))
    ax3.set_ylabel('Score duration', fontsize=12)
    plt.xlabel('Time (s)')
    # plt.tight_layout()

    plt.show()


    # hsmm._pathPlot(None,path_gt,path)


def phonemeSegAllRecordings(wav_path,
                           textgrid_path,
                           scaler,
                           test_recordings,
                           model_keras_cnn_0,
                            cnnModel_name,
                           eval_results_path,
                           threshold=0.54,
                           obs_cal='tocal',
                           decoding_method='viterbi',
                            plot=False):
    """
    ODF and viterbi decoding
    :param recordings:
    :param textgrid_path:
    :param dataset_path:
    :param feature_type: 'mfcc', 'mfccBands1D' or 'mfccBands2D'
    :param dmfcc: delta for 'mfcc'
    :param nbf: context frames
    :param mth: jordi, jordi_horizontal_timbral, jan, jan_chan3
    :param late_fusion: Bool
    :return:
    """

    for artist_path, rn in test_recordings:
        # rn = rn.split('.')[0]

        # take the teacher's textgrid as the score
        score_textgrid_file = join(textgrid_path, artist_path, 'teacher.TextGrid')

        groundtruth_textgrid_file   = join(textgrid_path, artist_path, rn+'.TextGrid')

        wav_file = join(wav_path, artist_path, rn + '.wav')

        scoreSyllableLists, scorePhonemeLists = textgridSyllablePhonemeParser(score_textgrid_file,
                                                                              'dianSilence',
                                                                              'details')
        gtSyllableLists, gtPhonemeLists = textgridSyllablePhonemeParser(groundtruth_textgrid_file,
                                                                        'dianSilence',
                                                                        'details')

        # calculate mfcc
        mfcc = getMFCCBands2DMadmom(wav_file, fs, hopsize_t, channel=1)
        mfcc_scaled = scaler.transform(mfcc)
        mfcc_reshaped = featureReshape(mfcc_scaled, nlen=7)

        # essentia mfcc
        # mfcc_reshaped, mfcc = mfccFeature_pipeline(wav_file)

        for ii_line in range(len(gtSyllableLists)):
            frame_start, frame_end, \
            time_start, time_end, \
            syllable_gt_onsets, syllable_gt_labels, \
            phoneme_gt_onsets, phoneme_gt_labels, \
            syllable_score_onsets, syllable_score_labels, \
            phoneme_score_onsets, phoneme_score_labels, \
            syllable_score_durs, phoneme_list_score = \
                            gt_score_preparation_helper(gtSyllableLists,
                                                        scoreSyllableLists,
                                                        gtPhonemeLists,
                                                        scorePhonemeLists,
                                                        ii_line)

            # phoneme durations and labels
            phoneme_score_durs = []
            idx_syllable_score_phoneme = [] # index of syllable onsets in phoneme onsets list
            for ii_pls, pls in enumerate(phoneme_list_score):
                # where meeting next syllable or is the last phoneme
                phoneme_score_durs.append(pls[1] - pls[0])

                if pls[0] in syllable_score_onsets:
                    idx_syllable_score_phoneme.append(ii_pls)

            # map the phone labels
            phoneme_score_labels_mapped = [dic_pho_map[l] for l in phoneme_score_labels]

            # normalize phoneme score durations
            phoneme_score_durs = np.array(phoneme_score_durs)
            phoneme_score_durs *= (time_end - time_start) / np.sum(phoneme_score_durs)

            # onsets start from time 0, syllable and phoneme onsets
            syllable_gt_onsets_0start = np.array(syllable_gt_onsets) - syllable_gt_onsets[0]
            phoneme_gt_onsets_0start = np.array(phoneme_gt_onsets) - phoneme_gt_onsets[0]
            phoneme_gt_onsets_0start_without_syllable_onsets = \
                np.setdiff1d(phoneme_gt_onsets_0start, syllable_gt_onsets_0start)

            # check the annotations, if syllable onset are also phoneme onsets
            if not set(syllable_gt_onsets).issubset(set(phoneme_gt_onsets)):
                raise
            if not set(syllable_score_onsets).issubset(set(phoneme_score_onsets)):
                raise

            # line level mfcc
            mfcc_line          = mfcc[frame_start:frame_end]
            mfcc_reshaped_line = mfcc_reshaped[frame_start:frame_end]
            mfcc_reshaped_line = np.expand_dims(mfcc_reshaped_line, axis=1)

            # transition matrix
            mat_tran = singleTransMatBuild(phoneme_score_labels_mapped)

            # initialize the the HSMM
            hsmm = LRHSMM(mat_tran,
                          phoneme_score_labels_mapped,
                          phoneme_score_durs,
                          proportionality_std=0.1)

            path, posteri_proba = hsmm._viterbiHSMM(observations=mfcc_reshaped_line,
                                                    kerasModel=model_keras_cnn_0)

            # construct ground truth path
            phoneme_gt_onsets_0start_frame = list(np.floor(phoneme_gt_onsets_0start * (len(path)/(time_end-time_start))))
            path_gt = np.zeros((len(path),), dtype='int')
            state_num = 0
            for ii_path in range(len(path)):
                if ii_path in phoneme_gt_onsets_0start_frame[1:]:
                    state_num += 1
                path_gt[ii_path] = state_num

            # detected phoneme onsets
            phoneme_start_frame = [0]
            for ii_path in range(len(path)-1):
                if path[ii_path] != path[ii_path+1]:
                    phoneme_start_frame.append(ii_path+1)

            boundaries_phoneme_start_time = list(np.array(phoneme_start_frame)*(time_end-time_start)/len(path))
            boundaries_syllable_start_time = [boundaries_phoneme_start_time[ii_bpst]
                                              for ii_bpst in range(len(boundaries_phoneme_start_time))
                                              if ii_bpst in idx_syllable_score_phoneme]

            # print(syllable_gt_onsets_0start)
            # print(syllable_gt_labels)
            # print(boundaries_syllable_start_time)
            # print(syllable_score_labels)
            #
            # print(phoneme_gt_onsets_0start)
            # print(phoneme_gt_labels)
            # print(boundaries_phoneme_start_time)
            # print(phoneme_score_labels)

            # remove the silence phonemes
            if u'' in phoneme_gt_labels:
                phoneme_gt_onsets_0start, phoneme_gt_labels = removeSilence(phoneme_gt_onsets_0start, phoneme_gt_labels)

            if u'' in phoneme_score_labels:
                boundaries_phoneme_start_time, phoneme_score_labels = removeSilence(boundaries_phoneme_start_time, phoneme_score_labels)

            # print(phoneme_gt_labels)
            # print(phoneme_score_labels)

            results_aggregation_save_helper(syllable_gt_onsets_0start,
                                            syllable_gt_labels,
                                            boundaries_syllable_start_time,
                                            syllable_score_labels,
                                            phoneme_gt_onsets_0start,
                                            phoneme_gt_labels,
                                            boundaries_phoneme_start_time,
                                            phoneme_score_labels,
                                            eval_results_path,
                                            artist_path,
                                            rn,
                                            ii_line,
                                            time_end-time_start)

            if plot:
                figurePlot(mfcc_line,
                           syllable_gt_onsets_0start,
                           phoneme_gt_onsets_0start_without_syllable_onsets,
                           hsmm,
                           phoneme_score_labels_mapped,
                           path,
                           boundaries_phoneme_start_time,
                           boundaries_syllable_start_time,
                           syllable_score_durs,
                           phoneme_score_durs)

if __name__ == '__main__':

    import pickle

    primarySchool_test_recordings, _, _ = getTestTrainRecordingsJoint()

    scaler = pickle.load(open(kerasScaler_path, 'rb'))
    #
    # model_keras_cnn_0 = load_model(full_path_keras_cnn_0 + str(0) + '.h5')

    model_keras_cnn_0 = LRHSMM.kerasModel(kerasModels_path)

    phonemeSegAllRecordings(wav_path=primarySchool_wav_path,
                           textgrid_path=primarySchool_textgrid_path,
                           scaler=scaler,
                           test_recordings=primarySchool_test_recordings,
                           model_keras_cnn_0=model_keras_cnn_0,
                           cnnModel_name=cnnModel_name,
                           eval_results_path=eval_results_path,
                           threshold=0.54,
                           obs_cal='tocal',
                           decoding_method='viterbi',
                            plot=False)

    run_eval_onset('hsmm')
    run_eval_segment('hsmm')
