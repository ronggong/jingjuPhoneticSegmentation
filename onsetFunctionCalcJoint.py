# -*- coding: utf-8 -*-
import matplotlib
matplotlib.use('TkAgg')

import pickle
from os import makedirs
from os.path import exists

import numpy as np
import pyximport
from keras.models import load_model

pyximport.install(reload_support=True,
                  setup_args={'include_dirs': np.get_include()})

from general.filePathJoint import *
from general.parameters import *
from general.textgridParser import textGrid2WordList, wordListsParseByLines
from lyricsRecognizer.audioPreprocessing import featureReshape
from general.trainTestSeparation import getTestTrainRecordingsJoint

# from peakPicking import peakPicking
# from madmom.features.onsets import OnsetPeakPickingProcessor
from lyricsRecognizer.audioPreprocessing import getMFCCBands2DMadmom
import viterbiDecoding
from helperCode import gt_score_preparation_helper
from helperCode import results_aggregation_save_helper
import matplotlib.pyplot as plt
import matplotlib.patches as patches
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


def smoothObs(obs):
    hann = np.hanning(5)
    hann /= np.sum(hann)

    obs = np.convolve(hann, obs, mode='same')
    return obs


def figurePlot(mfcc_line,
               syllable_gt_onsets_0start,
               phoneme_gt_onsets_0start_without_syllable_onsets,
               obs_syllable,
               boundaries_syllable_start_time,
               obs_phoneme,
               boundaries_phoneme_start_time,
               syllable_score_durs,
               phoneme_score_durs):
    # plot Error analysis figures
    plt.figure(figsize=(16, 8))
    # plt.figure(figsize=(8, 4))
    # class weight
    ax1 = plt.subplot(411)
    y = np.arange(0, 80)
    x = np.arange(0, mfcc_line.shape[0]) * hopsize_t
    cax = plt.pcolormesh(x, y, np.transpose(mfcc_line[:, 80 * 7:80 * 8]))
    for gso in syllable_gt_onsets_0start:
        plt.axvline(gso, color='r', linewidth=2)
    for gpo in phoneme_gt_onsets_0start_without_syllable_onsets:
        plt.axvline(gpo, color='k', linewidth=2)
        # for i_gs, gs in enumerate(groundtruth_onset):
        #     plt.axvline(gs, color='r', linewidth=2)
        # plt.text(gs, ax1.get_ylim()[1], groundtruth_syllables[i_gs])

    # cbar = fig.colorbar(cax)
    ax1.set_ylabel('Mel bands', fontsize=12)
    ax1.get_xaxis().set_visible(False)
    ax1.axis('tight')
    # plt.title('Calculating: '+rn+' phrase '+str(i_obs))

    ax2 = plt.subplot(412, sharex=ax1)
    plt.plot(np.arange(0, len(obs_syllable)) * hopsize_t, obs_syllable)
    for bsst in boundaries_syllable_start_time:
        plt.axvline(bsst, color='r', linewidth=2)
        # plt.text(i_boundary[i_ib] * hopsize_t, ax2.get_ylim()[1], syllables[i_line][i_ib])

    ax2.set_ylabel('ODF syllable', fontsize=12)
    ax2.axis('tight')

    ax3 = plt.subplot(413, sharex=ax1)
    plt.plot(np.arange(0, len(obs_phoneme)) * hopsize_t, obs_phoneme)
    for bpst in boundaries_phoneme_start_time:
        plt.axvline(bpst, color='k', linewidth=1)
    for bsst in boundaries_syllable_start_time:
        plt.axvline(bsst, color='r', linewidth=2)
    # for i_ib in range(len(i_boundary)-1):
    #     plt.axvline(i_boundary[i_ib] * hopsize_t, color='r', linewidth=2)
    # plt.text(i_boundary[i_ib] * hopsize_t, ax2.get_ylim()[1], syllables[i_line][i_ib])

    ax3.set_ylabel('ODF phoneme', fontsize=12)
    ax3.axis('tight')

    ax4 = plt.subplot(414, sharex=ax1)
    # print(duration_score)
    time_start = 0
    for ii_ds, ds in enumerate(syllable_score_durs):
        ax4.add_patch(
            patches.Rectangle(
                (time_start, 0),  # (x,y)
                ds,  # width
                3,  # height
                alpha=0.5
            ))
        time_start += ds

    time_start = 0
    for psd in phoneme_score_durs:
        ax4.add_patch(
            patches.Rectangle(
                (time_start, 0.5),  # (x,y)
                psd,  # width
                2,  # height
                color='r',
                alpha=0.5
            ))
        time_start += psd
    ax4.set_ylim((0, 3))
    ax4.set_ylabel('Score duration', fontsize=12)
    plt.xlabel('Time (s)')
    # plt.tight_layout()

    plt.show()


def onsetFunctionAllRecordings(wav_path,
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
                                                                              'detailsSilence')
        gtSyllableLists, gtPhonemeLists = textgridSyllablePhonemeParser(groundtruth_textgrid_file,
                                                                        'dianSilence',
                                                                        'details')

        if obs_cal == 'tocal':
            # load audio
            fs = 44100
            mfcc = getMFCCBands2DMadmom(wav_file, fs, hopsize_t, channel=1)
            mfcc_scaled = scaler.transform(mfcc)
            mfcc_reshaped = featureReshape(mfcc_scaled, nlen=7)


        # print lineList
        for ii_line in range(len(gtSyllableLists)):

            # observation path
            obs_path = join('./obs', cnnModel_name, artist_path)
            obs_syllable_filename = rn + '_syllable_' + str(ii_line + 1) + '.pkl'
            obs_phoneme_filename = rn + '_phoneme_' + str(ii_line + 1) + '.pkl'

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
            phoneme_score_durs_grouped_by_syllables = []
            phoneme_score_labels_grouped_by_syllables = []
            phoneme_score_durs_syllable = []
            phoneme_score_labels_syllable = []
            for pls in phoneme_list_score:
                # where meeting next syllable or is the last phoneme

                if pls[0] in syllable_score_onsets[1:]:
                    phoneme_score_durs_grouped_by_syllables.append(phoneme_score_durs_syllable)
                    phoneme_score_labels_grouped_by_syllables.append(phoneme_score_labels_syllable)
                    phoneme_score_durs_syllable = []
                    phoneme_score_labels_syllable = []

                phoneme_score_durs_syllable.append(pls[1] - pls[0])
                phoneme_score_labels_syllable.append(pls[2])

                if pls == phoneme_list_score[-1]:
                    phoneme_score_durs_grouped_by_syllables.append(phoneme_score_durs_syllable)
                    phoneme_score_labels_grouped_by_syllables.append(phoneme_score_labels_syllable)

            # onsets start from time 0
            syllable_gt_onsets_0start = np.array(syllable_gt_onsets) - syllable_gt_onsets[0]
            phoneme_gt_onsets_0start = np.array(phoneme_gt_onsets) - phoneme_gt_onsets[0]
            phoneme_gt_onsets_0start_without_syllable_onsets = \
                np.setdiff1d(phoneme_gt_onsets_0start, syllable_gt_onsets_0start)

            if not set(syllable_gt_onsets).issubset(set(phoneme_gt_onsets)):
                raise
            if not set(syllable_score_onsets).issubset(set(phoneme_score_onsets)):
                raise

            frame_start     = int(round(time_start / hopsize_t))
            frame_end       = int(round(time_end / hopsize_t))

            syllable_score_durs = np.array(syllable_score_durs)
            print(np.sum(syllable_score_durs))
            syllable_score_durs *= (time_end - time_start) / np.sum(syllable_score_durs)

            if obs_cal == 'tocal':
                mfcc_line          = mfcc[frame_start:frame_end]
                mfcc_reshaped_line = mfcc_reshaped[frame_start:frame_end]
                mfcc_reshaped_line = np.expand_dims(mfcc_reshaped_line, axis=1)

                obs_syllable, obs_phoneme = model_keras_cnn_0.predict(mfcc_reshaped_line, batch_size=128, verbose=2)

                # save onset curve
                print('save onset curve ... ...')
                if not exists(obs_path):
                    makedirs(obs_path)
                pickle.dump(obs_syllable, open(join(obs_path, obs_syllable_filename), 'w'))
                pickle.dump(obs_phoneme, open(join(obs_path, obs_phoneme_filename), 'w'))

            else:
                obs_syllable = pickle.load(open(join(obs_path, obs_syllable_filename), 'r'))
                obs_phoneme = pickle.load(open(join(obs_path, obs_phoneme_filename), 'r'))

            obs_syllable = np.squeeze(obs_syllable)
            obs_phoneme = np.squeeze(obs_phoneme)

            obs_syllable = smoothObs(obs_syllable)
            obs_phoneme = smoothObs(obs_phoneme)

            # decoding syllable boundaries
            obs_syllable[0] = 1.0
            obs_syllable[-1] = 1.0
            boundaries_syllable = viterbiDecoding.viterbiSegmental2(obs_syllable, syllable_score_durs, varin)

            # syllable boundaries
            boundaries_syllable_start_time = np.array(boundaries_syllable[:-1])*hopsize_t
            boundaries_syllable_end_time   = np.array(boundaries_syllable[1:])*hopsize_t

            # decoding phoneme boundaries
            boundaries_phoneme_start_time = np.array([])
            boundaries_phoneme_end_time = np.array([])
            phoneme_score_durs = np.array([])

            for ii_syl_boundary in range(len(boundaries_syllable)-1):
                frame_start_syl = boundaries_syllable[ii_syl_boundary]
                frame_end_syl = boundaries_syllable[ii_syl_boundary+1]
                obs_phoneme_syl = obs_phoneme[frame_start_syl: frame_end_syl]

                phoneme_score_durs_syl = np.array(phoneme_score_durs_grouped_by_syllables[ii_syl_boundary])
                phoneme_score_durs = np.concatenate((phoneme_score_durs, phoneme_score_durs_syl))

                if len(phoneme_score_durs_syl) < 2:
                    continue

                phoneme_score_durs_syl *= \
                    (boundaries_syllable_end_time[ii_syl_boundary] - boundaries_syllable_start_time[ii_syl_boundary]) \
                    / np.sum(phoneme_score_durs_syl)

                obs_phoneme_syl[0] = 1.0
                obs_phoneme_syl[-1] = 1.0
                boundaries_phoneme_syl = viterbiDecoding.viterbiSegmental2(obs_phoneme_syl, phoneme_score_durs_syl, varin)

                # phoneme boundaries
                boundaries_phoneme_syl_start_time = (np.array(boundaries_phoneme_syl[:-1]) + frame_start_syl) * hopsize_t
                boundaries_phoneme_syl_end_time = (np.array(boundaries_phoneme_syl[1:]) + frame_start_syl) * hopsize_t

                boundaries_phoneme_start_time = np.concatenate((boundaries_phoneme_start_time, boundaries_phoneme_syl_start_time))
                boundaries_phoneme_end_time = np.concatenate((boundaries_phoneme_end_time, boundaries_phoneme_syl_end_time))

            # print(np.sum(phoneme_score_durs))
            phoneme_score_durs *= (time_end-time_start)/np.sum(phoneme_score_durs)

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
                           obs_syllable,
                           boundaries_syllable_start_time,
                           obs_phoneme,
                           boundaries_phoneme_start_time,
                           syllable_score_durs,
                           phoneme_score_durs)
    # return eval_results_decoding_path


if __name__ == '__main__':

    # from eval_demo import eval_write_2_txt

    primarySchool_test_recordings, _, _ = getTestTrainRecordingsJoint()

    scaler = pickle.load(open(scaler_joint_model_path, 'rb'))

    model_keras_cnn_0 = load_model(full_path_keras_cnn_0 + str(0) + '.h5')

    onsetFunctionAllRecordings(wav_path=primarySchool_wav_path,
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
    run_eval_onset('joint')
    run_eval_segment('joint')