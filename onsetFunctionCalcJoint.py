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
import matplotlib.pyplot as plt
import matplotlib.patches as patches


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

            ###--- groundtruth, score preparation
            lineList_gt_syllable = gtSyllableLists[ii_line]
            lineList_score_syllable = scoreSyllableLists[ii_line]
            lineList_gt_phoneme = gtPhonemeLists[ii_line]
            lineList_score_phoneme = scorePhonemeLists[ii_line]

            time_start = lineList_gt_syllable[0][0]
            time_end = lineList_gt_syllable[0][1]

            # list has syllable and phoneme information
            syllable_list_gt = lineList_gt_syllable[1]
            phoneme_list_gt = lineList_gt_phoneme[1]

            syllable_list_score = lineList_score_syllable[1]
            phoneme_list_score = lineList_score_phoneme[1]

            # list only has onsets
            syllable_gt_onsets = [s[0] for s in syllable_list_gt]
            phoneme_gt_onsets = [p[0] for p in phoneme_list_gt]

            syllable_score_onsets = [s[0] for s in syllable_list_score]
            phoneme_score_onsets = [p[0] for p in phoneme_list_score]

            # syllable score durations and labels
            syllable_score_durs = [sls[1] - sls[0] for sls in syllable_list_score]
            syllable_score_labels = [sls[2] for sls in syllable_list_score]

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

            eval_results_decoding_path = eval_results_path

            filename_syll_lab = join(eval_results_decoding_path, artist_path, rn + '_' + str(ii_line + 1) + '.syll.lab')
            label = True

            # syllable boundaries
            boundaries_syllable_start_time = np.array(boundaries_syllable[:-1])*hopsize_t
            boundaries_syllable_end_time   = np.array(boundaries_syllable[1:])*hopsize_t

            # decoding phoneme boundaries
            filename_phn_lab = join(eval_results_decoding_path, artist_path,
                                     rn + '_' + str(ii_line + 1) + '.phn.lab')
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

            print(np.sum(phoneme_score_durs))
            phoneme_score_durs *= (time_end-time_start)/np.sum(phoneme_score_durs)

            print(boundaries_phoneme_start_time)
    #         # uncomment this section if we want to write boundaries to .syll.lab file
    #
    #         eval_results_data_path = dirname(filename_syll_lab)
    #
    #         print(eval_results_data_path)
    #
    #         if not exists(eval_results_data_path):
    #             makedirs(eval_results_data_path)
    #
    #         # write boundary lab file
    #         if not lab:
    #             if decoding_method == 'viterbi':
    #                 boundary_list = zip(time_boundray_start.tolist(), time_boundray_end.tolist(), filter(None,pinyins[i_line]))
    #             else:
    #                 boundary_list = zip(time_boundray_start.tolist(), time_boundray_end.tolist())
    #
    #         else:
    #             if decoding_method == 'viterbi':
    #                 boundary_list = zip(time_boundray_start.tolist(), time_boundray_end.tolist(), syllables[i_line])
    #                 label = True
    #
    #             else:
    #                 boundary_list = zip(time_boundray_start.tolist(), time_boundray_end.tolist())
    #                 label = False
    #
    #         boundaryLabWriter(boundaryList=boundary_list,
    #                           outputFilename=filename_syll_lab,
    #                             label=label)
    #
    #         print(i_boundary)
    #         print(len(obs_i))
    #         # print(np.array(groundtruth_syllable)*fs/hopsize)
    #
            if plot:

                # plot Error analysis figures
                plt.figure(figsize=(16, 8))
                # plt.figure(figsize=(8, 4))
                # class weight
                ax1 = plt.subplot(411)
                y = np.arange(0, 80)
                x = np.arange(0, mfcc_line.shape[0])*hopsize_t
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
                plt.plot(np.arange(0,len(obs_syllable))*hopsize_t, obs_syllable)
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
                ax4.set_ylim((0,3))
                ax4.set_ylabel('Score duration', fontsize=12)
                plt.xlabel('Time (s)')
                # plt.tight_layout()

                plt.show()
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
                               plot=True)

    # scaler = pickle.load(open(full_path_mfccBands_2D_scaler_onset, 'rb'))
    #
    # def peakPickingSubroutine(th, obs_cal):
    #     from src.utilFunctions import append_or_write
    #     import csv
    #
    #     eval_result_file_name = join(jingju_results_path,
    #                                  varin['sample_weighting'],
    #                                  cnnModel_name + '_peakPicking_threshold_results.txt')
    #
    #     list_recall_onset_25, list_precision_onset_25, list_F1_onset_25 = [], [], []
    #     list_recall_onset_5, list_precision_onset_5, list_F1_onset_5 = [], [], []
    #     list_recall_25, list_precision_25, list_F1_25 = [], [], []
    #     list_recall_5, list_precision_5, list_F1_5 = [], [], []
    #
    #     for ii in range(5):
    #
    #         if obs_cal:
    #             model_keras_cnn_0 = load_model(full_path_keras_cnn_0 + str(ii) + '.h5')
    #         else:
    #             model_keras_cnn_0 = None
    #
    #         if varin['dataset'] != 'ismir':
    #             # nacta2017
    #             onsetFunctionAllRecordings(wav_path=nacta2017_wav_path,
    #                                        textgrid_path=nacta2017_textgrid_path,
    #                                        score_path=nacta2017_score_pinyin_path,
    #                                        test_recordings=testNacta2017,
    #                                        model_keras_cnn_0=model_keras_cnn_0,
    #                                        cnnModel_name=cnnModel_name + str(ii),
    #                                        eval_results_path=eval_results_path + str(ii),
    #                                        scaler=scaler,
    #                                        feature_type='madmom',
    #                                        dmfcc=False,
    #                                        nbf=True,
    #                                        mth=mth_ODF,
    #                                        late_fusion=fusion,
    #                                        threshold=th,
    #                                        obs_cal=obs_cal,
    #                                        decoding_method='peakPicking')
    #
    #         eval_results_decoding_path = onsetFunctionAllRecordings(wav_path=nacta_wav_path,
    #                                                                 textgrid_path=nacta_textgrid_path,
    #                                                                 score_path=nacta_score_pinyin_path,
    #                                                                 test_recordings=testNacta,
    #                                                                 model_keras_cnn_0=model_keras_cnn_0,
    #                                                                 cnnModel_name=cnnModel_name + str(ii),
    #                                                                 eval_results_path=eval_results_path + str(ii),
    #                                                                 scaler=scaler,
    #                                                                 feature_type='madmom',
    #                                                                 dmfcc=False,
    #                                                                 nbf=True,
    #                                                                 mth=mth_ODF,
    #                                                                 late_fusion=fusion,
    #                                                                 threshold=th,
    #                                                                 obs_cal=obs_cal,
    #                                                                 decoding_method='peakPicking')
    #
    #         append_write = append_or_write(eval_result_file_name)
    #         with open(eval_result_file_name, append_write) as testfile:
    #             csv_writer = csv.writer(testfile)
    #             csv_writer.writerow([th])
    #
    #         # eval_results_decoding_path = cnnModel_name + str(ii) + '_peakPickingMadmom'
    #         precision_onset, recall_onset, F1_onset, \
    #         precision, recall, F1, \
    #             = eval_write_2_txt(eval_result_file_name,
    #                                eval_results_decoding_path,
    #                                label=False,
    #                                decoding_method='peakPicking')
    #
    #         list_precision_onset_25.append(precision_onset[0])
    #         list_precision_onset_5.append(precision_onset[1])
    #         list_recall_onset_25.append(recall_onset[0])
    #         list_recall_onset_5.append(recall_onset[1])
    #         list_F1_onset_25.append(F1_onset[0])
    #         list_F1_onset_5.append(F1_onset[1])
    #         list_precision_25.append(precision[0])
    #         list_precision_5.append(precision[1])
    #         list_recall_25.append(recall[0])
    #         list_recall_5.append(recall[1])
    #         list_F1_25.append(F1[0])
    #         list_F1_5.append(F1[1])
    #
    #     return list_precision_onset_25, list_recall_onset_25, list_F1_onset_25, list_precision_25, list_recall_25, list_F1_25, \
    #            list_precision_onset_5, list_recall_onset_5, list_F1_onset_5, list_precision_5, list_recall_5, list_F1_5
    #
    #
    # def viterbiSubroutine(eval_label, obs_cal):
    #
    #     list_recall_onset_25, list_precision_onset_25, list_F1_onset_25 = [], [], []
    #     list_recall_onset_5, list_precision_onset_5, list_F1_onset_5 = [], [], []
    #     list_recall_25, list_precision_25, list_F1_25 = [], [], []
    #     list_recall_5, list_precision_5, list_F1_5 = [], [], []
    #     for ii in range(5):
    #
    #         if obs_cal == 'tocal':
    #
    #             model_keras_cnn_0 = load_model(full_path_keras_cnn_0 + str(ii) + '.h5')
    #
    #             print(full_path_keras_cnn_0)
    #
    #             if varin['dataset'] != 'ismir':
    #                 # nacta2017
    #                 onsetFunctionAllRecordings(wav_path=nacta2017_wav_path,
    #                                            textgrid_path=nacta2017_textgrid_path,
    #                                            score_path=nacta2017_score_unified_path,
    #                                            test_recordings=testNacta2017,
    #                                            model_keras_cnn_0=model_keras_cnn_0,
    #                                            cnnModel_name=cnnModel_name + str(ii),
    #                                            eval_results_path=eval_results_path + str(ii),
    #                                            scaler=scaler,
    #                                            feature_type='madmom',
    #                                            dmfcc=False,
    #                                            nbf=True,
    #                                            mth=mth_ODF,
    #                                            late_fusion=fusion,
    #                                            obs_cal=obs_cal,
    #                                            decoding_method='viterbi')
    #
    #             # nacta
    #             eval_results_decoding_path = onsetFunctionAllRecordings(wav_path=nacta_wav_path,
    #                                                                     textgrid_path=nacta_textgrid_path,
    #                                                                     score_path=nacta_score_unified_path,
    #                                                                     test_recordings=testNacta,
    #                                                                     model_keras_cnn_0=model_keras_cnn_0,
    #                                                                     cnnModel_name=cnnModel_name + str(ii),
    #                                                                     eval_results_path=eval_results_path + str(ii),
    #                                                                     scaler=scaler,
    #                                                                     feature_type='madmom',
    #                                                                     dmfcc=False,
    #                                                                     nbf=True,
    #                                                                     mth=mth_ODF,
    #                                                                     late_fusion=fusion,
    #                                                                     obs_cal=obs_cal,
    #                                                                     decoding_method='viterbi')
    #         else:
    #             eval_results_decoding_path = eval_results_path + str(ii)
    #
    #         precision_onset, recall_onset, F1_onset, \
    #         precision, recall, F1, \
    #             = eval_write_2_txt(eval_result_file_name=join(eval_results_decoding_path, 'results.csv'),
    #                                segSyllable_path=eval_results_decoding_path,
    #                                label=eval_label,
    #                                decoding_method='viterbi')
    #
    #         list_precision_onset_25.append(precision_onset[0])
    #         list_precision_onset_5.append(precision_onset[1])
    #         list_recall_onset_25.append(recall_onset[0])
    #         list_recall_onset_5.append(recall_onset[1])
    #         list_F1_onset_25.append(F1_onset[0])
    #         list_F1_onset_5.append(F1_onset[1])
    #         list_precision_25.append(precision[0])
    #         list_precision_5.append(precision[1])
    #         list_recall_25.append(recall[0])
    #         list_recall_5.append(recall[1])
    #         list_F1_25.append(F1[0])
    #         list_F1_5.append(F1[1])
    #
    #     return list_precision_onset_25, list_recall_onset_25, list_F1_onset_25, list_precision_25, list_recall_25, list_F1_25, \
    #            list_precision_onset_5, list_recall_onset_5, list_F1_onset_5, list_precision_5, list_recall_5, list_F1_5
    #
    #
    # def writeResults2Txt(filename,
    #                      eval_label_str,
    #                      decoding_method,
    #                     list_precision_onset_25,
    #                      list_recall_onset_25,
    #                      list_F1_onset_25,
    #                      list_precision_25,
    #                      list_recall_25,
    #                      list_F1_25,
    #                     list_precision_onset_5,
    #                      list_recall_onset_5,
    #                      list_F1_onset_5,
    #                      list_precision_5,
    #                      list_recall_5,
    #                      list_F1_5):
    #     """
    #     :param filename:
    #     :param eval_label_str: eval label or not
    #     :param decoding_method: viterbi or peakPicking
    #     :param list_precision_onset_25:
    #     :param list_recall_onset_25:
    #     :param list_F1_onset_25:
    #     :param list_precision_25:
    #     :param list_recall_25:
    #     :param list_F1_25:
    #     :param list_precision_onset_5:
    #     :param list_recall_onset_5:
    #     :param list_F1_onset_5:
    #     :param list_precision_5:
    #     :param list_recall_5:
    #     :param list_F1_5:
    #     :return:
    #     """
    #
    #     with open(filename, 'w') as f:
    #         f.write(decoding_method)
    #         f.write('\n')
    #         f.write(eval_label_str)
    #         f.write('\n')
    #         f.write(str(np.mean(list_precision_onset_25))+' '+str(np.std(list_precision_onset_25)))
    #         f.write('\n')
    #         f.write(str(np.mean(list_recall_onset_25))+' '+str(np.std(list_recall_onset_25)))
    #         f.write('\n')
    #         f.write(str(np.mean(list_F1_onset_25))+' '+str(np.std(list_F1_onset_25)))
    #         f.write('\n')
    #
    #         f.write(str(np.mean(list_precision_25))+' '+str(np.std(list_precision_25)))
    #         f.write('\n')
    #         f.write(str(np.mean(list_recall_25))+' '+str(np.std(list_recall_25)))
    #         f.write('\n')
    #         f.write(str(np.mean(list_F1_25))+' '+str(np.std(list_F1_25)))
    #         f.write('\n')
    #
    #         f.write(str(np.mean(list_precision_onset_5)) + ' ' + str(np.std(list_precision_onset_5)))
    #         f.write('\n')
    #         f.write(str(np.mean(list_recall_onset_5)) + ' ' + str(np.std(list_recall_onset_5)))
    #         f.write('\n')
    #         f.write(str(np.mean(list_F1_onset_5)) + ' ' + str(np.std(list_F1_onset_5)))
    #         f.write('\n')
    #
    #         f.write(str(np.mean(list_precision_5)) + ' ' + str(np.std(list_precision_5)))
    #         f.write('\n')
    #         f.write(str(np.mean(list_recall_5)) + ' ' + str(np.std(list_recall_5)))
    #         f.write('\n')
    #         f.write(str(np.mean(list_F1_5)) + ' ' + str(np.std(list_F1_5)))
    #
    #
    # def viterbiLabelEval(eval_label, obs_cal):
    #
    #     list_precision_onset_25, list_recall_onset_25, list_F1_onset_25, list_precision_25, list_recall_25, list_F1_25, \
    #     list_precision_onset_5, list_recall_onset_5, list_F1_onset_5, list_precision_5, list_recall_5, list_F1_5 = \
    #         viterbiSubroutine(eval_label, obs_cal)
    #
    #     postfix_statistic_sig = 'label' if eval_label else 'nolabel'
    #     pickle.dump(list_F1_onset_25,
    #                 open(join('./statisticalSignificance/data/jingju', varin['sample_weighting'],
    #                           cnnModel_name + '_' + 'viterbi' + '_' + postfix_statistic_sig + '.pkl'), 'w'))
    #
    #     writeResults2Txt(join(jingju_results_path, varin['sample_weighting'], cnnModel_name + '_viterbi' + '_' + postfix_statistic_sig + '.txt'),
    #                      postfix_statistic_sig,
    #                      'viterbi',
    #                      list_precision_onset_25,
    #                      list_recall_onset_25,
    #                      list_F1_onset_25,
    #                      list_precision_25,
    #                      list_recall_25,
    #                      list_F1_25,
    #                      list_precision_onset_5,
    #                      list_recall_onset_5,
    #                      list_F1_onset_5,
    #                      list_precision_5,
    #                      list_recall_5,
    #                      list_F1_5)
    #
    # ##-- viterbi evaluation
    #
    # obs_cal = 'tocal'
    #
    # viterbiLabelEval(eval_label=True, obs_cal=obs_cal)
    #
    # obs_cal = 'toload'
    #
    # viterbiLabelEval(eval_label=False, obs_cal=obs_cal)
    #
    # ##-- peak picking evaluation
    # # scan the best threshold
    # best_F1_onset_25, best_th = 0, 0
    # for th in range(1, 9):
    #     th *= 0.1
    #
    #     _,_,list_F1_onset_25,_,_,_,_,_,_,_,_,_= peakPickingSubroutine(th, obs_cal)
    #
    #     if np.mean(list_F1_onset_25) > best_F1_onset_25:
    #         best_th = th
    #         best_F1_onset_25 = np.mean(list_F1_onset_25)
    #
    # # finer scan the best threshold
    # for th in range(int((best_th - 0.1) * 100), int((best_th + 0.1) * 100)):
    #     th *= 0.01
    #
    #     _,_,list_F1_onset_25,_,_,_,_,_,_,_,_,_= peakPickingSubroutine(th, obs_cal)
    #
    #     if np.mean(list_F1_onset_25) > best_F1_onset_25:
    #         best_th = th
    #         best_F1_onset_25 = np.mean(list_F1_onset_25)
    #
    # # get the statistics of the best th
    # list_precision_onset_25, list_recall_onset_25, list_F1_onset_25, list_precision_25, list_recall_25, list_F1_25, \
    # list_precision_onset_5, list_recall_onset_5, list_F1_onset_5, list_precision_5, list_recall_5, list_F1_5 = \
    #     peakPickingSubroutine(best_th, obs_cal)
    #
    # print('best_th', best_th)
    #
    # pickle.dump(list_F1_onset_25,
    #             open(join('./statisticalSignificance/data/jingju', varin['sample_weighting'], cnnModel_name + '_peakPickingMadmom.pkl'), 'w'))
    #
    # writeResults2Txt(join(jingju_results_path, varin['sample_weighting'], cnnModel_name + '_peakPickingMadmom' + '.txt'),
    #                  str(best_th),
    #                  'peakPicking',
    #                  list_precision_onset_25,
    #                  list_recall_onset_25,
    #                  list_F1_onset_25,
    #                  list_precision_25,
    #                  list_recall_25,
    #                  list_F1_25,
    #                  list_precision_onset_5,
    #                  list_recall_onset_5,
    #                  list_F1_onset_5,
    #                  list_precision_5,
    #                  list_recall_5,
    #                  list_F1_5)
    #
