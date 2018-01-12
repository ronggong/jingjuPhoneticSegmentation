from filePathHsmm import eval_results_path as eval_results_path_hsmm
from filePathJoint import eval_results_path as eval_results_path_joint
from filePathHsmm import cnn_file_name
from filePathJoint import cnnModel_name
from parameters import *
from general.trainTestSeparation import getTestTrainRecordingsJoint
from general.trainTestSeparation import getRecordings
from evaluation import onsetEval
from evaluation import segmentEval
from evaluation import metrics
import os
import pickle
import numpy as np


def writeResults2Txt(filename,
                     decoding_method,
                     results):
    """

    :param filename:
    :param eval_label_str:
    :param decoding_method:
    :param precision_nolabel_25:
    :param recall_nolabel_25:
    :param F1_nolabel_25:
    :param precision_25:
    :param recall_25:
    :param F1_25:
    :param precision_nolabel_5:
    :param recall_nolabel_5:
    :param F1_nolabel_5:
    :param precision_5:
    :param recall_5:
    :param F1_5:
    :return:
    """

    with open(filename, 'w') as f:
        f.write(decoding_method)
        f.write('\n')

        # no label 0.025
        f.write(str(results[0]))
        f.write('\n')
        f.write(str(results[1]))
        f.write('\n')
        f.write(str(results[2]))
        f.write('\n')

        # no label 0.05
        f.write(str(results[3]))
        f.write('\n')
        f.write(str(results[4]))
        f.write('\n')
        f.write(str(results[5]))
        f.write('\n')

        # label 0.025
        f.write(str(results[6]))
        f.write('\n')
        f.write(str(results[7]))
        f.write('\n')
        f.write(str(results[8]))
        f.write('\n')

        # label 0.05
        f.write(str(results[9]))
        f.write('\n')
        f.write(str(results[10]))
        f.write('\n')
        f.write(str(results[11]))

def batch_run_metrics_calculation(sumStat, gt_onsets, detected_onsets):
    """
    Batch run the metric calculation
    :param sumStat:
    :param gt_onsets:
    :param detected_onsets:
    :return:
    """
    counter = 0
    for l in [False, True]:
        for t in [0.025, 0.05]:
            numDetectedOnsets, numGroundtruthOnsets, \
            numOnsetCorrect, _, _ = onsetEval(gt_onsets, detected_onsets, t, l)

            sumStat[counter, 0] += numDetectedOnsets
            sumStat[counter, 1] += numGroundtruthOnsets
            sumStat[counter, 2] += numOnsetCorrect

            counter += 1

def metrics_aggregation(sumStat):

    recall_nolabel_25, precision_nolabel_25, F1_nolabel_25 = metrics(sumStat[0, 0], sumStat[0, 1], sumStat[0, 2])
    recall_nolabel_5, precision_nolabel_5, F1_nolabel_5 = metrics(sumStat[1, 0], sumStat[1, 1], sumStat[1, 2])
    recall_label_25, precision_label_25, F1_label_25 = metrics(sumStat[2, 0], sumStat[2, 1], sumStat[2, 2])
    recall_label_5, precision_label_5, F1_label_5 = metrics(sumStat[3, 0], sumStat[3, 1], sumStat[3, 2])

    return  precision_nolabel_25, recall_nolabel_25, F1_nolabel_25, \
            precision_nolabel_5, recall_nolabel_5, F1_nolabel_5, \
            precision_label_25, recall_label_25, F1_label_25, \
            precision_label_5, recall_label_5, F1_label_5

def run_eval_onset(method='hsmm'):
    """
    run evaluation for onset detection
    :param method  hsmm or joint:
    :return:
    """
    if method == 'hsmm':
        eval_results_path = eval_results_path_hsmm
        eval_filename = cnn_file_name
    else:
        eval_results_path = eval_results_path_joint
        eval_filename = cnnModel_name

    primarySchool_test_recordings, _, _ = getTestTrainRecordingsJoint()

    sumStat_syllable = np.zeros((4, 3), dtype='int')
    sumStat_phoneme = np.zeros((4, 3), dtype='int')

    for artist, rn in primarySchool_test_recordings:
        results_path = os.path.join(eval_results_path, artist)
        result_files = getRecordings(results_path)

        for rf in result_files:
            result_filename = os.path.join(results_path, rf+'.pkl')
            syllable_gt_onsets, syllable_detected_onsets, \
            phoneme_gt_onsets, phoneme_detected_onsets, _ \
                                        = pickle.load(open(result_filename, 'r'))

            batch_run_metrics_calculation(sumStat_syllable, syllable_gt_onsets, syllable_detected_onsets)
            batch_run_metrics_calculation(sumStat_phoneme, phoneme_gt_onsets, phoneme_detected_onsets)

    result_syllable = metrics_aggregation(sumStat_syllable)
    result_phoneme = metrics_aggregation(sumStat_phoneme)

    current_path = os.path.dirname(os.path.abspath(__file__))

    writeResults2Txt(os.path.join(current_path, '../'+method, eval_filename+'_syllable_onset'+'.txt'),
                     method,
                     result_syllable)

    writeResults2Txt(os.path.join(current_path, '../'+method, eval_filename+'_phoneme_onset'+'.txt'),
                     method,
                     result_phoneme)


def segmentEvalHelper(onsets, line_time):
    onsets_frame = np.round(np.array([sgo[0] for sgo in onsets]) / hopsize_t)

    resample = [onsets[0][1]]

    current = onsets[0][1]

    for ii_sample in range(1, int(round(line_time / hopsize_t))):

        if ii_sample in onsets_frame:
            idx_onset = np.where(onsets_frame == ii_sample)
            idx_onset = idx_onset[0][0]
            current = onsets[idx_onset][1]
        resample.append(current)

    return resample


def run_eval_segment(method='hsmm'):
    if method == 'hsmm':
        eval_results_path = eval_results_path_hsmm
        eval_filename = cnn_file_name
    else:
        eval_results_path = eval_results_path_joint
        eval_filename = cnnModel_name

    primarySchool_test_recordings, _, _ = getTestTrainRecordingsJoint()

    sumSampleCorrect_syllable, sumSampleCorrect_phoneme, \
    sumSample_syllable, sumSample_phoneme = 0,0,0,0
    for artist, rn in primarySchool_test_recordings:
        results_path = os.path.join(eval_results_path, artist)
        result_files = getRecordings(results_path)

        for rf in result_files:
            result_filename = os.path.join(results_path, rf+'.pkl')
            syllable_gt_onsets, syllable_detected_onsets, \
            phoneme_gt_onsets, phoneme_detected_onsets, line_time \
                                        = pickle.load(open(result_filename, 'r'))

            syllable_gt_onsets_resample = segmentEvalHelper(syllable_gt_onsets, line_time)
            syllable_detected_onsets_resample = segmentEvalHelper(syllable_detected_onsets, line_time)
            phoneme_gt_onsets_resample = segmentEvalHelper(phoneme_gt_onsets, line_time)
            phoneme_detected_onsets_resample = segmentEvalHelper(phoneme_detected_onsets, line_time)

            # print(phoneme_gt_onsets)
            # print(phoneme_gt_onsets_resample)
            # print(phoneme_detected_onsets)
            # print(phoneme_detected_onsets_resample)

            sample_correct_syllable, sample_syllable = segmentEval(syllable_gt_onsets_resample, syllable_detected_onsets_resample)
            sample_correct_phoneme, sample_phoneme = segmentEval(phoneme_gt_onsets_resample, phoneme_detected_onsets_resample)

            sumSampleCorrect_syllable += sample_correct_syllable
            sumSampleCorrect_phoneme += sample_correct_phoneme
            sumSample_syllable += sample_syllable
            sumSample_phoneme += sample_phoneme

    acc_syllable = sumSampleCorrect_syllable/float(sumSample_syllable)
    acc_phoneme = sumSampleCorrect_phoneme/float(sumSample_phoneme)

    current_path = os.path.dirname(os.path.abspath(__file__))

    with open(os.path.join(current_path, '../'+method, eval_filename+'_syllable_segment'+'.txt'), 'w') as f:
        f.write(method)
        f.write('\n')
        f.write(str(acc_syllable))

    with open(os.path.join(current_path, '../'+method, eval_filename+'_phoneme_segment'+'.txt'), 'w') as f:
        f.write(method)
        f.write('\n')
        f.write(str(acc_phoneme))

if __name__ == '__main__':
    run_eval_onset('hsmm')
    run_eval_onset('joint')
    run_eval_segment('hsmm')
    run_eval_segment('joint')