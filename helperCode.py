import os
import pickle
from general.parameters import *


def gt_score_preparation_helper(gtSyllableLists,
                                scoreSyllableLists,
                                gtPhonemeLists,
                                scorePhonemeLists,
                                ii_line):
    """
    Prepare the onset times labels for syllable or phoneme ground truth or score
    :param gtSyllableLists:
    :param scoreSyllableLists:
    :param gtPhonemeLists:
    :param scorePhonemeLists:
    :param ii_line:
    :return:
    """

    ###--- groundtruth, score preparation
    lineList_gt_syllable = gtSyllableLists[ii_line]
    lineList_score_syllable = scoreSyllableLists[ii_line]
    lineList_gt_phoneme = gtPhonemeLists[ii_line]
    lineList_score_phoneme = scorePhonemeLists[ii_line]

    time_start = lineList_gt_syllable[0][0]
    time_end = lineList_gt_syllable[0][1]

    frame_start = int(round(time_start / hopsize_t))
    frame_end = int(round(time_end / hopsize_t))

    # list has syllable and phoneme information
    syllable_list_gt = lineList_gt_syllable[1]
    phoneme_list_gt = lineList_gt_phoneme[1]

    syllable_list_score = lineList_score_syllable[1]
    phoneme_list_score = lineList_score_phoneme[1]

    # list only has onsets
    syllable_gt_onsets = [s[0] for s in syllable_list_gt]
    syllable_gt_labels = [s[2] for s in syllable_list_gt]

    phoneme_gt_onsets = [p[0] for p in phoneme_list_gt]
    phoneme_gt_labels = [p[2] for p in phoneme_list_gt]

    syllable_score_onsets = [s[0] for s in syllable_list_score]
    phoneme_score_onsets = [p[0] for p in phoneme_list_score]
    phoneme_score_labels = [p[2] for p in phoneme_list_score]

    # syllable score durations and labels
    syllable_score_durs = [sls[1] - sls[0] for sls in syllable_list_score]
    syllable_score_labels = [sls[2] for sls in syllable_list_score]

    return frame_start, frame_end, \
           time_start, time_end, \
           syllable_gt_onsets, syllable_gt_labels, \
           phoneme_gt_onsets, phoneme_gt_labels, \
           syllable_score_onsets, syllable_score_labels, \
           phoneme_score_onsets, phoneme_score_labels, \
           syllable_score_durs, phoneme_list_score


def results_aggregation_save_helper(syllable_gt_onsets_0start,
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
                                    line_time):
    """
    Aggregate the ground truth and detected results into a list, and dump
    :param syllable_gt_onsets_0start:
    :param syllable_gt_labels:
    :param boundaries_syllable_start_time:
    :param syllable_score_labels:
    :param phoneme_gt_onsets_0start:
    :param phoneme_gt_labels:
    :param boundaries_phoneme_start_time:
    :param phoneme_score_labels:
    :param eval_results_path:
    :param artist_path:
    :param rn:
    :param ii_line:
    :param line_time:
    :return:
    """
    # aggregate the results
    syllable_gt_onsets_to_save = [[syllable_gt_onsets_0start[ii_sgo], syllable_gt_labels[ii_sgo]]
                                  for ii_sgo in range(len(syllable_gt_onsets_0start))]
    syllable_detected_onsets_to_save = [[boundaries_syllable_start_time[ii_bsst], syllable_score_labels[ii_bsst]]
                                        for ii_bsst in range(len(boundaries_syllable_start_time))]

    phoneme_gt_onsets_to_save = [[phoneme_gt_onsets_0start[ii_pgo], phoneme_gt_labels[ii_pgo]]
                                 for ii_pgo in range(len(phoneme_gt_onsets_0start))]
    phoneme_detected_onsets_to_save = [[boundaries_phoneme_start_time[ii_bpst], phoneme_score_labels[ii_bpst]]
                                       for ii_bpst in range(len(boundaries_phoneme_start_time))]

    gt_detected_to_save = [syllable_gt_onsets_to_save, syllable_detected_onsets_to_save,
                           phoneme_gt_onsets_to_save, phoneme_detected_onsets_to_save, line_time]

    # save to pickle
    path_gt_detected_to_save = os.path.join(eval_results_path, artist_path)
    filename_gt_detected_to_save = rn + '_' + str(ii_line) + '.pkl'

    if not os.path.exists(path_gt_detected_to_save):
        os.makedirs(path_gt_detected_to_save)

    pickle.dump(gt_detected_to_save, open(os.path.join(path_gt_detected_to_save, filename_gt_detected_to_save), 'w'))

