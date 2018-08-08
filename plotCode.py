# -*- coding: utf-8 -*-
"""some plot function mainly for debugging,
used in proposed_method_pipeline.py"""
import matplotlib
matplotlib.use('TkAgg')

import matplotlib.style
import matplotlib as mpl
mpl.style.use('classic')

from matplotlib import gridspec
from matplotlib import cm

from parameters import *
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches


def figure_plot_joint(mfcc_line,
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
    # class weight
    ax1 = plt.subplot(411)
    y = np.arange(0, 80)
    x = np.arange(0, mfcc_line.shape[0]) * hopsize_t
    plt.pcolormesh(x, y, np.transpose(mfcc_line[:, 80 * 7:80 * 8]))
    for gso in syllable_gt_onsets_0start:
        plt.axvline(gso, color='r', linewidth=2)
    for gpo in phoneme_gt_onsets_0start_without_syllable_onsets:
        plt.axvline(gpo, color='k', linewidth=2)
        # for i_gs, gs in enumerate(groundtruth_onset):
        #     plt.axvline(gs, color='r', linewidth=2)
        # plt.text(gs, ax1.get_ylim()[1], groundtruth_syllables[i_gs])

    ax1.set_ylabel('Mel bands', fontsize=12)
    ax1.get_xaxis().set_visible(False)
    ax1.axis('tight')

    ax2 = plt.subplot(412, sharex=ax1)
    plt.plot(np.arange(0, len(obs_syllable)) * hopsize_t, obs_syllable)
    for bsst in boundaries_syllable_start_time:
        plt.axvline(bsst, color='r', linewidth=2)

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


def figure_plot_hsmm(mfcc_line,
                     syllable_gt_onsets_0start,
                     phoneme_gt_onsets_0start_without_syllable_onsets,
                     B_map,
                     phoneme_score_labels,
                     path,
                     boundaries_phoneme_start_time,
                     boundaries_syllable_start_time,
                     syllable_score_durs,
                     phoneme_score_durs,
                     obs_joint_phn):

    # plot Error analysis figures
    tPlot, axes = plt.subplots(
        nrows=4, ncols=1, sharex=True, sharey=False,
        gridspec_kw={'height_ratios': [1, 2, 1, 1]})
    ax1, ax2, ax3, ax4 = axes[0], axes[1], axes[2], axes[3]

    y = np.arange(0, 80)
    x = np.arange(0, mfcc_line.shape[0]) * hopsize_t
    ax1.pcolormesh(x, y, np.transpose(mfcc_line[:, 80 * 10:80 * 11]))
    for gso in syllable_gt_onsets_0start:
        ax1.axvline(gso, color='r', linewidth=2)
    for gpo in phoneme_gt_onsets_0start_without_syllable_onsets:
        ax1.axvline(gpo, color='k', linewidth=2)

    ax1.set_ylabel('Mel bands', fontsize=12)
    ax1.get_xaxis().set_visible(False)
    ax1.axis('tight')

    # plot observation proba matrix
    n_states = B_map.shape[0]
    n_frame = B_map.shape[1]
    y = np.arange(n_states + 1)
    x = np.arange(n_frame) * hopsize / float(fs)
    ax2.pcolormesh(x, y, B_map)
    ax2.set_yticks(y)
    ax2.set_yticklabels(phoneme_score_labels, fontdict={'fontsize': 6})
    ax2.plot(x, path, 'b', linewidth=2)     # plot the decoding path
    for bpst in boundaries_phoneme_start_time:
        ax2.axvline(bpst, color='k', linewidth=1)
    for bsst in boundaries_syllable_start_time:
        ax2.axvline(bsst, color='r', linewidth=2)

    ax2.axis('tight')

    # # ax3 = plt.subplot(313, sharex=ax1)
    # # print(duration_score)
    # time_start = 0
    # # print(syllable_score_durs)
    # for ii_ds, ds in enumerate(syllable_score_durs):
    #     ax3.add_patch(
    #         patches.Rectangle(
    #             (time_start, 0),  # (x,y)
    #             ds,  # width
    #             3,  # height
    #             alpha=0.5
    #         ))
    #     time_start += ds
    #
    # time_start = 0
    # for psd in phoneme_score_durs:
    #     ax3.add_patch(
    #         patches.Rectangle(
    #             (time_start, 0.5),  # (x,y)
    #             psd,  # width
    #             2,  # height
    #             color='r',
    #             alpha=0.5
    #         ))
    #     time_start += psd
    # ax3.set_ylim((0, 3))
    # ax3.set_ylabel('Score duration', fontsize=12)
    # ax3.axis('tight')

    if obs_joint_phn is not None:
        ax4.plot(x, obs_joint_phn)
        ax4.axis('tight')

    plt.xlabel('Time (s)')

    plt.show()


def plot_data_parser_joint(filename):

    mfcc_line, \
    syllable_gt_onsets_0start, \
    phoneme_gt_onsets_0start_without_syllable_onsets, \
    obs_syllable, \
    boundaries_syllable_start_time, \
    obs_phoneme, \
    boundaries_phoneme_start_time = pickle.load(open(filename, 'r'))

    return mfcc_line, \
           syllable_gt_onsets_0start, \
           phoneme_gt_onsets_0start_without_syllable_onsets, \
           obs_syllable, \
           boundaries_syllable_start_time, \
           obs_phoneme, \
           boundaries_phoneme_start_time


def plot_data_parser_hsmm(filename):

    mfcc_line, \
    syllable_gt_onsets_0start, \
    phoneme_gt_onsets_0start_without_syllable_onsets, \
    B_map, \
    phoneme_score_labels_mapped, \
    path, \
    boundaries_syllable_start_time, \
    boundaries_phoneme_start_time = pickle.load(open(filename, 'r'))

    return mfcc_line, \
           syllable_gt_onsets_0start, \
           phoneme_gt_onsets_0start_without_syllable_onsets, \
           B_map, \
           phoneme_score_labels_mapped, \
           path, \
           boundaries_syllable_start_time, \
           boundaries_phoneme_start_time


def plot_joint_hsmm_interspeech(fn_joint, fn_hsmm, include_hsmm_matrix_labels=False):

    # load joint model plot data
    mfcc_line, \
    syllable_gt_onsets_0start, \
    phoneme_gt_onsets_0start_without_syllable_onsets, \
    obs_syllable_joint, \
    boundaries_syllable_start_time_joint, \
    obs_phoneme_joint, \
    boundaries_phoneme_start_time_joint = pickle.load(open(fn_joint, 'r'))

    # load hsmm model plot data
    _, \
    _, \
    _, \
    B_map, \
    phoneme_score_labels_mapped, \
    path_hsmm, \
    boundaries_syllable_start_time_hsmm, \
    boundaries_phoneme_start_time_hsmm = pickle.load(open(fn_hsmm, 'r'))

    linestyle_phoneme_onset = '--'

    fig = plt.figure(figsize=(16, 8))
    gs = gridspec.GridSpec(4, 1, height_ratios=[1, 1, 1, 1])

    # class weight
    ax1 = plt.subplot(gs[0])
    y = np.arange(0, 80)
    x = np.arange(0, mfcc_line.shape[0]) * hopsize_t
    plt.pcolormesh(x, y, np.transpose(mfcc_line[:, 80 * 7:80 * 8]))
    for gso in syllable_gt_onsets_0start:
        plt.axvline(gso, color='r', linewidth=2)
    for gpo in phoneme_gt_onsets_0start_without_syllable_onsets:
        plt.axvline(gpo, color='k', linewidth=2, linestyle=linestyle_phoneme_onset)
        # for i_gs, gs in enumerate(groundtruth_onset):
        #     plt.axvline(gs, color='r', linewidth=2)
        # plt.text(gs, ax1.get_ylim()[1], groundtruth_syllables[i_gs])

    ax1.set_ylabel('Ground truth', fontsize=15)
    ax1.get_xaxis().set_visible(False)
    ax1.get_yaxis().set_ticks(np.arange(0, 80, 20))
    ax1.axis('tight')

    ax2 = plt.subplot(gs[1], sharex=ax1)
    plt.plot(np.arange(0, len(obs_syllable_joint)) * hopsize_t, obs_syllable_joint)
    for bsst in boundaries_syllable_start_time_joint:
        plt.axvline(bsst, color='r', linewidth=2)

    ax2.set_ylabel('Proposed\nsyllable', fontsize=15)
    ax2.axis('tight')

    ax3 = plt.subplot(gs[2], sharex=ax1)
    plt.plot(np.arange(0, len(obs_phoneme_joint)) * hopsize_t, obs_phoneme_joint)
    for bpst in boundaries_phoneme_start_time_joint:
        plt.axvline(bpst, color='k', linewidth=2, linestyle=linestyle_phoneme_onset)
    for bsst in boundaries_syllable_start_time_joint:
        plt.axvline(bsst, color='r', linewidth=2)
    # for i_ib in range(len(i_boundary)-1):
    #     plt.axvline(i_boundary[i_ib] * hopsize_t, color='r', linewidth=2)
    # plt.text(i_boundary[i_ib] * hopsize_t, ax2.get_ylim()[1], syllables[i_line][i_ib])
    ax3.set_ylabel('Proposed\nphoneme', fontsize=15)
    ax3.axis('tight')
    ax3.set_ylim([0, 1.1])

    ax4 = plt.subplot(gs[3], sharex=ax1)
    for bpst in boundaries_phoneme_start_time_hsmm:
        ax4.axvline(bpst, color='k', linewidth=2, linestyle=linestyle_phoneme_onset)
    for bsst in boundaries_syllable_start_time_hsmm:
        ax4.axvline(bsst, color='r', linewidth=2)
    ax4.set_ylabel('Baseline', fontsize=15)

    if include_hsmm_matrix_labels:
        n_states = B_map.shape[0]
        n_frame = B_map.shape[1]
        y = np.arange(n_states + 1)
        x = np.arange(n_frame) * hopsize / float(fs)
        emission = ax4.pcolormesh(x, y, B_map, cmap=cm.coolwarm)
        cbar_ax4 = fig.colorbar(emission, orientation="horizontal", pad=0.2)
        cbar_ax4.ax.set_xlabel('log probabilities')

        ax4.set_yticks(y)
        ax4.set_yticklabels(phoneme_score_labels_mapped, fontdict={'fontsize': 10})
        ax4.plot(x, path_hsmm, 'b', linewidth=2)  # plot the decoding path
    else:
        ax4.get_yaxis().set_ticks([])

    ax4.axis('tight')

    plt.xlim([-0.01, mfcc_line.shape[0]*hopsize_t])
    plt.xlabel('Time (s)', fontsize=15)
    # plt.show()
    plt.savefig('google_speech_summit.png', dpi=200)


def plot_joint_hsmm_thesis(fn_joint, fn_hsmm, include_hsmm_matrix_labels=False):

    # load joint model plot data
    mfcc_line, \
    syllable_gt_onsets_0start, \
    phoneme_gt_onsets_0start_without_syllable_onsets, \
    obs_syllable_joint, \
    boundaries_syllable_start_time_joint, \
    obs_phoneme_joint, \
    boundaries_phoneme_start_time_joint = pickle.load(open(fn_joint, 'r'))

    # load hsmm model plot data
    _, \
    _, \
    _, \
    B_map, \
    phoneme_score_labels_mapped, \
    path_hsmm, \
    boundaries_syllable_start_time_hsmm, \
    boundaries_phoneme_start_time_hsmm = pickle.load(open(fn_hsmm, 'r'))

    linestyle_phoneme_onset = '--'

    fig = plt.figure(figsize=(16, 8))
    gs = gridspec.GridSpec(2, 1, height_ratios=[1, 3])

    # class weight
    ax1 = plt.subplot(gs[0])
    y = np.arange(0, 80)
    x = np.arange(0, mfcc_line.shape[0]) * hopsize_t
    plt.pcolormesh(x, y, np.transpose(mfcc_line[:, 80 * 7:80 * 8]))
    for gso in syllable_gt_onsets_0start:
        plt.axvline(gso, color='r', linewidth=2)
    for gpo in phoneme_gt_onsets_0start_without_syllable_onsets:
        plt.axvline(gpo, color='k', linewidth=2, linestyle=linestyle_phoneme_onset)
        # for i_gs, gs in enumerate(groundtruth_onset):
        #     plt.axvline(gs, color='r', linewidth=2)
        # plt.text(gs, ax1.get_ylim()[1], groundtruth_syllables[i_gs])

    ax1.set_ylabel('Ground truth', fontsize=15)
    ax1.get_xaxis().set_visible(False)
    ax1.get_yaxis().set_ticks(np.arange(0, 80, 20))
    ax1.axis('tight')

    ax2 = plt.subplot(gs[1], sharex=ax1)
    for bpst in boundaries_phoneme_start_time_hsmm:
        ax2.axvline(bpst, color='k', linewidth=2, linestyle=linestyle_phoneme_onset)
    for bsst in boundaries_syllable_start_time_hsmm:
        ax2.axvline(bsst, color='r', linewidth=2)
    ax2.set_ylabel('Baseline', fontsize=15)

    if include_hsmm_matrix_labels:
        n_states = B_map.shape[0]
        n_frame = B_map.shape[1]
        y = np.arange(n_states + 1)
        x = np.arange(n_frame) * hopsize / float(fs)
        emission = ax2.pcolormesh(x, y, B_map, cmap=cm.coolwarm)
        cbar_ax4 = fig.colorbar(emission, orientation="horizontal", pad=0.2)
        cbar_ax4.ax.set_xlabel('log probabilities')

        ax2.set_yticks(y)
        ax2.set_yticklabels(phoneme_score_labels_mapped, fontdict={'fontsize': 10})
        ax2.plot(x, path_hsmm, 'b', linewidth=2)  # plot the decoding path
    else:
        ax2.get_yaxis().set_ticks([])

    ax2.axis('tight')

    plt.xlim([-0.01, mfcc_line.shape[0]*hopsize_t])
    plt.xlabel('Time (s)', fontsize=15)
    # plt.show()
    plt.savefig('hsmm_baseline_results.png', bbox_inches='tight')

if __name__ == '__main__':
    fn_joint = '/home/gong/Documents/pycharmProjects/jingjuPhoneticSegmentation/plot_data/jan_joint_0/20171211SongRuoXuan/daxp_Qing_zao_qi_lai-Mai_shui-dxjky/student01_1.pkl'
    fn_hsmm = '/home/gong/Documents/pycharmProjects/jingjuPhoneticSegmentation/plot_data/hsmm/20171211SongRuoXuan/daxp_Qing_zao_qi_lai-Mai_shui-dxjky/student01_1.pkl'
    plot_joint_hsmm_thesis(fn_joint, fn_hsmm, include_hsmm_matrix_labels=True)