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

import os
import numpy as np
from itertools import combinations
from textgridParser import syllableTextgridExtraction

from parameters import *

# wav_path        = '/Users/gong/Documents/MTG document/Jingju arias/phonemeSeg/wav'
# textgrid_path   = '/Users/gong/Documents/MTG document/Jingju arias/phonemeSeg/textgrid'

def getRecordings(wav_path):
    recordings      = []
    for root, subFolders, files in os.walk(wav_path):
            for f in files:
                file_prefix, file_extension = os.path.splitext(f)
                if file_prefix != '.DS_Store':
                    recordings.append(file_prefix)

    return recordings

def testRecordings(boundaries,proportion_testset):
    '''
    :param boundaries: a list of boundary number of each recording
    :param proportion_testset:
    :return: a list of test recordings
    '''

    sum_boundaries = sum(boundaries)
    boundaries     = np.array(boundaries)
    subsets        = []

    for ii in range(1,len(boundaries)):
        for subset in combinations(range(len(boundaries)),ii):
            subsets.append([subset,abs(sum(boundaries[np.array(subset)])/float(sum_boundaries)-proportion_testset)])

    subsets        = np.array(subsets)
    subsets_sorted = subsets[np.argsort(subsets[:,1]),0]

    return subsets_sorted[0]

def getRecordingNumber(train_test_string):
    '''
    return number of recording, test recording contains 25% ground truth boundaries
    :param train_test_string: 'TRAIN' or 'TEST'
    :return:
    '''

    train_recordings        = [0,2,4,5,6,8,9,10,12,13,14]
    test_recordings         = [1,3,7,11]

    if train_test_string == 'TRAIN':
        number_recording     = train_recordings
    else:
        number_recording     = test_recordings

    return number_recording

if __name__ == '__main__':

    recordings = getRecordings(wav_path)

    boundaries  = []
    numSyllable_all, numVoiced_all, numUnvoiced_all = 0,0,0
    lengthSyllable_all, lengthVoiced_all, lengthUnvoiced_all = [],[],[]
    for recording in recordings:

        boundaries_oneSong  = 0
        nestedPhonemeLists, numSyllables, numPhonemes   \
            = syllableTextgridExtraction(textgrid_path,recording,'pinyin','details')

        numSyllable_all += numSyllables

        for pho in nestedPhonemeLists:
            lengthSyllable_all.append(pho[0][1]-pho[0][0])
            for p in pho[1]:
                if p[2] == '':
                    continue
                if p[2] in ['c','k','f','x']:
                    numUnvoiced_all += 1
                    lengthUnvoiced_all.append(p[1]-p[0])
                else:
                    numVoiced_all   += 1
                    lengthVoiced_all.append(p[1]-p[0])

        for nestedPho in nestedPhonemeLists:
            boundaries_oneSong  += len(nestedPho[1])-1

        boundaries.append(boundaries_oneSong)

    proportion_testset = 0.25

    index_testset      = testRecordings(boundaries, proportion_testset)

    # output test set index
    print index_testset

    # output statistics of the dataset
    print numSyllable_all, numVoiced_all, numUnvoiced_all
    print np.mean(lengthSyllable_all), np.mean(lengthVoiced_all), np.mean(lengthUnvoiced_all)
    print np.std(lengthSyllable_all), np.std(lengthVoiced_all), np.std(lengthUnvoiced_all)

