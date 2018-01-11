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

import numpy as np


def boundaryDetection(groundtruthBoundaries, detectedBoundaries, tolerance):
    '''
    :param groundtruthBoundaries:   phone boundary time in second, include the syllable start and end
    :param detectedBoundaries:      same structure as groundtruthBoundaries, include the syllable start and end
    :param tolerance:               tolerance in second
    :return:                        number of detected phone boundaries,
                                    number of ground truth phone boundaries,
                                    number of correct detected boundaries,

    '''

    numDetectedBoundaries       = len(detectedBoundaries)
    numGroundtruthBoundaries    = len(groundtruthBoundaries)

    correctTag                  = [0]*numGroundtruthBoundaries
    detectedBoundariesTag       = [0]*numDetectedBoundaries
    startBoundDetected          = 0
    endBoundDetected            = 0
    detectedBoundariesToDelete  = 0

    for idx_gtb,gtb in enumerate(groundtruthBoundaries):
        for idx_db,db in enumerate(detectedBoundaries):

            if not correctTag[idx_gtb] and not detectedBoundariesTag[idx_db] and abs(db-gtb) < tolerance:
                # we detected a phone boundary at the same time the syllable boundary
                # but we don't count this kind of boundary
                correctTag[idx_gtb]                  = 1         # found boundary for boundary idx
                detectedBoundariesTag[idx_db]        = 1
                if not startBoundDetected and idx_gtb == 0:
                    detectedBoundariesToDelete      += 1
                    startBoundDetected              = 1
                if not endBoundDetected and idx_gtb == numGroundtruthBoundaries-1:
                    detectedBoundariesToDelete      += 1
                    endBoundDetected                 = 1
                break

    # don't count the syllable start and end
    numGroundtruthBoundaries    -= 2
    numDetectedBoundaries       -= detectedBoundariesToDelete

    numCorrect          = sum(correctTag) - startBoundDetected - endBoundDetected

    return numDetectedBoundaries, numGroundtruthBoundaries, numCorrect

def offsetTh(groundtruth,tolerance):

    thOffset = []

    groundtruth = np.array(groundtruth)
    if len(groundtruth):
        for gt in groundtruth[:,1]-groundtruth[:,0]:
            twentyP = gt*0.2
            thOffset.append(max(twentyP,tolerance))

    return thOffset

def intervalDetection(groundtruth, detected, tolerance):
        '''
        :param groundtruth: a list [[start0,end0],[start1,end1],[start2,end2],...]
        :param detected: a list [[start0,end0],[start1,end1],[start2,end2],...]
        :return: COnOff, COn, OBOn, OBOff
        '''

        numGroundtruth    = len(groundtruth)
        numDetected       = len(detected)
        numCorrect        = 0

        thOnset     = tolerance
        thOffset    = offsetTh(groundtruth,tolerance)

        if numGroundtruth:
            for s in detected:
                for gti in range(len(groundtruth)):
                    if abs(groundtruth[gti][0]-s[0]) < thOnset and abs(groundtruth[gti][1]-s[1]) < thOffset[gti]:
                        numCorrect += 1
                        break       # break the first loop if s have been already mapped to

        return numDetected, numGroundtruth, numCorrect

def metrics(numDetectedBoundaries, numGroundtruthBoundaries, numCorrect):

    # hit rate or correct detection rate or recall rate
    HR  = numCorrect/float(numGroundtruthBoundaries)

    # over segmentation
    OS  = numDetectedBoundaries/float(numGroundtruthBoundaries) - 1.0

    # false alarm rate
    FAR = (numDetectedBoundaries-numCorrect)/float(numGroundtruthBoundaries)

    # F-measure
    PCR = numCorrect/float(numDetectedBoundaries)     # precision rate
    F   = 2.0*PCR*HR/(PCR+HR)

    # R-value
    r1  = np.sqrt((100-HR*100)**2.0 + (OS*100)**2.0)
    r2  = (-OS*100 + HR*100 - 100)/np.sqrt(2.0)
    R   = (1.0 - (np.abs(r1)+np.abs(r2))/200.0)

    deletion    = numGroundtruthBoundaries  - numCorrect
    insertion   = numDetectedBoundaries     - numCorrect

    return HR, OS, FAR, F, R, deletion, insertion

def sortResultByCol(result_filename, result_sorted_filename,col):

    result  = np.loadtxt(result_filename,delimiter=',')
    index_sorted = np.argsort(result[:,col])
    result_sorted = result[index_sorted,:]
    np.savetxt(result_sorted_filename,result_sorted,fmt='%.3f',delimiter=',')

    return None
