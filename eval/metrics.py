import numpy as np


def boundaryDetection(groundtruthBoundaries, detectedBoundaries, tolerance):
    '''
    :param groundtruthBoundaries:   phone boundary time in second, include the syllable start and end
    :param detectedBoundaries:      same structure as groundtruthBoundaries
    :param tolerance:               tolerance in second
    :return:                        number of detected phone boundaries,
                                    number of ground truth phone boundaries,
                                    number of correct detected boundaries,

    '''

    numDetectedBoundaries       = len(detectedBoundaries)
    numGroundtruthBoundaries    = len(groundtruthBoundaries)

    groundtruthTag              = [0]*numGroundtruthBoundaries

    for idx_gtb,gtb in enumerate(groundtruthBoundaries):
        for idx_db, db in enumerate(detectedBoundaries):

            if abs(db-gtb) < tolerance:
                # if idx_gtb has not been detected and is either syllable start or end
                # we detected a phone boundary as the syllable boundary
                # but we don't count this kind of boundary
                if not groundtruthTag[idx_gtb] and (idx_gtb == 0 or idx_gtb == numGroundtruthBoundaries-1):
                    numDetectedBoundaries   -= 1
                groundtruthTag[idx_gtb] = 1                                    # found boundary for boundary idx

    # don't count the syllable start and end
    numGroundtruthBoundaries    -= 2
    groundtruthTag.pop(-1)
    groundtruthTag.pop(0)

    numCorrect          = sum(groundtruthTag)

    return numDetectedBoundaries, numGroundtruthBoundaries, numCorrect

def metrics(numDetectedBoundaries, numGroundtruthBoundaries, numCorrect):

    # hit rate or correct detection rate or recall rate
    HR  = numCorrect/float(numGroundtruthBoundaries)

    # over segmentation
    OS  = numDetectedBoundaries/float(numGroundtruthBoundaries) - 1.0

    # false alarm rate
    FAR = (numDetectedBoundaries-numCorrect)/float(numGroundtruthBoundaries)

    # F-measure
    PCR = 1.0 - FAR     # precision rate
    F   = 2.0*PCR*HR/(PCR+HR)

    # R-value
    r1  = np.sqrt((100-HR*100)**2.0 + (OS*100)**2.0)
    r2  = (-OS*100 + HR*100 - 100)/np.sqrt(2.0)
    R   = (1.0 - (np.abs(r1)+np.abs(r2))/200.0)

    deletion    = numGroundtruthBoundaries  - numCorrect
    insertion   = numDetectedBoundaries     - numCorrect

    return HR, OS, FAR, F, R, deletion, insertion