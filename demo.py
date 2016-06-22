# -*- coding: utf-8 -*-

import sys,os,time,csv
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), "speechSegmentation"))
sys.path.append(os.path.join(os.path.dirname(__file__), "eval"))


import Aversano as Ave
import Hoang    as Hoa
import Estevan  as Est
import Winebarger as Win

import textgridParser
import metrics
import essentia.standard as ess
import features

def syllableTextgridExtraction(textgrid_path, recording):

    '''
    Extract syllable boundary and phoneme boundary from textgrid
    :param textgrid_path:
    :param recording:
    :return:
    nestedPhonemeList, element[0] - syllable, element[1] - a list containing the phoneme of the syllable
    '''

    textgrid_file   = os.path.join(textgrid_path,recording+'.TextGrid')

    syllableList    = textgridParser.textGrid2WordList(textgrid_file, whichTier='pinyin')
    phonemeList     = textgridParser.textGrid2WordList(textgrid_file, whichTier='xsampadetails')

    # parse syllables of groundtruth
    nestedPhonemeLists, numSyllables, numPhonemes   = textgridParser.wordListsParseByLines(syllableList, phonemeList)

    return nestedPhonemeLists, numSyllables, numPhonemes

def featureSyllableSegmentation(feature_path, recording, nestedPhonemeLists,varin):

    '''
    Segment the audio for each syllable
    :param wav_path:
    :param recording:
    :param fs:
    :param nestedPhonemeLists:
    :return:
    feature_syllables is a list containing the feature for each syllable, len(audio_syllables) = len(nestedPhonemeLists)
    '''
    hopsize         = varin['hopsize']
    fs              = varin['fs']

    energy_filename         = os.path.join(feature_path,'energy'+'_'+recording+'_'
                                               +str(varin['framesize'])+'_'+str(varin['hopsize'])+'.npy')
    spec_filename           = os.path.join(feature_path,'spec'+'_'+recording+'_'
                                               +str(varin['framesize'])+'_'+str(varin['hopsize'])+'.npy')
    feature_filename        = os.path.join(feature_path,varin['feature_select']+'_'+recording+'_'
                                               +str(varin['framesize'])+'_'+str(varin['hopsize'])+'.npy')

    feature1                  = np.load(feature_filename)
    spec                      = np.load(spec_filename)

    if varin['phonemeSegFunction'] == Win.mainFunction:
        mfcc_feature_filename     = os.path.join(feature_path,'mfcc'+'_'+recording+'_'
                                               +str(varin['framesize'])+'_'+str(varin['hopsize'])+'.npy')
        feature2                  = np.load(mfcc_feature_filename)
    else:
        feature2                  = np.load(energy_filename)

    feature1_syllables      = []
    feature2_syllables      = []
    spec_syllables          = []
    for nestedPho in nestedPhonemeLists:
        syllable_start_frame    = int(round(nestedPho[0][0]*fs/hopsize))
        syllable_end_frame      = int(round(nestedPho[0][1]*fs/hopsize))
        feature_syllable        = feature1[syllable_start_frame:syllable_end_frame,:]
        feature1_syllables.append(feature_syllable)

        if varin['phonemeSegFunction'] == Win.mainFunction:
            # mfcc
            feature2_syllable       = feature2[syllable_start_frame:syllable_end_frame,:]
        else:
            # energy
            feature2_syllable       = feature2[syllable_start_frame:syllable_end_frame,]

        feature2_syllables.append(feature2_syllable)

        spec_syllable           = spec[syllable_start_frame:syllable_end_frame,:]
        spec_syllables.append(spec_syllable)


    return feature1_syllables, feature2_syllables, spec_syllables

def eval4oneSong(feature1_syllables, feature2_syllables, spec_syllables, nestedPhonemeLists,
                 tolerance, phonemeSegfunction, varin):

    sumNumGroundtruthBoundaries,sumNumDetectedBoundaries,sumNumCorrect = 0,0,0

    for ii, feature in enumerate(feature1_syllables):

        print 'evaluate syllable ', ii+1, ' in', len(feature1_syllables), ' feature: ', varin['feature_select']

        # phonemes of syllable
        phoList                 = nestedPhonemeLists[ii][1]
        syllable_start_time     = phoList[0][0]
        groundtruthBoundaries   = []

        for pho in phoList:
            groundtruthBoundaries.append(pho[0] - syllable_start_time)

        # syllable end time
        groundtruthBoundaries.append(phoList[-1][1] - syllable_start_time)

        if varin['phonemeSegFunction'] == Hoa.mainFunction:
            varin['energy']      = feature2_syllables[ii]
        elif varin['phonemeSegFunction'] == Win.mainFunction and varin['feature_select'] != 'mfcc':
            varin['mfcc']        = feature2_syllables[ii]

        detectedBoundaries   = phonemeSegfunction(feature,spec_syllables[ii],varin)

        numDetectedBoundaries, numGroundtruthBoundaries, numCorrect = \
            metrics.boundaryDetection(groundtruthBoundaries=groundtruthBoundaries,
                                  detectedBoundaries=detectedBoundaries,
                                  tolerance=tolerance)

        sumNumGroundtruthBoundaries += numGroundtruthBoundaries
        sumNumDetectedBoundaries    += numDetectedBoundaries
        sumNumCorrect               += numCorrect

    return  sumNumGroundtruthBoundaries, sumNumDetectedBoundaries, sumNumCorrect


# filename    = '/Users/gong/Documents/MTG document/NACTA/Recordings/ZhuoFangCao-WangWenZhi/160407/可恨老狗太不良/characters/student02/fen/teacher02.wav'
# filename    = '/Users/gong/Documents/MTG document/TIMIT/LDC93S1_short.wav'
framesize_t = 0.020
hopsize_t   = 0.010
# fs          = 16000
fs          = 44100

varin       = {}
varin['phonemeSegFunction'] = Hoa.mainFunction
varin['plot']        = False
varin['fs']          = 44100

varin['framesize']   = int(round(framesize_t*fs))
varin['hopsize']     = int(round(hopsize_t*fs))

feature_set          = ['mfcc', 'mfccBands','gfcc','plpcc','plp','rasta-plpcc', 'rasta-plp', 'bark','mrcg']
varin['feature_select'] = 'mfccBands'
varin['rasta']       = False

varin['a']           = 20
varin['b']           = 0.1
varin['c']           = 7

# aveBoundary   = Ave.mainFunction(filename=filename,varin=varin)

varin['max_band']    = 8
varin['l']           = 2                 # segmentation interval
varin['h0']          = 0.6               # a0 peak threshold
varin['h1']          = 0.08              # a1 peak threshold
varin['h2']          = 0.0725
varin['th_phone']    = 0.2
varin['q']           = 3
varin['k']           = 3
varin['delta']       = 2.0
varin['step2']       = False

# hoaBoundary   = Hoa.mainFunction(filename=filename,varin=varin)

varin['N_win']  = 18            # sliding window
varin['gamma']  = 10**(-1)          # RBF width

# Est.mainFunction(filename=filename,varin=varin)

varin['p_lambda']    = 0.35
varin['mode_bic']    = 0
varin['h2']          = 0.0
varin['alpha']       = 0.5
varin['winmax']      = 0.35              # max dynamic window size
# winBoundary   = Win.mainFunction(filename=filename,varin=varin)


wav_path        = '/Users/gong/Documents/MTG document/Jingju arias/phonemeSeg/wav'
textgrid_path   = '/Users/gong/Documents/MTG document/Jingju arias/phonemeSeg/textgrid'
feature_path    = '/Users/gong/Documents/MTG document/Jingju arias/phonemeSeg/feature'

tolerance       = 0.02

recordings      = []
for root, subFolders, files in os.walk(wav_path):
        for f in files:
            file_prefix, file_extension = os.path.splitext(f)
            recordings.append(file_prefix)

# for feature in feature_set:
#     varin['feature_select']     = feature
#     if feature == 'mrcg':
#         varin['hopsize']        = 0.01

'''
####---- feature extraction
for recording in recordings:
    wav_file   = os.path.join(wav_path,recording+'.wav')
    energy_filename         = os.path.join(feature_path,'energy'+'_'+recording+'_'
                                               +str(varin['framesize'])+'_'+str(varin['hopsize'])+'.npy')
    spec_filename           = os.path.join(feature_path,'spec'+'_'+recording+'_'
                                               +str(varin['framesize'])+'_'+str(varin['hopsize'])+'.npy')
    for featurename in feature_set:
        print 'saving feature for ', recording, ', feature ', featurename
        feature_filename        = os.path.join(feature_path,featurename+'_'+recording+'_'
                                               +str(varin['framesize'])+'_'+str(varin['hopsize'])+'.npy')
        varin['feature_select'] = featurename
        feature, energy, spec = features.features(wav_file,varin)

        np.save(feature_filename,feature)

        if featurename == feature_set[-1]:
            np.save(energy_filename,energy)
            np.save(spec_filename,spec)

'''


####---- Hoang's method

start = time.time()

varin['phonemeSegFunction'] = Hoa.mainFunction
varin['feature_select'] = 'mfccBands'

eval_result_file_name = '/Users/gong/Documents/MTG document/Jingju arias/phonemeSeg' + '/hoa_mfccBands2.csv'

with open(eval_result_file_name, 'w') as testfile:
    csv_writer = csv.writer(testfile)

    for l in [2,4,6,8]:
        for h1 in [0.0,0.2,0.4,0.6,0.8,1.0]:
            for h2 in [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]:

                varin['l']          = l
                varin['h1']         = h1
                varin['h2']         = h2

                numGroundtruthBoundaries, numDetectedBoundaries, numCorrect = 0,0,0
                for recording in recordings:
                    print 'evaluate ', recording, ' l,h1,h2', l, h1, h2
                    nestedPhonemeLists, numSyllables, numPhonemes = syllableTextgridExtraction(textgrid_path,recording)
                    feature_Syllables, spec_syllables, energy_syllables = featureSyllableSegmentation(feature_path, recording,
                                                                                                nestedPhonemeLists, varin)
                    sumNumGroundtruthBoundaries, sumNumDetectedBoundaries, sumNumCorrect = \
                        eval4oneSong(feature_Syllables, spec_syllables, energy_syllables, nestedPhonemeLists,
                                     tolerance, varin['phonemeSegFunction'], varin)

                    numGroundtruthBoundaries    += sumNumGroundtruthBoundaries
                    numDetectedBoundaries       += sumNumDetectedBoundaries
                    numCorrect                  += sumNumCorrect

                HR, OS, FAR, F, R, deletion, insertion = \
                    metrics.metrics(numDetectedBoundaries, numGroundtruthBoundaries, numCorrect)

                csv_writer.writerow([l, h1, h2, HR, OS, FAR, F, R, deletion, insertion])

end = time.time()
print 'elapse time: ', end-start



'''
####---- Winebarger's method

start = time.time()

varin['phonemeSegFunction'] = Hoa.mainFunction
varin['feature_select'] = 'mfcc'

eval_result_file_name = '/Users/gong/Documents/MTG document/Jingju arias/phonemeSeg' + '/win_mfcc.csv'

with open(eval_result_file_name, 'w') as testfile:
    csv_writer = csv.writer(testfile)

    for h2 in [0.0,0.02,0.04,0.06,0.08,0.1]:
        for alpha in [0.2,0.4,0.6,0.8,1.0]:
            for p_lambda in [0.2,0.4,0.6,0.8,1.0]:

                varin['h2']             = h2
                varin['alpha']          = alpha
                varin['p_lambda']       = p_lambda

                numGroundtruthBoundaries, numDetectedBoundaries, numCorrect = 0,0,0
                for recording in recordings:
                    print 'evaluate ', recording, ' l,h1,h2', h2, alpha, p_lambda
                    nestedPhonemeLists, numSyllables, numPhonemes = syllableTextgridExtraction(textgrid_path,recording)
                    feature_Syllables, spec_syllables, energy_syllables = featureSyllableSegmentation(feature_path, recording,
                                                                                                nestedPhonemeLists, varin)
                    sumNumGroundtruthBoundaries, sumNumDetectedBoundaries, sumNumCorrect = \
                        eval4oneSong(feature_Syllables, spec_syllables, energy_syllables, nestedPhonemeLists,
                                     tolerance, varin['phonemeSegFunction'], varin)

                    numGroundtruthBoundaries    += sumNumGroundtruthBoundaries
                    numDetectedBoundaries       += sumNumDetectedBoundaries
                    numCorrect                  += sumNumCorrect

                HR, OS, FAR, F, R, deletion, insertion = \
                    metrics.metrics(numDetectedBoundaries, numGroundtruthBoundaries, numCorrect)

                csv_writer.writerow([h2, alpha, p_lambda, HR, OS, FAR, F, R, deletion, insertion])

end = time.time()
print 'elapse time: ', end-start
'''





