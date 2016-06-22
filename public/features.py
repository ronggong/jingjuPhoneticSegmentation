import numpy as np
import essentia.standard as ess
from PLP import PLP
from MRCG import MRCG

def features(filename,varin):


    fs              = varin['fs']
    framesize       = varin['framesize']
    hopsize         = varin['hopsize']
    feature_select  = varin['feature_select']

    audio           = ess.MonoLoader(downmix = 'left', filename = filename, sampleRate = fs)()

    # spectrogram init
    winAnalysis     = 'hann'
    N               = 2 * framesize                     # padding 1 time framesize
    SPECTRUM        = ess.Spectrum(size=N)
    WINDOW          = ess.Windowing(type=winAnalysis, zeroPadding=N-framesize)
    highFrequencyBound = fs/2 if fs/2<11000 else 11000
    MFCC            = ess.MFCC(sampleRate=fs,highFrequencyBound=highFrequencyBound,inputSize=framesize+1)
    GFCC            = ess.GFCC(sampleRate=fs,highFrequencyBound=highFrequencyBound)
    ENERGY          = ess.Energy()
    mfcc            = []
    mfccBands       = []
    gfcc            = []
    energy          = []
    mX              = []

    print 'calculating ', feature_select, ' ... ...'

    for frame in ess.FrameGenerator(audio, frameSize=framesize, hopSize=hopsize):

        frame           = WINDOW(frame)
        mXFrame         = SPECTRUM(frame)
        mX.append(mXFrame)

        energyFrame     = ENERGY(mXFrame)
        energy.append(energyFrame)

        if feature_select == 'mfcc':
            bands,mfccFrame = MFCC(mXFrame)
            mfccFrame       = mfccFrame[1:]
            mfcc.append(mfccFrame)

        if feature_select == 'mfccBands':
            bands,mfccFrame = MFCC(mXFrame)
            mfccBands.append(bands)

        if feature_select == 'gfcc':
            bands,gfccFrame = GFCC(mXFrame)
            gfccFrame       = gfccFrame[1:]
            gfcc.append(gfccFrame)

    mX              = np.array(mX)

    if feature_select == 'mfcc':
        feature         = np.array(mfcc)

    elif feature_select == 'mfccBands':
        feature         = np.array(mfccBands)

    elif feature_select == 'gfcc':
        feature         = np.array(gfcc)

    elif feature_select == 'plpcc':
        feature,plp,bark    = PLP(mX,modelorder=12,rasta=False)
        feature         = feature[:,1:]

    elif feature_select == 'plp':
        plpcc,feature,bark  = PLP(mX,modelorder=12,rasta=False)
        feature         = feature[:,1:]

    elif feature_select == 'rasta-plpcc':
        feature,plp,bark    = PLP(mX,modelorder=12,rasta=True)
        feature         = feature[:,1:]

    elif feature_select == 'rasta-plp':
        plpcc,feature,bark  = PLP(mX,modelorder=12,rasta=True)
        feature         = feature[:,1:]

    elif feature_select == 'bark':
        plpcc,plp,feature   = PLP(mX,modelorder=12,rasta=False)

    else:
        feature,d_MRCG,dd_MRCG = MRCG(audio,fs=fs)

    return feature,energy,mX