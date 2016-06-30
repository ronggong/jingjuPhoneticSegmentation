import numpy as np
import essentia.standard as ess
from PLP import PLP
from MRCG import MRCG
from Fdeltas import Fdeltas

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
    ZCR             = ess.ZeroCrossingRate()
    ENERGY          = ess.Energy()
    mfcc            = []
    mfccBands       = []
    gfcc            = []
    energy          = []
    zcr             = []
    autoCorrelation = []
    mX              = []

    print 'calculating ', feature_select, ' ... ...'

    for frame in ess.FrameGenerator(audio, frameSize=framesize, hopSize=hopsize):

        frame_audio     = frame
        frame           = WINDOW(frame)
        mXFrame         = SPECTRUM(frame)
        mX.append(mXFrame)

        energyFrame     = ENERGY(mXFrame)
        energy.append(energyFrame)

        if feature_select == 'mfcc' or feature_select == 'dmfcc':
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

        if feature_select == 'zcr':
            zcrFrame        = ZCR(frame_audio)
            zcr.append(zcrFrame)

        if feature_select == 'autoCorrelation':
            autoCorrelationFrame = np.corrcoef(frame_audio)
            autoCorrelation.append(autoCorrelationFrame)

    mX              = np.array(mX)

    if feature_select == 'mfcc':
        feature         = np.array(mfcc)

    if feature_select == 'dmfcc':
        dmfcc           = Fdeltas(np.array(mfcc).transpose(),w=9)
        ddmfcc          = Fdeltas(dmfcc,w=5)
        feature         = np.transpose(np.vstack((dmfcc,ddmfcc)))

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

    elif feature_select == 'zcr':
        feature             = np.array(zcr)

    elif feature_select == 'autoCorrelation':
        feature             = np.array(autoCorrelation)

    else:
        feature,d_MRCG,dd_MRCG = MRCG(audio,fs=fs)

    return feature,energy,mX