import numpy as np
import essentia.standard as ess
from get_avg import get_avg
from Fdeltas import Fdeltas


def MRCG(x,fs=44100,framesize1=0.02,framesize2=0.2,hopsize=0.01):

    hopsize         = int(hopsize*fs)
    # spectrogram init
    winAnalysis     = 'hann'

    ####---- cochleagram 1
    framesize       = int(framesize1*fs)
    N               = 2 * framesize                     # padding 1 time framesize
    SPECTRUM        = ess.Spectrum(size=N)
    WINDOW          = ess.Windowing(type=winAnalysis, zeroPadding=N-framesize)
    highFrequencyBound = fs/2 if fs/2<11000 else 11000
    ERBBANDS        = ess.ERBBands(sampleRate=fs,highFrequencyBound=highFrequencyBound,inputSize=framesize+1)

    cochlea1        = []
    for frame in ess.FrameGenerator(x, frameSize=framesize, hopSize=hopsize):
        frame       = WINDOW(frame)
        mXFrame     = SPECTRUM(frame)
        erbFrame    = np.log10(ERBBANDS(mXFrame)+np.finfo(np.float).eps)
        cochlea1.append(erbFrame)
    cochlea1        = np.array(cochlea1)

    ####---- cochleagram 2
    framesize       = int(framesize2*fs)
    N               = 2 * framesize                     # padding 1 time framesize
    SPECTRUM        = ess.Spectrum(size=N)
    WINDOW          = ess.Windowing(type=winAnalysis, zeroPadding=N-framesize)
    highFrequencyBound = fs/2 if fs/2<11000 else 11000
    ERBBANDS        = ess.ERBBands(sampleRate=fs,highFrequencyBound=highFrequencyBound,inputSize=framesize+1)

    cochlea2        = []
    for frame in ess.FrameGenerator(x, frameSize=framesize, hopSize=hopsize):
        frame       = WINDOW(frame)
        mXFrame     = SPECTRUM(frame)
        erbFrame    = np.log10(ERBBANDS(mXFrame)+np.finfo(np.float).eps)
        cochlea2.append(erbFrame)
    cochlea2        = np.array(cochlea2)

    ####---- smoothed version
    cochlea3        = get_avg(cochlea1,5,5)
    cochlea4        = get_avg(cochlea1,11,11)

    all_cochleas    = np.hstack((cochlea1,cochlea2,cochlea3,cochlea4))

    ####---- delta
    d_all_cochleas  = Fdeltas(all_cochleas.T)
    dd_all_cochleas = Fdeltas(Fdeltas(all_cochleas.T,5),5)

    d_all_cochleas  = d_all_cochleas.T
    dd_all_cochleas = dd_all_cochleas.T

    return all_cochleas, d_all_cochleas, dd_all_cochleas



