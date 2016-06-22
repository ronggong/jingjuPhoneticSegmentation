import essentia.standard as ess
import numpy as np
import matplotlib.pyplot as plt

import sys,os

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)),"../public"))

from Fdeltas    import Fdeltas
from heuristics import heuristics

def mainFunction(filename,fs,framesize,hopsize,h2,alpha,p_lambda):

    '''
    main procedure of algorithm
    :param filename:
    :param fs:
    :param framesize:
    :param hopsize:
    :return:
    '''

    # load audio
    audio           = ess.MonoLoader(filename = filename, sampleRate = fs)()

    # spectrogram init
    winAnalysis     = 'hann'
    N               = 2 * framesize                     # padding 1 time framesize
    SPECTRUM        = ess.Spectrum(size=N)
    WINDOW          = ess.Windowing(type=winAnalysis, zeroPadding=N-framesize)
    highFrequencyBound = fs/2 if fs/2<11000 else 11000
    MFCC            = ess.MFCC(sampleRate=fs,highFrequencyBound=highFrequencyBound)
    PEAK            = ess.PeakDetection(interpolate=False,maxPeaks=99999)
    mfcc            = []
    mX              = []

    print 'calculating MFCC ... ...'

    for frame in ess.FrameGenerator(audio, frameSize=framesize, hopSize=hopsize):

        frame           = WINDOW(frame)
        mXFrame         = SPECTRUM(frame)
        mX.append(mXFrame)
        bands,mfccFrame = MFCC(mXFrame)
        mfccFrame       = mfccFrame[1:]

        mfcc.append(mfccFrame)

    mX              = np.array(mX)
    mX              = np.transpose(mX)
    mfcc            = np.array(mfcc)
    T               = mfcc.shape[0]                         # time
    D               = mfcc.shape[1]                         # feature dimension

    print 'calculating delta mfcc ... ...'

    d_mfcc          = Fdeltas(mfcc.transpose(), w=9)
    d_mfcc          = np.transpose(d_mfcc)

    # Spectral variation function
    SVF             = np.sqrt(np.sum(d_mfcc**2.0,axis=1))
    SVF             = (SVF - np.min(SVF))/(np.max(SVF)-np.min(SVF))

    # peaks and valleys
    p_SVF,a_SVF     = PEAK(np.array(SVF,dtype=np.float32))
    p_SVF           = np.array(np.round(p_SVF*(T-1)),dtype=np.int)

    p_v_SVF,a_v_SVF = PEAK(np.array(1-SVF,dtype=np.float32))
    p_v_SVF         = np.array(np.round(p_v_SVF*(T-1)),dtype=np.int)

    # heuristics
    p_SVF,a_SVF,p_v_SVF,a_v_SVF = heuristics(p_SVF,a_SVF,p_v_SVF,a_v_SVF,SVF,fs,hopsize,h2,alpha)

    index2Delete    = []
    if len(p_SVF) > 3:
        # BIC
        ii              = 1
        jj              = 1
        # dynamic windowing BIC
        while ii < len(p_SVF)-1:
            p_0             = p_SVF[ii-jj]
            p_1             = p_SVF[ii]
            p_2             = p_SVF[ii+1]

            delta_ABF2   = ABF2(d_mfcc[p_0:p_1,:],d_mfcc[p_1:p_2,:],d_mfcc[p_0:p_2,:],p_lambda)
            if  delta_ABF2 > 0:
                jj              = 1

            else:
                jj              += 1
                index2Delete.append(ii)
            ii              += 1

            if ii >= len(p_SVF)-1: break

            # print delta_BIC, p_0, p_1, p_2,

    p_ABF2          = np.delete(p_SVF,index2Delete)
    a_ABF2          = np.delete(a_SVF,index2Delete)


