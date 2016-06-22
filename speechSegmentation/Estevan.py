# -*- coding: utf-8 -*-

import essentia.standard as ess
import numpy as np
import matplotlib.pyplot as plt
from rlscore.learner import mmc

import sys,os

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)),"../public"))

from Fdeltas import Fdeltas

def mainFunction(audio,varin):

    '''
    main procedure of algorithm
    :param audio:
    :param fs:
    :param framesize:
    :param hopsize:
    :return:
    '''

    fs              = varin['fs']
    framesize       = varin['framesize']
    hopsize         = varin['hopsize']
    N_win           = varin['N_win']
    gamma           = varin['gamma']

    # spectrogram init
    winAnalysis     = 'hann'
    N               = 2 * framesize                     # padding 1 time framesize
    SPECTRUM        = ess.Spectrum(size=N)
    WINDOW          = ess.Windowing(type=winAnalysis, zeroPadding=N-framesize)
    highFrequencyBound = fs/2 if fs/2<11000 else 11000
    MFCC            = ess.MFCC(sampleRate=fs,highFrequencyBound=highFrequencyBound)
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

    print 'calculating delta, delta-delta mfcc ... ...'

    d_mfcc          = Fdeltas(mfcc.transpose(), w=9)
    dd_mfcc         = Fdeltas(d_mfcc,           w=5)
    d_mfcc          = np.transpose(d_mfcc)
    dd_mfcc         = np.transpose(dd_mfcc)

    mfcc            = np.hstack((mfcc,d_mfcc,dd_mfcc))

    T               = mfcc.shape[0]                         # time
    D               = mfcc.shape[1]                         # feature dimension
    swcr            = np.zeros(shape=(T-N_win,N_win))       # sliding window clustering representation
    ED              = np.zeros(shape=(T-N_win,1))           # euclidean distance array

    print 'calculating MMC, ED ... ...'

    for ii in range(T-N_win):
        mfcc_sliding    = mfcc[ii:ii+N_win,:]
        mmc_object      = mmc.MMC(X=mfcc_sliding,number_of_clusters=2,kernel='GaussianKernel',gamma=gamma,regparam=1.0)
        results         = mmc_object.getResults()
        swcr[ii,:]      = results['predicted_clusters_for_training_data']

        # mfcc vectors of two clusters
        mfcc_c1         = mfcc_sliding[swcr[ii,:]==-1,:]
        mfcc_c2         = mfcc_sliding[swcr[ii,:]==1,:]

        # mean vector of two clusters
        if not mfcc_c1.size or not mfcc_c2.size:
            ed          = 0
        else:
            mfcc_c1_mean    = np.mean(mfcc_c1,axis=0)
            mfcc_c2_mean    = np.mean(mfcc_c2,axis=0)
            ed              = np.linalg.norm(mfcc_c1_mean-mfcc_c2_mean)

        # Euclidean distance
        ED[ii]          = ed

    swcr            = np.transpose(swcr)

    timestamps_audio= np.arange(len(audio))/float(fs)
    timestamps      = 8 * hopsize/float(fs) + np.arange(N_win/2,T-N_win/2) * (hopsize/float(fs))
    binFreqs        = np.arange(-N_win/2,N_win/2)
    f, axarr = plt.subplots(3, sharex=True)
    axarr[0].plot(timestamps_audio,audio)
    axarr[1].pcolormesh(timestamps, binFreqs, swcr)
    axarr[1].set_ylabel('frame index in window')
    axarr[2].plot(timestamps,ED)
    axarr[2].set_ylabel('ED between clusters')
    plt.show()
