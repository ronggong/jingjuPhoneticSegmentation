import essentia.standard as ess
import numpy as np
import matplotlib.pyplot as plt

import sys,os

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)),"../public"))

from Fdeltas    import Fdeltas
from heuristics import heuristics
from modelSelection import BIC

def mainFunction(feature,spec,varin):

    '''
    main procedure of algorithm
    :param feature: observation * features
    :param fs:
    :param framesize:
    :param hopsize:
    :return:
    '''

    fs              = varin['fs']
    framesize       = varin['framesize']
    hopsize         = varin['hopsize']
    # h2 = [0.0,0.02,0.04,0.06,0.08,0.1]
    h2              = varin['h2']
    # alpha = [0.2,0.4,0.6,0.8,1.0]
    alpha           = varin['alpha']
    # p_lambda = [0.2,0.4,0.6,0.8,1.0] mode_bic = BIC
    p_lambda        = varin['p_lambda']
    mode_bic        = varin['mode_bic']
    try:
        winmax          = varin['winmax']
    except:
        winmax          = 0.35
    plot            = varin['plot']

    if varin['feature_select'] == 'mfcc':
        mfcc        = feature
    else:
        mfcc        = varin['mfcc']

    # # spectrogram init
    # winAnalysis     = 'hann'
    # N               = 2 * framesize                     # padding 1 time framesize
    # SPECTRUM        = ess.Spectrum(size=N)
    # WINDOW          = ess.Windowing(type=winAnalysis, zeroPadding=N-framesize)
    # highFrequencyBound = fs/2 if fs/2<11000 else 11000
    # MFCC            = ess.MFCC(sampleRate=fs,highFrequencyBound=highFrequencyBound,inputSize=framesize+1)
    # GFCC            = ess.GFCC(sampleRate=fs,highFrequencyBound=highFrequencyBound)
    # mfcc            = []
    # gfcc            = []
    # mX              = []
    #
    # print 'calculating MFCC ... ...'
    #
    # for frame in ess.FrameGenerator(audio, frameSize=framesize, hopSize=hopsize):
    #
    #     frame           = WINDOW(frame)
    #     mXFrame         = SPECTRUM(frame)
    #     mX.append(mXFrame)
    #     bands,mfccFrame = MFCC(mXFrame)
    #     mfccFrame       = mfccFrame[1:]
    #     bands,gfccFrame = GFCC(mXFrame)
    #     gfccFrame       = gfccFrame[1:]
    #
    #     mfcc.append(mfccFrame)
    #     gfcc.append(gfccFrame)
    #
    # mX              = np.array(mX)
    # mfcc            = np.array(mfcc)
    # gfcc            = np.array(gfcc)
    T               = mfcc.shape[0]                         # time
    D               = mfcc.shape[1]                         # feature dimension
    winmax_frame    = np.int(np.round(winmax*fs/hopsize))
    PEAK            = ess.PeakDetection(interpolate=False,maxPeaks=99999)

    # plpcc,plp       = PLP(mX,modelorder=12,rasta=False)
    # all_MRCG,d_MRCG,dd_MRCG = MRCG(audio,fs=fs)

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
            p_0             = p_SVF[ii] - winmax_frame  if p_1-p_0 > winmax_frame   else p_0

            # try to fix the small sample problem
            # p_0             = p_SVF[ii] - D             if p_1-p_0 < D              else p_0
            # p_2             = p_SVF[ii] + D             if p_2-p_1 < D              else p_2
            #
            # if p_0 < 0 or p_2 > p_SVF[-1]:
            #     print p_0, p_1, p_2
            #     index2Delete.append(ii)
            #     jj              = 1
            #     ii              += 1
            #     continue

            delta_BIC   = BIC(feature[p_0:p_1,:],feature[p_1:p_2,:],feature[p_0:p_2,:],p_lambda,mode=mode_bic,shrinkage=2)

            if  delta_BIC > 0:
                jj              = 1

            else:
                jj              += 1
                index2Delete.append(ii)
            ii              += 1

            if ii >= len(p_SVF)-1: break

            # print delta_BIC, p_0, p_1, p_2,

    p_BIC           = np.delete(p_SVF,index2Delete)
    a_BIC           = np.delete(a_SVF,index2Delete)

    timestamps_p_BIC= p_BIC * (hopsize/float(fs))

    # plot
    if plot:
        mX              = spec
        mX              = np.transpose(mX)
        maxplotfreq     = 6001.0
        eps             = np.finfo(np.float).eps
        mXPlot          = mX[:int(N*(maxplotfreq/fs))+1,:]
        binFreqs        = np.arange(mXPlot.shape[0])*fs/float(N)
        timestamps_spec = np.arange(mXPlot.shape[1]) * (hopsize/float(fs))

        timestamps_audio= np.arange(len(audio))/float(fs)
        timestamps      = np.arange(T) * (hopsize/float(fs))
        timestamps_p_SVF= p_SVF * (hopsize/float(fs))
        timestamps_p_BIC= p_BIC * (hopsize/float(fs))

        f, axarr = plt.subplots(3, sharex=True)
        axarr[0].plot(timestamps_audio,audio)

        axarr[1].plot(timestamps, SVF)
        axarr[1].stem(timestamps_p_SVF, a_SVF)
        axarr[1].set_ylabel('SVF')
        axarr[1].set_title('boundary heuristics')

        axarr[2].plot(timestamps, SVF)
        axarr[2].stem(timestamps_p_BIC, a_BIC)
        axarr[2].set_title('boundary DISTBIC')

        axarr[3].pcolormesh(timestamps_spec, binFreqs, 20*np.log10(mXPlot+eps))
        axarr[3].set_title('spectrogram')
        for ii in range(0,len(timestamps_p_BIC)):
            axarr[3].axvline(timestamps_p_BIC[ii])

        # plpcc           = np.transpose(plp[:,1:])
        # binFreqs        = np.arange(plpcc.shape[0])
        # axarr[4].pcolormesh(timestamps_spec, binFreqs, plpcc)
        #
        # plt.show()
        #
        # all_MRCG        = np.transpose(all_MRCG)
        # binFreqs        = np.arange(all_MRCG.shape[0])
        # timestamps_spec = np.arange(all_MRCG.shape[1]) * (hopsize/float(fs))
        # plt.figure()
        # plt.pcolormesh(timestamps_spec, binFreqs, all_MRCG)
        # plt.show()

    return timestamps_p_BIC