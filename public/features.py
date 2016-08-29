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

    elif feature_select == 'dmfcc':
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