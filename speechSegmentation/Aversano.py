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

# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt


def mainFunction(feature,spec,varin):

    '''
    main procedure of algorithm
    :param feature: observation * features
    :param spec: spectrogram
    :param fs:
    :param framesize:
    :param hopsize:
    :return: detected boundary in sec
    '''

    fs              = varin['fs']
    framesize       = varin['framesize']
    hopsize         = varin['hopsize']
    a               = varin['a']
    b               = varin['b']
    c               = varin['c']
    plot            = varin['plot']

    N               = 2 * framesize                     # padding 1 time framesize

    if feature.shape[0] < 2*a:
        return np.array([])

    print 'calculating jump function ... ...'

    S               = np.zeros(shape=(feature.shape[0]-2*a,12))
    acc             = np.zeros(shape=(S.shape[0],1))

    for ii in range(0,12):
        J                   = jump(feature[:,ii],a)
        J                   = (J-min(J))/(max(J)-min(J))
        index_peak          = peakDetection(J,b)
        S[index_peak,ii]    = 1

    print 'calculating Fitting function ... ...'
    for n in range(0,S.shape[0]-c):
        nwin                = FIT(S,n,c)
        acc[nwin]           += 1

    acc                     = np.vstack((np.zeros(shape=(a,1)),acc,(np.zeros(shape=(a,1)))))

    index_acc_peak          = peakDetection(acc,3)

    index_acc_time          = np.array(index_acc_peak)*(hopsize/float(fs))

    if plot:
        plt.figure()
        plt.plot(acc)
        plt.show()

        mX                  = spec
        mX                  = np.transpose(mX)
        maxplotfreq         = 16001.0
        eps                 = np.finfo(np.float).eps
        mXPlot              = mX[:int(N*(maxplotfreq/fs))+1,:]
        binFreqs            = np.arange(mXPlot.shape[0])*fs/float(N)
        timestamps          = np.arange(mXPlot.shape[1]) * (hopsize/float(fs))

        plt.figure()
        plt.pcolormesh(timestamps, binFreqs, 20*np.log10(mXPlot+eps))
        plt.title('Aversano')
        plt.xlabel('time(s)')
        plt.ylabel('freq')

        for ii in range(0,len(index_acc_peak)):
            index_acc_time = 8 * hopsize/float(fs) + index_acc_peak[ii]*(hopsize/float(fs))
            plt.axvline(index_acc_time)
        plt.show()

    return index_acc_time

def jump(x,a):
    '''
    jump function
    :param x:       1*T, T frames feature
    :param a:       half sliding window
    :return:
    '''

    N   = len(x)
    J   = [0]*(N-2*a)

    for n in range(a,N-a):
        p       = sum(x[n-a:n]/a)
        q       = sum(x[n+1:n+a+1]/a)

        J[n-a]    = abs(p-q)

    return J

def peakDetection(J,b):
    '''
    detect peak
    :param J:           1*T T frames jump function
    :param b:           threshold
    :return:
    '''

    N           = len(J)
    index_peak  = []

    for ii in range(1,N):

        l = ii
        while l > 0:
            if J[l] <= J[l-1]:
                break
            l -= 1

        m = ii
        while m < N-1:
            if J[m] < J[m+1]:
                break
            m += 1

        if l != ii and m != ii:
            h = min(J[ii]-J[l], J[ii]-J[m])
            if h > b:
                index_peak.append(ii)

    return index_peak

def FIT(S,n,c):
    '''
    fitting procedure
    :param S:       N*T jump function peak matrix
    :param c:       window size
    :return:
    '''

    f   = np.zeros(shape=(c,1))
    for ii in range(n,n+c):
        for m in range(n,n+c):
            for jj in range(0,12):
                f[ii-n] += S[m,jj]*abs(ii-m)
    nwin = np.argmin(f)

    return nwin+n

