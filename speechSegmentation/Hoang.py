# -*- coding: utf-8 -*-

import essentia.standard as ess
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
    :return:
    '''

    fs              = varin['fs']
    framesize       = varin['framesize']
    hopsize         = varin['hopsize']

    try:
        max_band        = varin['max_band']
    except:
        max_band        = 8
    # l = [2,4,6,8,10]
    try:
        l               = varin['l']
    except:
        l               = 2
    try:
        h0              = varin['h0']
    except:
        h0              = 0.6
    # h1 = [0.6,0.8,1.0]
    try:
        h1              = varin['h1']
    except:
        h1              = 0.08
    # h2 = [0.5,0.6,0.7,0.8,0.9,1.0]
    try:
        h2              = varin['h2']
    except:
        h2              = 0.0725
    th_phone        = varin['th_phone']
    q               = varin['q']
    k               = varin['k']
    delta           = varin['delta']
    step2           = varin['step2']
    plot            = varin['plot']
    energy          = varin['energy']

    M               = 3                        # legendre order
    PEAK            = ess.PeakDetection(interpolate=False,maxPeaks=99999)
    T               = feature.shape[0]         # time
    D               = feature.shape[1]         # feature dimension
    legCoefs        = np.zeros(shape=(T,M+1))
    G               = np.zeros(shape=(max_band,T))
    m_a0            = np.zeros(shape=(max_band,T))
    th_e            = np.mean(energy)/100

    for ii in range(max_band):
        # triangular filtering
        feature[:,ii]      = triangularFilter(feature[:,ii])
        # normalizing
        feature[:,ii]      = (feature[:,ii]-min(feature[:,ii]))/(max(feature[:,ii])-min(feature[:,ii]))
        # segmentation, fitting legendre polynomial
        for jj in range(l,T-l-1):
            f                           = np.zeros(shape=(T,1))
            seg                         = feature[jj-l:jj+l+1,ii]
            seg                         = (seg-min(seg))/(max(seg)-min(seg))
            seg                         = (seg-0.5)*2
            f[jj-l:jj+l+1,0]            = seg
            # legendre coef
            coef                        = np.polynomial.legendre.legfit(np.arange(T).transpose(),f,M)
            legCoefs[jj,:]              = coef.transpose()

        legCoefs        = np.array(legCoefs)
        a0              = legCoefs[:,0]
        m_a0[ii,:]      = a0

        if max(legCoefs[:,1])!=min(legCoefs[:,1]) and max(legCoefs[:,2])!=min(legCoefs[:,2]):
            a1              = (legCoefs[:,1]-min(legCoefs[:,1]))/(max(legCoefs[:,1])-min(legCoefs[:,1]))
            a2              = (legCoefs[:,2]-min(legCoefs[:,2]))/(max(legCoefs[:,2])-min(legCoefs[:,2]))
        else:
            continue

        # detect peaks of a1
        p_a1,a_a1       = PEAK(np.array(a1,dtype=np.float32))
        p_a1            = np.array(np.round(p_a1*(T-1)),dtype=np.int)

        # remove peaks which is silence
        for jj,ii_p in reversed(list(enumerate(p_a1))):
            if energy[ii_p] < th_e or a_a1[jj] < h1:
                p_a1    = np.delete(p_a1,jj)
                a_a1    = np.delete(a_a1,jj)

        # detect valleys of a2
        p_a2_v,a_a2_v   = PEAK(np.array(1-a2,dtype=np.float32))
        p_a2_v          = np.array(np.round(p_a2_v*(T-1)),dtype=np.int)

        # detect peaks of a2
        p_a2_p,a_a2_p   = PEAK(np.array(a2,dtype=np.float32))
        p_a2_p          = np.array(np.round(p_a2_p*(T-1)),dtype=np.int)

        # BE change
        if len(p_a1) and len(p_a2_p):
            be_change_ii        = BE_change(p_a1,p_a2_p,p_a2_v,feature[:,ii],h2=h2,frame_interval=5)
            G[ii,be_change_ii]  = 1

    # equation 5
    g               = np.sum(G,axis=0)

    phoneBoundary   = phoneBoundaryStep1(g)

    if step2:
        d               = phoneBoundaryStep2(phoneBoundary,m_a0,hopsize,fs,th_phone,q,k,delta)

        p_d,a_d         = PEAK(np.array(d[:,0],dtype=np.float32))

        for ii,ii_a_d in enumerate(a_d):
            if ii_a_d > h0:
                phoneBoundary.append(p_d[ii])

    phoneBoundary_time = np.array(phoneBoundary)*(hopsize/float(fs))

    if plot:
        plt.figure()
        plt.plot(a1)
        plt.stem(p_a1,a_a1)

        plt.show()

        mX              = spec
        mX              = np.transpose(mX)
        maxplotfreq     = 16001.0
        eps             = np.finfo(np.float).eps
        mXPlot          = mX[:int(N*(maxplotfreq/fs))+1,:]
        binFreqs        = np.arange(mXPlot.shape[0])*fs/float(N)
        timestamps      = np.arange(mXPlot.shape[1]) * (hopsize/float(fs))

        plt.figure()
        plt.pcolormesh(timestamps, binFreqs, 20*np.log10(mXPlot+eps))
        plt.title('Hoang')
        plt.xlabel('time(s)')
        plt.ylabel('freq')
        for ii in range(len(phoneBoundary)):
            phoneBoundary_time = phoneBoundary[ii]*(hopsize/float(fs))
            plt.axvline(phoneBoundary_time)
        plt.show()

    return phoneBoundary_time

def triangularFilter(s_apos):
    '''
    triangular filter to reduce the effect of noise
    :param s:
    :return:
    '''

    N       = len(s_apos)
    s       = np.zeros(shape=(N,1))
    s[0]    = s_apos[0]
    s[-1]   = s_apos[-1]
    for ii in range(1,N-1):
        s[ii]       = 0.25*s_apos[ii-1]+0.5*s_apos[ii]+0.25*s_apos[ii+1]

    return s.transpose()

def BE_change(p_a1,p_a2_p,p_a2_v,s,h2,frame_interval):
    '''
    algorithm of BE change
    :param a1:
    :param a0:
    :param h1:
    :param h2:
    :param frame_interval:
    :return:
    '''

    be_change_list  = []
    be_change       = -1
    for pp_a1 in p_a1:
        be_change               = pp_a1
        min_interval            = np.inf
        pp_a2_v_found           = -1
        for pp_a2_v in p_a2_v:
            if abs(pp_a2_v-pp_a1) < frame_interval/2.0 and abs(pp_a2_v-pp_a1) < min_interval:
                min_interval    = abs(pp_a2_v-pp_a1)
                pp_a2_v_found   = pp_a2_v
        if pp_a2_v_found > 0:
            pp_a2_p_left        = 0
            pp_a2_p_right       = p_a2_p[-1]
            for ii_pp_a2_p,pp_a2_p in enumerate(p_a2_p):
                if pp_a2_p - pp_a2_v_found < 0:
                    pp_a2_p_left    = pp_a2_p
                if pp_a2_p - pp_a2_v_found > 0:
                    pp_a2_p_right   = pp_a2_p
                    break

            if abs(s[pp_a2_p_left]-s[pp_a2_p_right]) > h2:
                be_change = pp_a2_v_found
            else:
                be_change = -1
        else:
            be_change = -1

        if be_change >= 0:
            be_change_list.append(be_change)

    return be_change_list

def phoneBoundaryStep1(g):
    '''
    detecting phone boundary step 1
    :param g:
    :return:
    '''

    phoneBoundary   = []
    N1,N2           = -1,-1
    for ii in range(len(g)-1):
        if g[ii] == 0 and g[ii+1] != 0:
            N1      = ii
            N2      = -1
        elif g[ii] != 0 and g[ii+1] == 0:
            N2      = ii+1
            if N1 >=0 and N2 >= 0:
                # equation 6
                p   = sum(g[N1:N2])
                if p <= 2:
                    g[N1:N2] = 0
                else:
                    # equation 7
                    g[N1:N2] /= p
                    # equation 8
                    sum_g    = 0
                    for m in range(N1,N2+1):
                        sum_g += g[m]
                        if sum_g >= 0.5:
                            phoneBoundary.append(m)
                            break
            N1      = -1

    return phoneBoundary

def phoneBoundaryStep2(phoneBoundary,m_a0,hopsize,fs,th_phone,q,k,delta):
    '''

    :param phoneBoundary:
    :param hopsize:
    :param th_phone:
    :param q:
    :param k:
    :param delta:
    :param m_a0:
    :return:
    '''

    d       = np.zeros(shape=(m_a0.shape[1],1))
    for ii in range(len(phoneBoundary)-1):
        if (phoneBoundary[ii+1]-phoneBoundary[ii])*hopsize/float(fs) < th_phone:
            N1      = phoneBoundary[ii]+q+k
            N2      = phoneBoundary[ii+1]-q-k
            if N2 > N1:
                for n in range(phoneBoundary[ii]+q+k,phoneBoundary[ii+1]-q-k):
                    # equation 11
                    x       = np.sum(m_a0[:,ii-q-k:ii-q],axis=1)/k
                    y       = np.sum(m_a0[:,ii+q+1:ii+q+k+1],axis=1)/k
                    # equation 9
                    f       = np.exp(-np.linalg.norm(x-y)**2/(2*delta**2))
                    d[n]    = 1-f

    return d