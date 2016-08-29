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

import essentia.standard as ess
import numpy as np
import matplotlib.pyplot as plt
from equalLoudness  import equalLoudness
from LPC            import LPC
from lpc2cep        import lpc2cep
from rastafilt      import rastafilt

def PLP(spec, fs=44100, modelorder=12, rasta=False):
    '''

    :param spec: frame * frequency
    :param fs:
    :param modelorder:
    :param rasta
    :return: outMat frame * plp bin, 0 bin is LPC error
    '''

    nframe,nfreq    = spec.shape

    # 27 band center frequencies
    bandCenters     = [25, 75, 125, 175, 250, 350, 450, 570, 700, 840,
                       1000, 1170, 1370, 1600, 1850, 2150, 2500, 2900, 3400, 4000,
                       4800, 5800, 7000, 8500, 10500, 13500, 18000]

    nbands          = len(bandCenters)

    BARKBANDS       = ess.BarkBands(sampleRate=fs,numberBands=nbands)

    # init of matrices
    barkenergyMat   = np.zeros((nframe,nbands))
    plpMat          = np.zeros((nframe,modelorder+1))
    plpccMat        = np.zeros((nframe,modelorder+1))

    # critical band energy
    for ii in range(nframe):
        barkenergy          = BARKBANDS(spec[ii,:])
        barkenergyMat[ii,:] = barkenergy

    # RASTA filtering
    if rasta:
        # log domain
        nl_barkenergyMat    = np.log(barkenergyMat+np.finfo(np.float).eps)

        nl_barkenergyMat    = rastafilt(nl_barkenergyMat)

        barkenergyMat       = np.exp(nl_barkenergyMat)

    for ii in range(nframe):

        # equal loudness curve
        eql             = equalLoudness(bandCenters)

        barkenergyMat[ii,:] = barkenergyMat[ii,:] * eql

        # intensity loudness power law
        barkenergyMat[ii,:] = barkenergyMat[ii,:]**0.33

        # LPC
        plp             = LPC(barkenergyMat[ii,:],modelorder)

        plpcc           = lpc2cep(plp,modelorder+1)

        plpMat[ii,:]    = plp
        plpccMat[ii,:]  = plpcc

    return plpccMat,plpMat,barkenergyMat




