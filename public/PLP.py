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




