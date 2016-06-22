import numpy as np
from levinson import LEVINSON


def LPC(specMag,order=8):

# compute autoregressive model from spectral magnitude samples

# one dimension

    nbands          = len(specMag)

    # autocorrelation
    r               = np.real(np.fft.ifft(np.hstack((specMag,specMag[::-1][1:-1]))))

    # first half only
    r               = r[:nbands]

    # find LPC coef
    y,e,k           = LEVINSON(r,order=order)

    # normalize
    y               = np.hstack(([1],y))/e

    return y