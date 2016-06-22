import numpy as np
from scipy.signal import lfilter,lfiltic


def rastafilt(x):

    """Apply RASTA filtering to the input signal.

    :param x: the input audio signal to filter.
        cols of x = critical bands, rows of x = frame
        same for y but after filtering
        default filter is single pole at 0.94
    """
    x = x.T

    # rasta filter
    numer = np.array(range(-2,3),dtype=np.float)
    numer = -numer / np.sum(numer*numer)
    denom = [1,-0.94]

    # numer = np.arange(.2, -.3, -.1)
    # denom = np.array([1, -0.98])

    # Initialize the state.  This avoids a big spike at the beginning
    # resulting from the dc offset level in each band.
    # (this is effectively what rasta/rasta_filt.c does).
    # Because Matlab uses a DF2Trans implementation, we have to
    # specify the FIR part to get the state right (but not the IIR part)
    y = np.zeros(x.shape)
    zf = np.zeros((x.shape[0], 4))
    for i in range(y.shape[0]):
        y[i, :4], zf[i, :4] = lfilter(numer, 1, x[i, :4], axis=-1, zi=[0, 0, 0, 0])

    # .. but don't keep any of these values, just output zero at the beginning
    y = np.zeros(x.shape)

    # Apply the full filter to the rest of the signal, append it
    for i in range(y.shape[0]):
        y[i, 4:] = lfilter(numer, denom, x[i, 4:], axis=-1, zi=zf[i, :])[0]

    return y.T

