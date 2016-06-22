import numpy as np


def lpc2cep(lpcCoef,numcep=13):
    '''
    convert lpc to cepstra
    :param lpcCoef:
    :param numcep: number of cepstra
    :return:
    '''

    if numcep > len(lpcCoef):
        numcep  = len(lpcCoef)

    c       = np.zeros((numcep,))

    # First cep is log(Error) from Durbin
    c[0]    = -np.log(lpcCoef[0])

    # Renormalize lpc A coeffs
    lpcCoef = lpcCoef / lpcCoef[0]

    for n in range(1,numcep):
        sum = 0
        for m in range(1,n+1):
            sum = sum + (n - m) * lpcCoef[m] * c[n - m + 1]
            # print n-m+1
            # print lpcCoef[m], sum, n-1, c[n-m+1]

        c[n] = -(lpcCoef[n] + sum / n)

    return c

