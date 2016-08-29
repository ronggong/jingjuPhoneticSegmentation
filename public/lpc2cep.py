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

