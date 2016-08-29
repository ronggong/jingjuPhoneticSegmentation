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