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
from sklearn.covariance import LedoitWolf, OAS

def BIC(X,Y,Z,p_lambda,mode=0,shrinkage=0):
    '''
    Bayesian information criterion
    delta_BIC = BIC_M1 - BIC_M0
    M1: feature vectors X and Y modeled by two multivariate gaussians
    M0:                 Z       modeled by one              gaussian
    delta_BIC > 0: accept M1
    delta_BIC < 0: accept M0

    covariance matrix dimension size < observation size, rank(cov) <= observation size - 1
    rank deficient: http://stats.stackexchange.com/questions/60622/why-is-a-sample-covariance-matrix-singular-when-sample-size-is-less-than-number

    mode
    0: BIC
    1: BICc
    2: ABF2

    shrinkage
    0: no shrinkage
    1: Ledoit-Wolf
    2: OAS

    :param X: frame * feature
    :param Y: frame * feature
    :param Z: frame * feature
    :param p_lambda:
    :return:
    '''

    p       = X.shape[1]
    N_x     = X.shape[0]
    N_y     = Y.shape[0]
    N_z     = Z.shape[0]

    # centering data
    mean_X  = np.mean(X,axis=0)
    mean_Y  = np.mean(Y,axis=0)
    mean_Z  = np.mean(Z,axis=0)

    X       = X - mean_X
    Y       = Y - mean_Y
    Z       = Z - mean_Z

    if shrinkage == 1:
        lw = LedoitWolf(store_precision=False, assume_centered=False)
        lw.fit(X)
        sigma_x= lw.covariance_
        lw.fit(Y)
        sigma_y= lw.covariance_
        lw.fit(Z)
        sigma_z= lw.covariance_
    elif shrinkage == 2:
        oa = OAS(store_precision=False, assume_centered=False)
        oa.fit(X)
        sigma_x = oa.covariance_
        oa.fit(Y)
        sigma_y = oa.covariance_
        oa.fit(Z)
        sigma_z = oa.covariance_
    else:
        sigma_x = np.cov(X,rowvar=0)
        sigma_y = np.cov(Y,rowvar=0)
        sigma_z = np.cov(Z,rowvar=0)


    sign_z,logdet_z   = np.linalg.slogdet(sigma_z)
    sign_y,logdet_y   = np.linalg.slogdet(sigma_y)
    sign_x,logdet_x   = np.linalg.slogdet(sigma_x)

    # det_z   = sign_z*np.exp(logdet_z)
    # det_y   = sign_y*np.exp(logdet_y)
    # det_x   = sign_x*np.exp(logdet_x)

    R = (N_z/2.0) * logdet_z - \
        (N_y/2.0) * logdet_y - \
        (N_x/2.0) * logdet_x

    k_z = (p+p*(p+1)/2.0)

    if mode == 0:
        P = k_z*np.log(N_z)/2.0
    elif mode == 1:
        P = k_z*np.log(N_z)*(2.0/(N_z-2*k_z-1)-(1.0/(N_z-k_z-1)))/2.0
        P *= 10000
    elif mode == 2:
        P = P_ABF2(mean_X,mean_Y,mean_Z,sigma_x,sigma_y,sigma_z,N_x,N_y,N_z)

    # print R, P, R-p_lambda*P, logdet_z, logdet_y, logdet_x, N_z, N_y, N_x
    # if det_z <0: print det_z
    # if det_y <0: print det_y
    # if det_x <0: print det_x

    return R-p_lambda*P

def P_ABF2(mean_X,mean_Y,mean_Z,sigma_X,sigma_Y,sigma_Z,N_x,N_y,N_z):
    '''
    calculate P for ABF2
    :param mean_X:
    :param mean_Y:
    :param mean_Z:
    :param sigma_X:
    :param sigma_Y:
    :param sigma_Z:
    :return:
    '''

    p               = len(mean_X)

    mean_X          = np.matrix(mean_X)
    mean_Y          = np.matrix(mean_Y)
    mean_Z          = np.matrix(mean_Z)

    inv_sigma_X     = np.linalg.inv(sigma_X)
    inv_sigma_Y     = np.linalg.inv(sigma_Y)
    inv_sigma_Z     = np.linalg.inv(sigma_Z)

    E_x             = np.dot(np.dot(mean_X,inv_sigma_X/N_x),mean_X.T)
    E_y             = np.dot(np.dot(mean_Y,inv_sigma_Y/N_y),mean_Y.T)
    E_z             = np.dot(np.dot(mean_Z,inv_sigma_Z/N_z),mean_Z.T)

    E_1             = E_x[0,0] + E_y[0,0]
    E_0             = E_z[0,0]

    k_1             = 2*(p+p*(p+1)/2.0)
    k_0             = k_1/2

    if k_0 > E_0:
        P_0         = k_0*(1+np.log(k_0/E_0))
    else:
        P_0         = -E_0

    if k_1 > E_1:
        P_1         = k_1*(1+np.log(k_1/E_1))
    else:
        P_1         = -E_1

    return (P_1-P_0)/2.0




