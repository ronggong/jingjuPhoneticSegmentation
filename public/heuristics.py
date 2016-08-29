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

def heuristics(p_SVF,a_SVF,p_v_SVF,a_v_SVF,SVF,fs,hopsize,h2,alpha):
    '''
    heuristics eliminate candidates
    :param p_SVF:
    :param a_SVF:
    :param p_v_SVF:
    :param p_a_SVF:
    :return:
    '''

    # minimum distance between candidate peaks
    for ii in np.arange(1,len(p_SVF))[::-1]:
        if (p_SVF[ii] - p_SVF[ii-1])*hopsize/float(fs) < h2:
            p_SVF[ii-1]     = np.int(np.floor((p_SVF[ii] + p_SVF[ii-1])/2.0))
            a_SVF[ii-1]     = SVF[p_SVF[ii-1]]
            p_SVF           = np.delete(p_SVF,ii)
            a_SVF           = np.delete(a_SVF,ii)

    # amplitude threshold
    for ii in np.arange(len(p_SVF))[::-1]:
        pp_v_SVF_left   = -1
        pp_v_SVF_right  = -1
        for ii_pp_v_SVF,pp_v_SVF in enumerate(p_v_SVF):
            if pp_v_SVF - p_SVF[ii] < 0:
                pp_v_SVF_left   = pp_v_SVF
            if pp_v_SVF - p_SVF[ii] > 0:
                pp_v_SVF_right  = pp_v_SVF
                break

        # print ii_pp_v_SVF_left, ii_pp_v_SVF_right

        if pp_v_SVF_left >= 0 and pp_v_SVF_right >= 0:
            d_left      = a_SVF[ii] - SVF[pp_v_SVF_left]
            d_right     = a_SVF[ii] - SVF[pp_v_SVF_right]
            std_signal  = np.std(SVF[pp_v_SVF_left:pp_v_SVF_right+1])

            if d_left <= alpha*std_signal or d_right <= alpha*std_signal:
                p_SVF   = np.delete(p_SVF,ii)
                a_SVF   = np.delete(a_SVF,ii)

                # print ii, ' deleted'
        # left limit
        elif pp_v_SVF_left < 0 and pp_v_SVF_right >= 0:
            d_right     = a_SVF[ii] - SVF[pp_v_SVF_right]
            std_signal  = np.std(SVF[0:pp_v_SVF_right+1])

            if d_right <= alpha*std_signal:
                p_SVF   = np.delete(p_SVF,ii)
                a_SVF   = np.delete(a_SVF,ii)
                # print ii, ' deleted'

        # right limit
        elif pp_v_SVF_right < 0 and pp_v_SVF_left >= 0:
            d_left      = a_SVF[ii] - SVF[pp_v_SVF_left]
            std_signal  = np.std(SVF[pp_v_SVF_left:-1])

            if d_left <= alpha*std_signal:
                p_SVF   = np.delete(p_SVF,ii)
                a_SVF   = np.delete(a_SVF,ii)
                # print ii, ' deleted'

        else:
            p_SVF   = np.delete(p_SVF,ii)
            a_SVF   = np.delete(a_SVF,ii)
            # print ii, ' deleted'

    return (p_SVF, a_SVF, p_v_SVF, a_v_SVF)