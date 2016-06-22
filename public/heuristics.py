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