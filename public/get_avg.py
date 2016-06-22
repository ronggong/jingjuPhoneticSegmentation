import numpy as np

def get_avg( m , v_span, h_span):
    # This function produces a smoothed version of cochleagram
    [nr,nc] = m.shape
    out = np.zeros((nr,nc))
    fil_size = (2*v_span+1)*(2*h_span+1)

    for i in range(nr):
        row_begin = 0
        row_end = nr-1
        col_begin =0
        col_end = nc-1
        if (i - v_span)>=0:
            row_begin = i - v_span
        if (i + v_span)<=nr-1:
            row_end = i + v_span

        for j in range(nc):
            if (j - h_span)>=0:
                col_begin = j - h_span

            if (j + h_span)<=nc-1:
                col_end = j + h_span

            tmp = m[row_begin:row_end,col_begin:col_end]
            out[i,j] = np.sum(np.sum(tmp))/fil_size

    return out
