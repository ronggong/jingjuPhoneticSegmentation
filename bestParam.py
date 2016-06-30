import operator, os
import numpy as np

def sort_table(table, col=0):
    return sorted(table, key=operator.itemgetter(col))

OS = 4

# filenames   = ['win_mfcc2', 'win_mfccBands', 'win_gfcc', 'win_plpcc', 'win_plp', 'win_rasta-plpcc', 'win_rasta-plp',
#              'win_bark', 'ave_bark', 'hoa_mfccBands']

filenames   = ['hoa_mfccBands2']

for fn in filenames:

    csv_path    = '/Users/gong/Documents/MTG document/Jingju arias/phonemeSeg'

    fn_load     = os.path.join(csv_path, fn+'.csv')

    data        = np.loadtxt(fn_load, delimiter=',')

    fn_save     = os.path.join(csv_path, fn+'_sorted.csv')

    sorted_table = sort_table(data,col=OS)

    np.savetxt(fn_save, sorted_table, delimiter=',',fmt='%f')

