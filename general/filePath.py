'''
 * Copyright (C) 2017  Music Technology Group - Universitat Pompeu Fabra
 *
 * This file is part of jingjuSingingPhraseMatching
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

from os.path import join,dirname
# from parameters import am

###################################
###### set this dataset path ######
###################################

# dataset_path = '/Users/gong/Documents/MTG document/Jingju arias/jingju_a_cappella_singing_dataset/'
# dataset_path = '/path/to/your/jingju_a_cappella_singing_dataset/'

root_path    = join(dirname(__file__),'..')

# role-type class
class_name = 'danAll' # dan role-type
# class_name = 'laosheng' # laosheng role-type

# cnn_file_name = 'mfccBands_2D_all_optim'

cnn_file_name = 'hsmm_am_timbral'

primarySchool_dataset_root_path     = '/Users/gong/Documents/MTG document/Jingju arias/primary_school_recording'

primarySchool_wav_path = join(primarySchool_dataset_root_path, 'wav')
primarySchool_textgrid_path = join(primarySchool_dataset_root_path, 'textgrid')

cnnModel_name = 'am_cnn'

eval_results_path = join(root_path, 'eval', 'results', cnnModel_name)

primarySchool_results_path = join(root_path, 'eval', 'joint', 'results')

# path for keras cnn models
# kerasScaler_path        = join(root_path, 'cnnModels', class_name,
#                                'scaler_' + class_name + '_phonemeSeg_mfccBands2D.pkl')
# kerasModels_path        = join(root_path, 'cnnModels', class_name,
#                                'keras.cnn_' + cnn_file_name + '.h5')

kerasScaler_path        = join(root_path, 'cnnModels', 'scaler_'+cnn_file_name+'.pkl')
kerasModels_path        = join(root_path, 'cnnModels', cnn_file_name + '.h5')


# nacta 2017 dataset part 2
nacta2017_dataset_root_path     = '/Users/gong/Documents/MTG document/Jingju arias/jingju_a_cappella_singing_dataset_extended_nacta2017'

# nacta dataset part 1
nacta_dataset_root_path     = '/Users/gong/Documents/MTG document/Jingju arias/jingju_a_cappella_singing_dataset'

nacta2017_wav_path = join(nacta2017_dataset_root_path, 'wav')
nacta2017_textgrid_path = join(nacta2017_dataset_root_path, 'textgridDetails')

nacta_wav_path = join(nacta_dataset_root_path, 'wav')
nacta_textgrid_path = join(nacta_dataset_root_path, 'textgrid')

# acoustic model training dataset path
data_path_am = '/Users/gong/Documents/MTG document/dataset/acousticModels'