from os.path import join,dirname

root_path = join(dirname(__file__),'..')

# change this to your folder
path_jingju_dataset = '/media/gong/ec990efa-9ee0-4693-984b-29372dcea0d1/Data/RongGong'

# primary school dataset path
primarySchool_dataset_root_path = join(path_jingju_dataset, 'primary_school_recording')

# nacta dataset part 1
nacta_dataset_root_path = join(path_jingju_dataset, 'jingju_a_cappella_singing_dataset')

# nacta 2017 dataset part 2
nacta2017_dataset_root_path = join(path_jingju_dataset, 'jingju_a_cappella_singing_dataset_extended_nacta2017')

# acoustic model training dataset path
# change this to your folder
data_am_path = '/Users/gong/Documents/MTG document/dataset/acousticModels'

primarySchool_wav_path = join(primarySchool_dataset_root_path, 'wav')
primarySchool_textgrid_path = join(primarySchool_dataset_root_path, 'textgrid')

cnn_file_name = 'hsmm_am_timbral'
eval_results_path = join(root_path, 'eval', 'results', 'hsmm', cnn_file_name)

primarySchool_results_path = join(root_path, 'eval', 'joint', 'results')

kerasScaler_path = join(root_path, 'cnnModels', 'hsmm', 'scaler_'+cnn_file_name+'.pkl')
kerasModels_path = join(root_path, 'cnnModels', 'hsmm', cnn_file_name)

nacta2017_wav_path = join(nacta2017_dataset_root_path, 'wav')
nacta2017_textgrid_path = join(nacta2017_dataset_root_path, 'textgridDetails')

nacta_wav_path = join(nacta_dataset_root_path, 'wav')
nacta_textgrid_path = join(nacta_dataset_root_path, 'textgrid')
