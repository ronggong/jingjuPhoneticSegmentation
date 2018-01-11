from os.path import join, dirname
# from parameters import *

primarySchool_dataset_root_path     = '/Users/gong/Documents/MTG document/Jingju arias/primary_school_recording'

primarySchool_wav_path = join(primarySchool_dataset_root_path, 'wav')
primarySchool_textgrid_path = join(primarySchool_dataset_root_path, 'textgrid')

root_path       = join(dirname(__file__),'..')

joint_cnn_model_path = join(root_path, 'cnnModels', 'joint')

scaler_joint_model_path = join(joint_cnn_model_path,
                                    'scaler_joint.pkl')

cnnModel_name = 'jan_joint'

eval_results_path = join(root_path, 'eval', 'results', cnnModel_name)

primarySchool_results_path = join(root_path, 'eval', 'joint', 'results')

full_path_keras_cnn_0 = join(joint_cnn_model_path, cnnModel_name)

# where training features are saved
feature_data_path = '/Users/gong/Documents/MTG document/dataset/syllableSeg/'

# nacta 2017 dataset part 2
nacta2017_dataset_root_path     = '/Users/gong/Documents/MTG document/Jingju arias/jingju_a_cappella_singing_dataset_extended_nacta2017'

# nacta dataset part 1
nacta_dataset_root_path     = '/Users/gong/Documents/MTG document/Jingju arias/jingju_a_cappella_singing_dataset'

nacta2017_wav_path = join(nacta2017_dataset_root_path, 'wav')
nacta2017_textgrid_path = join(nacta2017_dataset_root_path, 'textgridDetails')

nacta_wav_path = join(nacta_dataset_root_path, 'wav')
nacta_textgrid_path = join(nacta_dataset_root_path, 'textgrid')