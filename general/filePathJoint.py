from os.path import join, dirname

root_path = join(dirname(__file__), '..')

path_jingju_dataset = '/media/gong/ec990efa-9ee0-4693-984b-29372dcea0d1/Data/RongGong'

# primary school dataset path
primarySchool_dataset_root_path = join(path_jingju_dataset, 'primary_school_recording')

# nacta dataset part 1
nacta_dataset_root_path = join(path_jingju_dataset, 'jingju_a_cappella_singing_dataset')

# nacta 2017 dataset part 2
nacta2017_dataset_root_path = join(path_jingju_dataset, 'jingju_a_cappella_singing_dataset_extended_nacta2017')

# path where to store the files for training the model
# change this to your folder
training_data_path = '/Users/gong/Documents/MTG document/dataset/syllableSeg/'

primarySchool_wav_path = join(primarySchool_dataset_root_path, 'wav')
primarySchool_textgrid_path = join(primarySchool_dataset_root_path, 'textgrid')

joint_cnn_model_path = join(root_path, 'cnnModels', 'joint')

scaler_joint_model_path = join(joint_cnn_model_path, 'scaler_joint.pkl')

# results path for outputing the metrics
cnnModel_name = 'jan_joint'
eval_results_path = join(root_path, 'eval', 'results', 'joint', cnnModel_name)

# results path to save the .pkl for the evaluation
primarySchool_results_path = join(root_path, 'eval', 'joint', 'results')

full_path_keras_cnn_0 = join(joint_cnn_model_path, cnnModel_name)

nacta_wav_path = join(nacta_dataset_root_path, 'wav')
nacta_textgrid_path = join(nacta_dataset_root_path, 'textgrid')

nacta2017_wav_path = join(nacta2017_dataset_root_path, 'wav')
nacta2017_textgrid_path = join(nacta2017_dataset_root_path, 'textgridDetails')