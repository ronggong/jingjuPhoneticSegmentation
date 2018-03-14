from os.path import join, dirname

root_path = join(dirname(__file__), '..')

path_jingju_dataset = '/media/gong/ec990efa-9ee0-4693-984b-29372dcea0d1/Data/RongGong'

# primary school dataset path
primarySchool_dataset_root_path = join(path_jingju_dataset, 'primary_school_recording')

# nacta dataset part 1
nacta_dataset_root_path = join(path_jingju_dataset, 'jingju_a_cappella_singing_dataset')

# nacta 2017 dataset part 2
nacta2017_dataset_root_path = join(path_jingju_dataset, 'jingju_a_cappella_singing_dataset_extended_nacta2017')

plot_data_path = join(root_path, 'plot_data')