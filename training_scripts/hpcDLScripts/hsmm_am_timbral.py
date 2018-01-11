import sys, os
os.environ["CUDA_VISIBLE_DEVICES"] = str(0)

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models import train_model_validation

if __name__ == '__main__':

    nlen = 15
    input_dim = (80, nlen)

    filename_train_validation_set = '/scratch/acousticModelsTraining/feature_hsmm_am.h5'
    filename_labels_train_validation_set = '/homedtic/rgong/acousticModelsTraining/dataset/labels_hsmm_am.pickle.gz'

    file_path_model = '/homedtic/rgong/acousticModelsTraining/out/hsmm_am_timbral.h5'
    file_path_log = '/homedtic/rgong/acousticModelsTraining/out/log/hsmm_am_timbral.csv'

    # filename_train_validation_set = '/Users/gong/Documents/MTG document/dataset/acousticModels/feature_hsmm_am.h5'
    # filename_labels_train_validation_set = '/Users/gong/Documents/MTG document/dataset/acousticModels/labels_hsmm_am.pickle.gz'
    #
    # file_path_model = '../../temp/hsmm_am_timbral.h5'
    # file_path_log = '../../temp/hsmm_am_timbral.csv'

    train_model_validation(filename_train_validation_set,
                            filename_labels_train_validation_set,
                            filter_density=4,
                            dropout=0.32,
                            input_shape=input_dim,
                            file_path_model = file_path_model,
                            filename_log = file_path_log,
                            channel=1)