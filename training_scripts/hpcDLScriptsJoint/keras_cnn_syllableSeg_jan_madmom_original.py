import sys, os

os.environ["CUDA_VISIBLE_DEVICES"] = str(0)

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models_joint import train_model_validation


if __name__ == '__main__':

    nlen = 15
    input_dim = (80, nlen)

    # filename_train_validation_set = '/scratch/rgongcnnSyllableSeg_jan_joint/syllableSeg/feature_all_joint.h5'
    # filename_labels_syllable_train_validation_set = '/homedtic/rgong/cnnSyllableSeg/syllableSeg/labels_joint_syllable.pickle.gz'
    # filename_labels_phoneme_train_validation_set = '/homedtic/rgong/cnnSyllableSeg/syllableSeg/labels_joint_phoneme.pickle.gz'
    # filename_sample_weights_syllable = '/homedtic/rgong/cnnSyllableSeg/syllableSeg/sample_weights_joint_syllable.pickle.gz'
    # filename_sample_weights_phoneme = '/homedtic/rgong/cnnSyllableSeg/syllableSeg/sample_weights_joint_phoneme.pickle.gz'


    filename_train_validation_set = '/Users/gong/Documents/MTG document/dataset/syllableSeg/feature_all_joint.h5'
    filename_labels_syllable_train_validation_set = '../../training_data/labels_joint_syllable.pickle.gz'
    filename_labels_phoneme_train_validation_set = '../../training_data/labels_joint_phoneme.pickle.gz'
    filename_sample_weights_syllable = '../../training_data/sample_weights_joint_syllable.pickle.gz'
    filename_sample_weights_phoneme = '../../training_data/sample_weights_joint_phoneme.pickle.gz'

    for running_time in range(5):
        # train the final model
        # file_path_model = '/homedtic/rgong/cnnSyllableSeg/out/jan_joint'+str(running_time)+'.h5'
        # file_path_log = '/homedtic/rgong/cnnSyllableSeg/out/log/jan_joint'+str(running_time)+'.csv'

        file_path_model = '../../temp/jan_joint.h5'
        file_path_log = '../../temp/jan_joint.csv'

        train_model_validation(filename_train_validation_set,
                               filename_labels_syllable_train_validation_set,
                               filename_labels_phoneme_train_validation_set,
                               filename_sample_weights_syllable,
                               filename_sample_weights_phoneme,
                                filter_density=1,
                                dropout=0.5,
                                input_shape=input_dim,
                                file_path_model = file_path_model,
                                filename_log = file_path_log,
                                channel=1)