import sys, os

# os.environ["CUDA_VISIBLE_DEVICES"] = str(0)

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import shutil
from models_joint import train_model_validation


if __name__ == '__main__':

    nlen = 15
    input_dim = (80, nlen)

    filename_train_validation_set = '/homedtic/rgong/cnnSyllableSeg/syllableSeg/feature_all_joint_subset.h5'
    filename_labels_syllable_train_validation_set = '/homedtic/rgong/cnnSyllableSeg/syllableSeg/labels_joint_syllable_subset.pickle.gz'
    filename_labels_phoneme_train_validation_set = '/homedtic/rgong/cnnSyllableSeg/syllableSeg/labels_joint_phoneme_subset.pickle.gz'
    filename_sample_weights_syllable = '/homedtic/rgong/cnnSyllableSeg/syllableSeg/sample_weights_joint_syllable_subset.pickle.gz'
    filename_sample_weights_phoneme = '/homedtic/rgong/cnnSyllableSeg/syllableSeg/sample_weights_joint_phoneme_subset.pickle.gz'

    # filename_train_validation_set = '/Users/gong/Documents/MTG document/dataset/syllableSeg/feature_all_joint_subset.h5'
    # filename_labels_syllable_train_validation_set = '../../training_data/labels_joint_syllable_subset.pickle.gz'
    # filename_labels_phoneme_train_validation_set = '../../training_data/labels_joint_phoneme_subset.pickle.gz'
    # filename_sample_weights_syllable = '../../training_data/sample_weights_joint_syllable_subset.pickle.gz'
    # filename_sample_weights_phoneme = '../../training_data/sample_weights_joint_phoneme_subset.pickle.gz'

    # import cPickle
    # import gzip
    # labels_syllable = cPickle.load(gzip.open(filename_labels_syllable_train_validation_set, 'rb'))
    # labels_pho = cPickle.load(gzip.open(filename_labels_phoneme_train_validation_set, 'rb'))
    #
    # print(len(labels_syllable[labels_syllable==1])/float(len(labels_syllable)))
    # print(len(labels_pho[labels_pho==1])/float(len(labels_pho)))

    # loss weight 3.35:1

    tmp_train_validation_set = '/tmp/phoneSeg_syllable_val_loss_weighted'
    if not os.path.isdir(tmp_train_validation_set):
        os.mkdir(tmp_train_validation_set)

    filename_temp_train_validation_set = os.path.join(tmp_train_validation_set, 'feature_all_joint_subset.h5')

    shutil.copy2(filename_train_validation_set, filename_temp_train_validation_set)

    for running_time in range(1, 5):
        # train the final model
        file_path_model = '/homedtic/rgong/cnnSyllableSeg/out/jan_joint_subset_syllable_val_loss_weighted'+str(running_time)+'.h5'
        file_path_log = '/homedtic/rgong/cnnSyllableSeg/out/log/jan_joint_subset_syllable_val_loss_weighted'+str(running_time)+'.csv'

        # file_path_model = '../../temp/jan_joint.h5'
        # file_path_log = '../../temp/jan_joint.csv'

        train_model_validation(filename_temp_train_validation_set,
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

    shutil.rmtree(tmp_train_validation_set)