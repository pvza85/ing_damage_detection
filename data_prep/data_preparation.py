import os
import h5py
import random
import pandas as pd
import os.path
from tflearn.data_utils import build_hdf5_image_dataset
from tflearn.data_utils import image_preloader


def data_prep(conf, method = 'seq', training_size = 0.7, clean_start = False, hdfs=True):
    '''
    create hdf5 files based on train and test selecting strategy
    and return results as train and test sets
    :param conf: parameters from congigure file
    :param method: random / sequence; at sequence get first training_size of images as training
    :param training_size: portion of images that will be used in trainig
    :param start_clean: start from zero or work based on previous work
    :return: X_train, Y_train, X_test, Y_test
    '''

    data_folder = conf['data_folder']
    if clean_start:
        try:
            os.remove(data_folder + 'train.h5')
            os.remove(data_folder + 'test.h5')
        except:
            print 'tain.h5 deleted already'
            pass
    #in the case files is available just read it and return
    if os.path.isfile(data_folder + 'train.h5')  and os.path.isfile(data_folder + 'test.h5'):
        _h5f = h5py.File(data_folder + 'train.h5', 'r')
        X_train = _h5f['X']
        Y_train = _h5f['Y']
        h5f_ = h5py.File(data_folder + 'test.h5', 'r')
        X_test = h5f_['X']
        Y_test = h5f_['Y']

        return X_train, Y_train, X_test, Y_test

    # if the file is not available, continue creating it
    test_counter = 0
    counter = 0
    train_file = data_folder + 'train.txt'
    test_file = data_folder + 'test.txt'
    with open(data_folder + 'labeling.csv') as inputFile:
        df = pd.read_csv(data_folder + conf['data_frame'])
        with open(train_file, 'w+') as trainFile:
            with open(test_file, 'w+') as testFile:
                for f in os.listdir(data_folder + conf['cropped_folder']):
                    is_train = False
                    is_test = False
                    if method == 'random':
                        # just set to all for production
                        if random.random() < training_size:
                            is_train = True
                        else:
                            is_test = True
                    else:
                        if counter < len(df) * training_size:
                            is_train = True
                        else:
                            if random.random() <= 1: #just set for production
                                is_train = True
                            is_test = True

                    label = int(df.loc[int(f.split('.')[0].split('_')[0]) - 1, 'manual_label'])
                    line = data_folder + conf['cropped_folder'] + f  + ' ' + str(label) + '\n'
                    if is_train:
                        trainFile.write(line)
                    if is_test and not '_' in f:
                        testFile.write(line)

    if hdfs:
        # Build a HDF5 dataset (only required once)
        build_hdf5_image_dataset(train_file, image_shape=(40, 40), mode='file', output_path=data_folder + 'train.h5',
                                 categorical_labels=True, normalize=True)
        try:
            build_hdf5_image_dataset(test_file, image_shape=(40, 40), mode='file', output_path=data_folder + 'test.h5',
                                     categorical_labels=True, normalize=True)
        except:
            build_hdf5_image_dataset(train_file, image_shape=(40, 40), mode='file', output_path=data_folder + 'test.h5',
                                 categorical_labels=True, normalize=True)
        # Load HDF5 dataset
        _h5f = h5py.File(data_folder + 'train.h5', 'r')
        X_train = _h5f['X']
        Y_train = _h5f['Y']

        h5f_ = h5py.File(data_folder + 'test.h5', 'r')
        X_test = h5f_['X']
        Y_test = h5f_['Y']
    else:
        X_train, Y_train = image_preloader(data_folder +'train.txt', image_shape=(40, 40), mode='file',
                                           categorical_labels=True, normalize=True)
        X_test, Y_test = image_preloader(data_folder +'test.txt', image_shape=(40, 40), mode='file',
                                         categorical_labels=True,   normalize=True)


    return X_train, Y_train, X_test, Y_test
def data_prep_(conf, method = 'seq', training_size = 0.5, clean_start = False, hdfs=True):
    '''
    create hdf5 files based on train and test selecting strategy
    and return results as train and test sets
    :param conf: parameters from congigure file
    :param method: random / sequence; at sequence get first training_size of images as training
    :param training_size: portion of images that will be used in trainig
    :param start_clean: start from zero or work based on previous work
    :return: X_train, Y_train, X_test, Y_test
    '''

    data_folder = conf['data_folder']
    if clean_start:
        os.remove(data_folder + 'train.h5')
        os.remove(data_folder + 'test.h5')
    #in the case files is available just read it and return
    if os.path.isfile(data_folder + 'train.h5')  and os.path.isfile(data_folder + 'validation.h5'):
        _h5f = h5py.File(data_folder + 'train.h5', 'r')
        X_train = _h5f['X']
        Y_train = _h5f['Y']
        h5f_ = h5py.File(data_folder + 'validation.h5', 'r')
        X_test = h5f_['X']
        Y_test = h5f_['Y']

        return X_train, Y_train, X_test, Y_test

    # if the file is not available, continue creating it
    test_counter = 0
    counter = 0
    train_file = data_folder + 'train.txt'
    test_file = data_folder + 'test.txt'
    with open(data_folder + 'labeling.csv') as inputFile:
        df = pd.read_csv(data_folder + conf['data_frame'])
        with open(train_file, 'w+') as trainFile:
            with open(test_file, 'w+') as testFile:
                for f in os.listdir(data_folder + conf['cropped_folder']):
                    is_train = False
                    is_test = False
                    if method == 'random':
                        if random.random() < training_size:
                            is_train = True
                        else:
                            is_test = True
                    else:
                        if counter < len(df) * training_size:
                            is_train = True
                        else:
                            if random.random() < 0.25:
                                is_train = True
                            is_test = True

                    label = int(df.loc[int(f.split('.')[0].split('_')[0]) - 1, 'manual_label'])
                    line = data_folder + conf['cropped_folder'] + f  + ' ' + str(label) + '\n'
                    if is_train:
                        trainFile.write(line)
                    if is_test and not '_' in f:
                        testFile.write(line)

    if hdfs:
        # Build a HDF5 dataset (only required once)
        build_hdf5_image_dataset(train_file, image_shape=(40, 40), mode='file', output_path=data_folder + 'train.h5',
                                 categorical_labels=True, normalize=True)
        build_hdf5_image_dataset(test_file, image_shape=(40, 40), mode='file', output_path=data_folder + 'test.h5',
                                 categorical_labels=True, normalize=True)
        # Load HDF5 dataset
        _h5f = h5py.File(data_folder + 'train.h5', 'r')
        X_train = _h5f['X']
        Y_train = _h5f['Y']

        h5f_ = h5py.File(data_folder + 'test.h5', 'r')
        X_test = h5f_['X']
        Y_test = h5f_['Y']
    else:
        X_train, Y_train = image_preloader(data_folder +'train.txt', image_shape=(40, 40), mode='file',
                                           categorical_labels=True, normalize=True)
        X_test, Y_test = image_preloader(data_folder +'test.txt', image_shape=(40, 40), mode='file',
                                         categorical_labels=True,   normalize=True)


    return X_train, Y_train, X_test, Y_test