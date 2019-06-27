import os
import h5py
import random
import pandas as pd
import os.path
from tflearn.data_utils import build_hdf5_image_dataset


def data_prep00(conf, method = 'random', training_size = 0.7, clean_start = False):
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
        df = pd.read_csv(data_folder + 'labeling.csv')
        with open(train_file, 'w+') as trainFile:
            with open(test_file, 'w+') as testFile:
                for row in df.iterrows():
                    row = row[1]
                    counter += 1
                    test_counter += 1
                    line = data_folder + 'ercis/' + str(int(row['index'] - 1)).zfill(5) + '.tif' \
                        + ' ' + str(int(row['manual_label'])) + '\n'
                    if method == 'random':
                        if random.random() < training_size:
                            trainFile.write(line)
                        else:
                            testFile.write(line)
                            test_counter += 1
                    else:
                        if counter < 37204 * training_size:  # random.random() < train_size:
                            trainFile.write(line)
                        else:
                            testFile.write(line)
                            test_counter += 1

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

    return X_train, Y_train, X_test, Y_test
def data_prep01(conf, method = 'random', training_size = 0.7, clean_start = False):
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
    #in the case files is available just read it and return
    if os.path.isfile(data_folder + 'train.h5')  and os.path.isfile(data_folder + 'test.h5'):
        _h5f = h5py.File(data_folder + 'train.h5', 'r')
        X_train = _h5f['X']
        Y_train = _h5f['Y']
        h5f_ = h5py.File(data_folder + 'test.h5', 'r')
        X_test = h5f_['X']
        Y_test = h5f_['Y']

        return X_train, Y_train, X_test, Y_test


    return None

def data_prep02(conf, method='random', training_size=0.7):
    '''
    create hdf5 files based on train and test selecting strategy
    and return results as train and test sets
    it augments train data
    :param conf: parameters from congigure file
    :param method: random / sequence; at sequence get first training_size of images as training
    :param training_size: portion of images that will be used in trainig
    :return: X_train, Y_train, X_test, Y_test
    '''

    data_folder = conf['data_folder']

    test_counter = 0
    counter = 0
    train_file = data_folder + 'train.txt'
    test_file = data_folder + 'test.txt'
    with open(data_folder + 'labeling.csv') as inputFile:
        df = pd.read_csv(data_folder + 'labeling.csv')
        with open(train_file, 'w+') as trainFile:
            with open(test_file, 'w+') as testFile:
                for row in df.iterrows():
                    row = row[1]
                    counter += 1
                    test_counter += 1
                    line = data_folder + 'ercis/' + str(int(row['index'] - 1)).zfill(5) + '.tif' \
                           + ' ' + str(int(row['manual_label'])) + '\n'
                    if method == 'random':
                        if random.random() < training_size:
                            trainFile.write(line)
                        else:
                            testFile.write(line)
                            trainFile.write(line)
                            test_counter += 1
                    else:
                        if counter < 37204 * training_size:  # random.random() < train_size:
                            trainFile.write(line)
                        else:
                            testFile.write(line)
                            trainFile.write(line)
                            test_counter += 1

    # Build a HDF5 dataset (only required once)
    build_hdf5_image_dataset(train_file, image_shape=(40, 40), mode='file', output_path=data_folder + 'train.h5',
                             categorical_labels=True, normalize=True)
    build_hdf5_image_dataset(test_file, image_shape=(40, 40), mode='file', output_path=data_folder + 'test.h5',
                             categorical_labels=True, normalize=True)
    # Load HDF5 dataset
    h5f = h5py.File('train.h5', 'w')
    X_train = h5f['X']
    Y_train = h5f['Y']

    h5f = h5py.File('test.h5', 'w')
    X_test = h5f['X']
    Y_test = h5f['Y']

    return X_train, Y_train, X_test, Y_test