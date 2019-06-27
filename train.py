from tflearn import *
import deep_learning
import configparser
import tflearn
import os.path
import pickle
import numpy as np
import pandas as pd
from data_prep import *


def train_(env, epoch_num, learning_rate = 0.01, clean_start = False, model_name = None):
    '''
    a function that create input, create network, train network and report results
    :param env: local/AWS; environment that our system work on
    :param epoch_num: number of epochs to run
    :return:
    '''

    #----------------------reading constants from config file---------------------------------
    config = configparser.ConfigParser()
    config.read('config.ini')
    conf = config[env]
    image_size = conf['window_size']
    mean_colors = [float(conf['mean_r']), float(conf['mean_g']), float(conf['mean_b'])]
    #-----------------------------------------------------------------------------------------


    #----------------------------------------input layer------------------------------------------
    # read input  http://tflearn.org/data_utils/#build-hdf5-image-dataset
    X_train, Y_train, X_test, Y_test = data_prep(conf, clean_start = clean_start)
    #prepare input layer  http://tflearn.org/data_preprocessing/
    img_prep = tflearn.ImagePreprocessing()
    img_prep.add_featurewise_zero_center(mean=mean_colors,per_channel=True)
    # http://tflearn.org/layers/core/#input-data
    input_layer = input_data(shape=[None, image_size, image_size, 3], name = 'input_layer')
                             #,data_preprocessing=img_prep)
    #---------------------------------------------------------------------------------------------




    #-------------------------------create model--------------------------------------------------------
    # network
    #softmax = deep_learning.inception(input_layer, 2)
    if model_name.split('.')[0].split('-')[0] == 'inception':
        softmax = deep_learning.inception(input_layer, 2)
    elif model_name.split('.')[0].split('-')[0] == 'ResNet':
        softmax = deep_learning.ResNet(input_layer, 2)
    elif model_name.split('.')[0].split('-')[0] == 'VGGNet':
        softmax = deep_learning.VGGNet(input_layer, 2)
    elif model_name.split('.')[0].split('-')[0] == 'NiN':
        softmax = deep_learning.NiN(input_layer, 2)
    else:
        return
    # estimator layer
    f_score = tflearn.metrics.F2Score()
    momentum = Momentum(learning_rate=0.1, lr_decay=0.9, decay_step=250)
    network = tflearn.regression(softmax, optimizer='adam',
                         loss='categorical_crossentropy',metric=f_score)  #if want to finetune give 'restore=False'
    # model  http://tflearn.org/models/dnn/
    model = tflearn.DNN(network, checkpoint_path='model_' + model_name,
                        max_checkpoints=1, tensorboard_verbose=0, tensorboard_dir="./logs")
    if model_name != None:
        if os.path.isfile(model_name) and not clean_start:
            print 'load model learning_rate: ' + str(learning_rate) 
            model.load(model_name,weights_only=True)

    model.fit(X_train, Y_train, validation_set = (X_test, Y_test),n_epoch=epoch_num,  shuffle=True,
              show_metric=True, batch_size=128, snapshot_epoch=True, run_id=model_name)
    #---------------------------------------------------------------------------------------------

    #model.save(model_name)
    return model_name
def train(env, epoch_num, learning_rate = 0.01, clean_start = True, model_name = None):
    '''
    a function that create input, create network, train network and report results
    :param env: local/AWS; environment that our system work on
    :param epoch_num: number of epochs to run
    :return:
    '''
    print 'model_name: ', model_name
    #----------------------reading constants from config file---------------------------------
    config = configparser.ConfigParser()
    config.read('config.ini')
    conf = config[env]
    image_size = conf['window_size']
    mean_colors = [float(conf['mean_r']), float(conf['mean_g']), float(conf['mean_b'])]
    data_splitting_method = conf['data_splitting_method']
    train_size = conf['train_size']
    #-----------------------------------------------------------------------------------------


    #----------------------------------------input layer------------------------------------------
    # read input  http://tflearn.org/data_utils/#build-hdf5-image-dataset
    X_train, Y_train, X_test, Y_test = data_prep(conf, clean_start = clean_start, method=data_splitting_method, training_size=train_size)
    #prepare input layer  http://tflearn.org/data_preprocessing/
    img_prep = tflearn.ImagePreprocessing()
    img_prep.add_featurewise_zero_center(mean=mean_colors,per_channel=True)
    # http://tflearn.org/layers/core/#input-data
    input_layer = input_data(shape=[None, image_size, image_size, 3], name = 'input_layer')
                             #,data_preprocessing=img_prep)
    #---------------------------------------------------------------------------------------------




    #-------------------------------create model--------------------------------------------------------
    # network
    #softmax = deep_learning.inception(input_layer, 2)
    #softmax = deep_learning.VGGNet(input_layer, 2)
    if model_name.split('.')[0].split('-')[0] == 'inception':
        softmax = deep_learning.inception(input_layer, 2)
    elif model_name.split('.')[0].split('-')[0] == 'ResNet':
        softmax = deep_learning.ResNet(input_layer, 2)
    elif model_name.split('.')[0].split('-')[0] == 'VGGNet':
        softmax = deep_learning.VGGNet(input_layer, 2)
    elif model_name.split('.')[0].split('-')[0] == 'NiN':
        softmax = deep_learning.NiN(input_layer, 2)
    else:
        return
    # estimator layer
    f_score = tflearn.metrics.F2Score()
    momentum = Momentum(learning_rate=0.1, lr_decay=0.9, decay_step=250)
    network = tflearn.regression(softmax, optimizer='adam',
                         loss='categorical_crossentropy',metric=f_score)  #if want to finetune give 'restore=False'
    # model  http://tflearn.org/models/dnn/
    model = tflearn.DNN(network, checkpoint_path='./models', best_checkpoint_path='./models/best',
                        max_checkpoints=1, tensorboard_verbose=0, tensorboard_dir="./logs", best_val_accuracy=0.0)
    if model_name != None:
        if os.path.isfile(model_name) and not clean_start:
            print 'load model learning_rate: ' + str(learning_rate) 
            model.load(model_name,weights_only=True)

    model.fit(X_train, Y_train, validation_set = (X_test, Y_test),n_epoch=epoch_num,  shuffle=True,
              show_metric=True, batch_size=128, snapshot_epoch=True, run_id=model_name)
    #---------------------------------------------------------------------------------------------
    print 'saving model.'
    model.save( model_name)
    print 'model: ', model_name, 'saved.'
    return model_name