from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import fbeta_score
from sklearn.metrics import confusion_matrix
from tflearn import *
import deep_learning
import configparser
import tflearn
import os.path
import pickle
import numpy as np
import pandas as pd
from data_prep import *

import datetime
def gettime():
    now = datetime.datetime.now().time()
    today = datetime.datetime.now().date()
    return  str(today.year) + '_' + str(today.month) + '_' + str(today.day) + '_' + str(now.hour) + '_' + str(now.minute)


def test_(env,  model_name, clean_start = False):
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
    X_train, Y_train, X_test, Y_test = deep_learning.data_prep_(conf)
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
    network = tflearn.regression(softmax, optimizer='momentum',
                         loss='categorical_crossentropy',
                         learning_rate=0.0001, metric=f_score)  #if want to finetune give 'restore=False'
    # model  http://tflearn.org/models/dnn/
    model = tflearn.DNN(network, checkpoint_path='model_' + model_name,
                        max_checkpoints=1, tensorboard_verbose=0, tensorboard_dir="./logs_test")
    if os.path.isfile(model_name) and not clean_start:
        model.load(model_name)

    #model.fit(X_train, Y_train, validation_set = (X_test, Y_test),n_epoch=epoch_num,  shuffle=True,
    #          show_metric=True, batch_size=100, snapshot_step=100, snapshot_epoch=False, run_id='inception_ercis')
    #---------------------------------------------------------------------------------------------
    #model.save('inception.model')

    print 'results: ', model.evaluate(X_test, Y_test, batch_size=128)

def test(env,  model_name, clean_start = False):
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
    X_train, Y_train, X_test, Y_test = data_prep(conf)
    #prepare input layer  http://tflearn.org/data_preprocessing/
    img_prep = tflearn.ImagePreprocessing()
    img_prep.add_featurewise_zero_center(mean=mean_colors,per_channel=True)
    # http://tflearn.org/layers/core/#input-data
    input_layer = input_data(shape=[None, image_size, image_size, 3], name = 'input_layer')
                             #,data_preprocessing=img_prep)
    #---------------------------------------------------------------------------------------------


    #-------------------------------create model--------------------------------------------------------
    # network
    if model_name.split('.')[0].split('-')[0] == 'inception':
        softmax = deep_learning.inception(input_layer, 2)
    elif model_name.split('.')[0].split('-')[0] == 'ResNet':
        softmax = deep_learning.ResNet(input_layer, 2)
    elif model_name.split('.')[0].split('-')[0] == 'VGGNet':
        softmax = deep_learning.VGGNet(input_layer, 2)
    elif model_name.split('.')[0].split('-')[0] == 'NiN':
        softmax = deep_learning.NiN(input_layer, 2)
    else:
        softmax = deep_learning.inception(input_layer, 2)
    #softmax = deep_learning.inception(input_layer, 2)
    #softmax = deep_learning.VGGNet(input_layer, 2)
    # estimator layer
    f_score = tflearn.metrics.F2Score()
    momentum = Momentum(learning_rate=0.1, lr_decay=0.9, decay_step=250)
    network = tflearn.regression(softmax, optimizer='adam',
                         loss='categorical_crossentropy',
                         learning_rate=0.0001, metric=f_score)  #if want to finetune give 'restore=False'
    # model  http://tflearn.org/models/dnn/
    model = tflearn.DNN(network, checkpoint_path='model_' + model_name,
                        max_checkpoints=1, tensorboard_verbose=0, tensorboard_dir="./logs_test")
    if os.path.isfile(model_name) and not clean_start:
        model.load( model_name)

    #model.fit(X_train, Y_train, validation_set = (X_test, Y_test),n_epoch=epoch_num,  shuffle=True,
    #          show_metric=True, batch_size=100, snapshot_step=100, snapshot_epoch=False, run_id='inception_ercis')
    #---------------------------------------------------------------------------------------------
    #model.save('inception.model')
    temp = []
    counter = 0
    for x in X_test:
        temp.append( model.predict(x.reshape(1,40, 40, 3))[0])
        counter += 1
        print '{0}\r'.format(counter),
    #print temp
    predict  = temp
    with open('results/test_pred_{0}.pik'.format(gettime()), 'w') as f:
        pickle.dump(predict, f)
    
    target = np.argmax(Y_test[()], axis=1)
    pred = np.argmax(predict, axis=1)
    
    #print Y_test.shape
    #for x,y in zip(target, pred):
    #    print x, y
    print type(target)
    print target.shape
    print type(pred)
    print pred.shape
    #for y in Y_test:
    #    print y
    #try reporting
    #try:
    print 'accuracy_score: ', accuracy_score(target, pred)
    print 'recall_score: ', recall_score(target, pred)
    print 'precision_score: ', precision_score(target, pred)
    print 'f2_score: ', fbeta_score(target, pred, 2)
    print 'confusion_matrix: '
    print confusion_matrix(target, pred)
    with open('results/results.txt', 'a+') as f:
        f.write('# ' + model_name.split('_')[0] + '_' + gettime() + '\n')
        f.write('accuracy_score: {0}\n'.format(accuracy_score(target, pred)))
        f.write('recall_score: {0}\n'.format(recall_score(target, pred)))
        f.write('precision_score: {0}\n'.format(precision_score(target, pred)))
        f.write('f2_score: {0}\n'.format(fbeta_score(target, pred, 2)))
        f.write('confusion_matrix: \n')
        f.write(str(confusion_matrix(target, pred)))
        f.write('\n\n\n')
    #except:
        #print 'there was some error'
        #pass