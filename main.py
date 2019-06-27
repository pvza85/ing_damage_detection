import warnings
warnings.filterwarnings('ignore')
import sys
from train import *
from test import *
from data_prep import *
import configparser
import fnmatch


def main():
    '''
    args = sys.argv
    model_name = args[2]
    if args[1] == 'train':
        train('AWS', int(args[3]),  model_name = model_name)
    elif args[1] == 'train_':
        train_('AWS', int(args[3]), learning_rate = float(args[4]), model_name = model_name)
    else:
        test('AWS', model_name)
    '''
    config = configparser.ConfigParser()
    config.read('config.ini')
    conf = config['AWS']


    image_file = conf['data_folder' ] + conf['image_file']
    out_folder = conf['data_folder' ] + conf['cropped_folder']
    win_size = int(conf['window_size'])
    

    args = sys.argv
    model_name = args[2]
    if args[1] == 'train':
        automatic_crop(file_name=image_file, window_size=win_size, step_size = 20, output_folder=out_folder)
        data_augmentation(conf)
        train('AWS', int(args[3]),  model_name = model_name)
    elif args[1] == 'train_':
        automatic_crop(file_name=image_file, window_size=win_size, step_size = 20, output_folder=out_folder)
        data_augmentation(conf, blur=False)
        train('AWS', int(args[3]),  model_name = model_name)
    else:
        import os
        
        m = model_name
        '''
        for f in os.listdir('./'):
            if fnmatch.fnmatch(f, 'model_' + model_name + '.*data*'):
                m = f
        if m == model_name:
            return
        '''
        print '********************'
        print 'test will run for: ', m
        print '********************'
        test('AWS', m)

if __name__ == '__main__':
    main()


