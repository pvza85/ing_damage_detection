import cv2
import imutils
import os
import pandas as pd

def augment(image, output_folder, blur=True):
    '''
    get an image with full path and extract augment it and save it
    :param image:
    :return:
    '''
    img = cv2.imread(image)
    image_name = os.path.abspath(image)
    image_name = image.split('/')[-1] # get file name
    image_name, ext = image_name.split('.') #get file name and extension
    if not output_folder.endswith('/'):
        output_folder += '/'

    cv2.imwrite('{0}{1}_v.{2}'.format(output_folder, image_name, ext), cv2.flip(img, 1))
    cv2.imwrite('{0}{1}_h.{2}'.format(output_folder, image_name, ext), cv2.flip(img, 0))
    cv2.imwrite('{0}{1}_90.{2}'.format(output_folder, image_name, ext), imutils.rotate(img, 90))
    cv2.imwrite('{0}{1}_180.{2}'.format(output_folder, image_name, ext), imutils.rotate(img, 180))
    cv2.imwrite('{0}{1}_270.{2}'.format(output_folder, image_name, ext), imutils.rotate(img, 270))
    if blur:
        cv2.imwrite('{0}{1}_b.{2}'.format(output_folder, image_name, ext), cv2.blur(img, (5, 5), 0))
        cv2.imwrite('{0}{1}_gb.{2}'.format(output_folder, image_name, ext), cv2.GaussianBlur(img, (5, 5), 0))

    
def data_augmentation(conf, blur = True ):
    workdir = conf['data_folder'] + conf['cropped_folder']
    df = pd.read_csv(conf['data_folder'] + conf['data_frame'])
    for row in df.iterrows():
        if row[1]['manual_label'] == 1:
            file_name = workdir + str(int(row[1]['index'] - 1)).zfill(5) + '.tif'
            augment(file_name, workdir, blur=blur)