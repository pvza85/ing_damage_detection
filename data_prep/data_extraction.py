from helper import *
import cv2
import os
import pandas as pd

def automatic_crop(file_name, window_size, step_size, output_folder='', output_file_name='', extract_to_txt=True, labeled=True):

    i = 0
    image = cv2.imread(file_name)
    abs_path =  os.path.abspath(file_name)
    current_folder = os.path.dirname(abs_path)
    abs_file_name = abs_path.split('/')[-1].split('.')[0]
    if output_file_name == '':
        output_file_name = current_folder + '/' + abs_file_name + '.txt'
    if output_folder == '':
        output_folder = current_folder + '/' + abs_file_name
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)
    
    for (x, y, window) in sliding_window(image, stepSize=step_size, windowSize=(window_size, window_size)):
        
        # if the window does not meet our desired window size, ignore it
        if window.shape[0] != window_size or window.shape[1] != window_size:
            continue
        else:
            i = i + 1
        name = '{0}/{1}.tif'.format(output_folder, str(i).zfill(5))
        cv2.imwrite(name, window)


def automatic_crop_(file_name, window_size, step_size, output_folder='', output_file_name='', extract_to_txt=True):

    i = 0
    image = cv2.imread(file_name)
    abs_path =  os.path.abspath(file_name)
    current_folder = os.path.dirname(abs_path)
    abs_file_name = abs_path.split('/')[-1].split('.')[0]
    if output_file_name == '':
        output_file_name = current_folder + '/' + abs_file_name + '.txt'
    if output_folder == '':
        output_folder = current_folder + '/' + abs_file_name
    os.mkdir(output_folder)
    if extract_to_txt:
        output_file = open(output_file_name, 'w')
    df = pd.DataFrame(columns=['path', 'minx_pix', 'minx_cord','miny_pix', 'miny_cord', 'maxx_pix', 'maxx_cord', 'maxy_pix', 'maxy_cord', 'label', 'prediction'])
    if if_geotiff(file_name):
        if_geo = True
    else:
        if_geo = False
    #print if_geo
    for (x, y, window) in sliding_window(image, stepSize=step_size, windowSize=(window_size, window_size)):
        i = i + 1
        # if the window does not meet our desired window size, ignore it
        if window.shape[0] != window_size or window.shape[1] != window_size:
            continue
        name = '{0}/{1}.tif'.format(output_folder, str(i).zfill(5))
        cv2.imwrite(name, window)
        if extract_to_txt:
            output_file.writelines(name + '\n')
        if if_geo:
            x0, y0 = convert_to_wgs84(file_name, (x, y))
            x1, y1 = convert_to_wgs84(file_name, (x+window_size, y+window_size))
        else:
            x0, y0 = -1, -1
            x1, y1 = -1, -1
        df = df.append(pd.DataFrame([{ 'path':name, 'minx_pix':x, 'minx_cord':x0,'miny_pix':y,
                                      'miny_cord':y0, 'maxx_pix':x + window_size, 'maxx_cord':x1,
                                      'maxy_pix':y + window_size, 'maxy_cord':y1, 'label':-1, 'prediction':-1}], index = [i]))
    df.to_csv(current_folder + '/' + abs_file_name + '.csv')
    return abs_file_name + '.csv'

def image_to_h5():
    automatic_crop()
    return None
def manual_label():
    pass