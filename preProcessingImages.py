import os,re
from PIL import Image
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt

def main():
    segmentation_images("masks_images/","masks_images_semantic_result/",debug_test_num=150 )

def atoi(text) : 
    return int(text) if text.isdigit() else text

def natural_keys(text) :
    return [atoi(c) for c in re.split('(\d+)', text)]

def segmentation_images(path, new_path, debug_test_num):
    filenames = []
    read_csv = pd.read_csv("class_dict_seg_original.csv", index_col=False, skipinitialspace=True)
    read_csv.head()
     
    for root, dirnames, filenames in os.walk(path):
        filenames.sort(key = natural_keys)
        rootpath = root

    count = 0
    for item in filenames:
        
        if debug_test_num !=0:
            if debug_test_num <= count:
                break
                
        count = count + 1
        
        if os.path.isfile(path+item):
            f, e = os.path.splitext(item)
            image_rgb = Image.open(path+item)
            image_rgb = np.asarray(image_rgb)
            new_image = np.zeros((image_rgb.shape[0],image_rgb.shape[1],3)).astype('int')

            for index, row  in read_csv.iterrows():
                new_image[(image_rgb[:,:,0]==row.r)&
                          (image_rgb[:,:,1]==row.g)&
                          (image_rgb[:,:,2]==row.b)]=np.array([index+1, index+1, index+1]).reshape(1,3)

            new_image = new_image[:,:,0]
            output_filename = new_path+f+'.png'
            cv2.imwrite(output_filename, new_image)
            print('writing file: ', output_filename)
            
        else:
            print('no file')
        
    print("number of files written: ", count)

if __name__ == '__main__':
    main() 