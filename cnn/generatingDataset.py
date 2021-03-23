import os
import random
import re
from PIL import Image
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import cv2

FRAME_PATH = "original_images/"
MASK_PATH = "masks_images/"
DATA_PATH = "C:/Users/dessa/Documents/ProjetoFinalMineracaoDados/cnn/"

def main():
    #create_dataset()
    zipData()

def dataGen(img_folder, mask_folder, batch_size):
  c = 0
  n = os.listdir(img_folder)
  random.shuffle(n)
  
  while (True):
    img = np.zeros((batch_size, 512, 512, 3)).astype('float')
    mask = np.zeros((batch_size, 512, 512, 1)).astype('float')

    for i in range(c, c + batch_size): 

      train_img = cv2.imread(img_folder+'/'+n[i])/255.
      train_img =  cv2.resize(train_img, (512, 512))
      
      img[i-c] = train_img                                                

      train_mask = cv2.imread(mask_folder+'/'+n[i], cv2.IMREAD_GRAYSCALE)/255.
      train_mask = cv2.resize(train_mask, (512, 512))
      train_mask = train_mask.reshape(512, 512, 1) 

      mask[i-c] = train_mask

    c+= batch_size
    if(c + batch_size >= len(os.listdir(img_folder))):
      c = 0
      random.shuffle(n)
    yield img, mask

def zipData():
    train_datagen = ImageDataGenerator(
            rescale=1./255,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True)      
    val_datagen = ImageDataGenerator(rescale=1./255)

    train_image_generator = train_datagen.flow_from_directory(
        DATA_PATH + "train_frames", batch_size = 8)

    train_mask_generator = train_datagen.flow_from_directory(
        DATA_PATH + "train_masks", batch_size = 8)

    val_image_generator = val_datagen.flow_from_directory(
        DATA_PATH + "val_frames", batch_size = 8)

    val_mask_generator = val_datagen.flow_from_directory(
        DATA_PATH + "val_masks", batch_size = 8)

    train_generator = zip(train_image_generator, train_mask_generator)
    val_generator = zip(val_image_generator, val_mask_generator)

######### CREATE DATASET ######################################################

def add_frames(dir_name, image): 
    img = Image.open(DATA_PATH + FRAME_PATH + image)
    img.save(DATA_PATH + dir_name + '/' + image) 
    
def add_masks(dir_name, image): 
    img = Image.open(DATA_PATH + MASK_PATH + image)
    img.save(DATA_PATH + dir_name + '/' + image)

def create_dataset():
    folders = ['train_frames', 'train_masks', 'val_frames', 'val_masks', 'test_frames', 'test_masks']
    for folder in folders:
        os.makedirs(DATA_PATH + folder)

    all_frames = os.listdir(DATA_PATH + FRAME_PATH)
    all_masks = os.listdir(DATA_PATH + MASK_PATH)
    all_frames.sort(key=lambda var:[int(x) if x.isdigit() else x 
                                    for x in re.findall(r'[^0-9]|[0-9]+', var)])
    all_masks.sort(key=lambda var:[int(x) if x.isdigit() else x 
                                   for x in re.findall(r'[^0-9]|[0-9]+', var)])
    random.seed(230)
    random.shuffle(all_frames)

    train_split = int(0.7*len(all_frames))
    val_split = int(0.9 * len(all_frames))

    train_frames = all_frames[:train_split]
    val_frames = all_frames[train_split:val_split]
    test_frames = all_frames[val_split:]

    train_masks = [f for f in all_masks if f in train_frames]
    val_masks = [f for f in all_masks if f in val_frames]
    test_masks = [f for f in all_masks if f in test_frames]

    frame_folders = [(train_frames, 'train_frames'), (val_frames, 'val_frames'), 
                     (test_frames, 'test_frames')]
    mask_folders = [(train_masks, 'train_masks'), (val_masks, 'val_masks'), 
                    (test_masks, 'test_masks')]

    for folder in frame_folders:
      array = folder[0]
      name = [folder[1]] * len(array)
      list(map(add_frames, name, array))             

    for folder in mask_folders:  
      array = folder[0]
      name = [folder[1]] * len(array)   
      list(map(add_masks, name, array))

if __name__ == '__main__':
    main() 