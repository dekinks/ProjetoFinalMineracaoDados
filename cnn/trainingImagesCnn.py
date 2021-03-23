from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
from keras.callbacks import ModelCheckpoint
from keras.callbacks import CSVLogger
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam
from keras import backend as K
import numpy as np 
import os
import skimage.io as io
import skimage.transform as trans
import numpy as np
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as keras
import generatingDataset

DATA_PATH = "C:/Users/dessa/Documents/ProjetoFinalMineracaoDados/cnn/"
NO_OF_TRAINING_IMAGES = len(os.listdir(DATA_PATH + 'train_frames/'))
NO_OF_VAL_IMAGES = len(os.listdir(DATA_PATH + 'val_frames/'))
NO_OF_EPOCHS = 2
BATCH_SIZE = 8
weights_path = DATA_PATH + 'train_result'

def main():
    cnnSegmentation()

def diceCoef(y_true, y_pred, smooth=1):
    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    return (2. * intersection + smooth) / (K.sum(K.square(y_true),-1) + K.sum(K.square(y_pred),-1) + smooth)

def diceCoefLoss(y_true, y_pred):
    return 1-diceCoef(y_true, y_pred)

def unet(pretrained_weights = None,input_size = (255,255,1)):
    inputs = Input(input_size)
    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

    conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool4)
    conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
    drop5 = Dropout(0.5)(conv5)

    up6 = Conv2D(512, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(drop5))
    merge6 = concatenate([drop4,up6], axis = 3)
    conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
    conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)

    up7 = Conv2D(256, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv6))
    merge7 = concatenate([conv3,up7], axis = 3)
    conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)
    conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)

    up8 = Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv7))
    merge8 = concatenate([conv2,up8], axis = 3)
    conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)
    conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)

    up9 = Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv8))
    merge9 = concatenate([conv1,up9], axis = 3)
    conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge9)
    conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    conv9 = Conv2D(2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    conv10 = Conv2D(1, 1, activation = 'sigmoid')(conv9)

    model = Model(inputs = inputs, outputs = conv10)

    model.compile(optimizer = Adam(lr = 1e-4), loss = 'binary_crossentropy', metrics = ['accuracy'])
    
    if(pretrained_weights):
    	model.load_weights(pretrained_weights)

    return model

def cnnSegmentation(): 
    m = unet()
    opt = Adam(lr=1E-5, beta_1=0.9, beta_2=0.999, epsilon=1e-08)

    m.compile(loss=diceCoefLoss,
                  optimizer=opt,
                  metrics=diceCoef)

    checkpoint = ModelCheckpoint(weights_path, monitor=diceCoefLoss, 
                                 verbose=1, save_best_only=True, mode='max')

    csv_logger = CSVLogger('./log.out', append=True, separator=';')

    earlystopping = EarlyStopping(monitor = diceCoefLoss, verbose = 1,
                                  min_delta = 0.01, patience = 3, mode = 'max')

    callbacks_list = [checkpoint, csv_logger, earlystopping]

    train_frame_path = DATA_PATH + "train_frames"
    train_mask_path = DATA_PATH + "train_masks"
    val_frame_path = DATA_PATH + "val_frames"
    val_mask_path = DATA_PATH  + "val_frames"

    train_gen = generatingDataset.dataGen(train_frame_path,train_mask_path, batch_size = 4)
    val_gen = generatingDataset.dataGen(val_frame_path,val_mask_path, batch_size = 4)

    results = m.fit_generator(train_gen, epochs=NO_OF_EPOCHS, 
                              steps_per_epoch = (NO_OF_TRAINING_IMAGES//BATCH_SIZE),
                              validation_data=val_gen, 
                              validation_steps=(NO_OF_VAL_IMAGES//BATCH_SIZE), 
                              callbacks=callbacks_list)
    m.save('Model.h5')

if __name__ == '__main__':
    main() 