from keras.layers import Input, Add, LeakyReLU, Activation, concatenate, BatchNormalization, Flatten, Conv2D, AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D
from keras.models import Model, load_model
import xml.etree.ElementTree as ET
from keras.preprocessing.image import load_img, img_to_array, array_to_img
from keras.optimizers import Adam
from keras.initializers import glorot_uniform
#%matplotlib inline
import Util_V1 as U
import os
from keras.utils.np_utils import to_categorical
import keras.backend as K
K.set_image_data_format('channels_last')
K.set_learning_phase(1)
from datetime import datetime
import sys
import numpy as np


def inception_block(X, filters, stage, block):
    """
    Implementation of the convolutional block as defined in Figure 4

    Arguments:
    X -- input tensor of shape (m, n_H_prev, n_W_prev, n_C_prev)
    f -- integer, specifying the shape of the middle CONV's window for the main path
    filters -- python list of integers, defining the number of filters in the CONV layers of the main path
    stage -- integer, used to name the layers, depending on their position in the network
    block -- string/character, used to name the layers, depending on their position in the network

    Returns:
    X -- output of the convolutional block, tensor of shape (n_H, n_W, n_C)
    """
    Conv_name_base = block + '_branch'
    BN_name_base = block + '_branch'
    F1, F2, F3, F4 = filters
    print(X)

    X1 = Conv2D(filters=F1, kernel_size=(1, 1), padding='valid', strides=(1, 1), name=Conv_name_base + 'Conv1a',
                kernel_initializer=glorot_uniform(seed=0))(X)
    X1 = BatchNormalization(axis=1, name=BN_name_base + 'Batch1a')(X1)
    X1 = Activation('relu')(X1)

    X2 = Conv2D(filters=F2, kernel_size=(1, 1), padding='valid', strides=(1, 1), name=Conv_name_base + 'Conv1b',
                kernel_initializer=glorot_uniform(seed=0))(X)
    X2 = BatchNormalization(axis=1, name=BN_name_base + 'Batch1b')(X2)
    X2 = Activation('relu')(X2)
    X2 = Conv2D(filters=F2, kernel_size=(3, 3), padding='same', strides=(1, 1), name=Conv_name_base + 'Conv2b',
                kernel_initializer=glorot_uniform(seed=0))(X2)
    X2 = BatchNormalization(axis=1, name=BN_name_base + 'Batch2b')(X2)
    X2 = Activation('relu')(X2)

    X3 = Conv2D(filters=F3, kernel_size=(1, 1), padding='valid', strides=(1, 1), name=Conv_name_base + 'Conv1c',
                kernel_initializer=glorot_uniform(seed=0))(X)
    X3 = BatchNormalization(axis=1, name=BN_name_base + 'Batch1c')(X3)
    X3 = Activation('relu')(X3)
    X3 = Conv2D(filters=F3, kernel_size=(5, 5), padding='same', strides=(1, 1), name=Conv_name_base + 'Conv2c',
                kernel_initializer=glorot_uniform(seed=0))(X3)
    X3 = BatchNormalization(axis=1, name=BN_name_base + 'Batch2c')(X3)
    X3 = Activation('relu')(X3)

    X4 = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(X)
    X4 = Conv2D(filters=F4, kernel_size=(1, 1), padding='valid', strides=(1, 1), name=Conv_name_base + 'Conv1d',
                kernel_initializer=glorot_uniform(seed=0))(X4)
    X4 = BatchNormalization(axis=1, name=BN_name_base + 'Batch1d')(X4)
    X4 = Activation('relu')(X4)

    X = concatenate([X1, X2, X3, X4],  axis=-1)
    return X


def network_architecture(input_data):
    """
    Implementation of the HappyModel.

    Arguments:
    input_shape -- shape of the images of the dataset

    Returns:
    model -- a Model() instance in Keras
    """
    X_shortcut_1 = 0
    X_shortcut_2 = 0
    Input_data = Input(input_data)  # 640x480
    X = Conv2D(filters=32, kernel_size=(8, 8), strides=2, kernel_initializer=glorot_uniform(seed=0), padding='valid')(
        Input_data)  # 317x237
    X_shortcut_1 = X
    X = BatchNormalization(axis=1)(X)
    X = LeakyReLU(alpha=0.3)(X)
    X = MaxPooling2D(pool_size=(3, 3), strides=2, padding='valid')(X)  # 158x118

    X = inception_block(X=X, filters=[64, 32, 32, 64], block='inception', stage=1)  # 158x118
    X_shortcut_2 = X
    X = BatchNormalization(axis=1)(X)
    X = LeakyReLU(alpha=0.3)(X)

    X = Conv2D(filters=128, kernel_size=(2, 2), strides=2, padding='valid', kernel_initializer=glorot_uniform(seed=0))(X)  # 79x59
    X = BatchNormalization(axis=1)(X)
    X = LeakyReLU(alpha=0.3)(X)

    X_shortcut_1 = Conv2D(filters=128, kernel_size=(5, 5), strides=4, padding='valid', kernel_initializer=glorot_uniform(seed=0))(X_shortcut_1)
    X_shortcut_1 = BatchNormalization(axis=1)(X_shortcut_1)
    X_shortcut_1 = LeakyReLU(alpha=0.3)(X_shortcut_1)

    X = Add()([X_shortcut_1, X])
    X = Activation('relu')(X)

    X = Conv2D(filters=256, kernel_size=(3, 3), strides=2, padding='valid', kernel_initializer=glorot_uniform(seed=0))(X)  # 39x29
    X = BatchNormalization(axis=1)(X)
    X = LeakyReLU(alpha=0.3)(X)

    X_shortcut_2 = Conv2D(filters=256, kernel_size=(6, 6), strides = 4, padding='valid', kernel_initializer=glorot_uniform(seed=0))(X_shortcut_2)
    X_shortcut_2 = BatchNormalization(axis= 1)(X_shortcut_2)
    X_shortcut_2 = LeakyReLU(alpha=0.3)(X_shortcut_2)

    X = concatenate([X_shortcut_2, X], axis=-1)

    X = AveragePooling2D(pool_size=(3, 3), strides=2, padding='valid')(X)  # 19x19
    
    X0 = Conv2D(filters=5, kernel_size=(1, 1), strides=1, padding='valid', kernel_initializer=glorot_uniform(seed=0))(X)  # 19x14
    X0 = Activation('sigmoid')(X0)
    X1 = Conv2D(filters=10, kernel_size=(1, 1), strides=1, padding='valid', kernel_initializer=glorot_uniform(seed=0))(X)  # 19x14
    X1 = Activation('softmax')(X1)
    X = concatenate(inputs=[X0, X1], axis= -1)
    model = Model(inputs=Input_data, outputs=X, name='SimpleYOLO')
    return model