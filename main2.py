import numpy as np
import os
import cv2
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential,Model
from tensorflow.keras.layers import ZeroPadding2D,Convolution2D,MaxPooling2D,InputLayer
from tensorflow.keras.layers import Dense,Dropout,Softmax,Flatten,Activation,BatchNormalization
from tensorflow.keras.preprocessing.image import load_img,img_to_array
from tensorflow.keras.applications.imagenet_utils import preprocess_input
import tensorflow.keras.backend as K
import datetime, os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.client import device_lib
# Define VGG_FACE_MODEL architecture

x_train = np.load('D:/Progetto FVAB/train_data_FERET_sheet4.npy')
y_train = np.load('D:/Progetto FVAB/train_labels_FERET_sheet4.npy')
x_test = np.load('D:/Progetto FVAB/test_data_FERET_sheet4.npy')
y_test = np.load('D:/Progetto FVAB/test_labels_FERET_sheet4.npy')

model = Sequential()
model.add(InputLayer(input_shape=(224,224, 4)))

model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(64, (3, 3), activation='relu'))
model.add(Flatten())
model.add(Dense(210,activation='softmax'))
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(x=x_train,y=y_train,validation_data=x_test,epochs=10)
