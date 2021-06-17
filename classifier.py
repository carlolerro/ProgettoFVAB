from datetime import datetime

import numpy as np

import tensorflow as tf
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.ensemble import AdaBoostClassifier
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Softmax, Flatten, Activation, BatchNormalization,MaxPooling2D
from tensorboard import program

x_test = np.load('D:/Progetto FVAB/test_data_FERET_sheet5.npy')
y_test = np.load('D:/Progetto FVAB/test_labels_FERET_sheet5.npy')

x_train = np.load('D:/Progetto FVAB/train_data_FERET_sheet5.npy')
y_train = np.load('D:/Progetto FVAB/train_labels_FERET_sheet5.npy')

log_dir = "logs/" + 'sheet5_25_32_206_206'
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir)



#
classifier_model = Sequential()
#
classifier_model.add(Dense(units=500,kernel_initializer='glorot_uniform',activation='relu'))
classifier_model.add(BatchNormalization())
classifier_model.add(Activation('tanh'))
classifier_model.add(Dropout(0.4))
#
classifier_model.add(Dense(units=300,kernel_initializer='glorot_uniform'))
classifier_model.add(BatchNormalization())
classifier_model.add(Activation('tanh'))
classifier_model.add(Dropout(0.4))
#
classifier_model.add(Dense(units=206,kernel_initializer='he_uniform'))
classifier_model.add(Activation('softmax'))
classifier_model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(), optimizer='nadam', metrics=['accuracy'])
#

classifier_model.fit(x_train, y_train, epochs=25, batch_size=32, validation_data=(x_test, y_test),callbacks=tensorboard_callback)
accuracy = classifier_model.evaluate(x_test, y_test, verbose=0)
print(accuracy)
tf.keras.models.save_model(classifier_model,'C:/Users/lerro/PycharmProjects/proj_feret/face_classifier_model.h5')
