import numpy as np

import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Softmax, Flatten, Activation, BatchNormalization


x_train = np.load('D:/Progetto FVAB/train_data_FERET_sheet1.npy')
y_train = np.load('D:/Progetto FVAB/train_labels_FERET_sheet1.npy')
x_test = np.load('D:/Progetto FVAB/test_data_FERET_sheet1.npy')
y_test = np.load('D:/Progetto FVAB/test_labels_FERET_sheet1.npy')
#
classifier_model = Sequential()

classifier_model.add(Dense(units=500, activation='relu'))
classifier_model.add(BatchNormalization())
classifier_model.add(Dropout(0.6))
#
classifier_model.add(Dense(units=300, activation='relu'))
classifier_model.add(BatchNormalization())
classifier_model.add(Dropout(0.4))
#
classifier_model.add(Dense(units=269, activation='relu'))
classifier_model.add(Activation('softmax'))
classifier_model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(), optimizer='adam', metrics=['accuracy'])
#
classifier_model.fit(x_train, y_train, epochs=100, batch_size=128, validation_data=(x_test, y_test))
accuracy = classifier_model.evaluate(x_test, y_test, verbose=0)
tf.keras.models.save_model(classifier_model,'C:/Users/lerro/PycharmProjects/proj_feret/face_classifier_model.h5')
print(accuracy)