import pickle

import numpy as np
import tensorflow as tf
from main import loadModel
import tensorflow.keras.backend as K
from tensorflow.keras.applications.imagenet_utils import preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
vgg_face=loadModel()
a_file = open("data.pkl", "rb")
person_rep = pickle.load(a_file)

crop_img=load_img('/log/00743_960530_hr.ppm', target_size=(224, 224))
crop_img=img_to_array(crop_img)
crop_img=np.expand_dims(crop_img,axis=0)
crop_img=preprocess_input(crop_img)
img_encode=vgg_face(crop_img)
#
classifier_model=tf.keras.models.load_model('C:/Users/lerro/PycharmProjects/proj_feret/face_classifier_model.h5')
embed=K.eval(img_encode)
person=classifier_model.predict(embed)
name=np.argmax(person)
for i,x  in enumerate(person_rep):
    if i==name:
        print(x)
        print(name)

        break
