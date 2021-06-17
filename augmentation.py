import pickle

import numpy as np
from keras_preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import os
from matplotlib import pyplot
import PIL
import numpy as np
import tensorflow.keras.backend as K
from tensorflow.keras.applications.imagenet_utils import preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array,save_img,array_to_img
import open3d as o3d
import tensorflow as tf
from tensorflow.keras import layers
person=os.listdir('D:/Progetto FVAB/sheet6/log frontali/')

data_augmentation = tf.keras.Sequential([

    layers.experimental.preprocessing.RandomTranslation(height_factor=(-0.2,0.3),width_factor=(-0.2,0.3)),
])


for i,classi in enumerate(person):
    #z=z+1
    for x in (os.listdir('D:/Progetto FVAB/sheet6/log frontali/'+classi)):
        img=load_img('D:/Progetto FVAB/sheet6/log frontali/'+classi+'/'+x)
        img=img_to_array(img)
        img=np.expand_dims(img,axis=0)
        for z in range(4):
            image = data_augmentation(img)
            image=np.squeeze(image,0)
            print(x[:-3])
            save_img('D:/Progetto FVAB/sheet9/train/'+x[:-4]+str(z)+'.png',image)

