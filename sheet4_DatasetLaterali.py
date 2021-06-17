import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow.keras.backend as K
import tifffile
from tensorflow.keras.applications.imagenet_utils import preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array,save_img,ImageDataGenerator,array_to_img
import pickle
# Prepare Train Data
from main import loadModel

vgg_face=loadModel()
import tensorflow as tf
# Prepare Test Data
depth_folders=os.listdir('D:/Progetto FVAB/sheet4/depth laterali/')
color_folders=os.listdir('D:/Progetto FVAB/sheet4/log laterali/')
x_test=[]
y_test=[]

partial=[]
for i,classi in enumerate(color_folders):
    partial=[]
    for x,y in zip(os.listdir('D:/Progetto FVAB/sheet4/depth laterali/'+classi),os.listdir('D:/Progetto FVAB/sheet4/log laterali/'+classi)):

        print(str(i) +" "+classi)
        ### merge..
        img=load_img('D:/Progetto FVAB/sheet4/log laterali/'+classi+'/'+x,target_size=(224,224))

        depth=load_img('D:/Progetto FVAB/sheet4/depth laterali/'+classi+'/'+y,target_size=(224,224))
        img=img_to_array(img)
        img=np.expand_dims(img,axis=0)
        img=preprocess_input(img)
        img_encode=vgg_face(img) #vgg_face() model outputs (1,2622)

        depth=img_to_array(depth)
        depth=np.expand_dims(depth,axis=0)
        depth_encode=vgg_face(depth) #vgg_face() model outputs (1,2622)


        concat=tf.concat([img_encode,depth_encode],1)

        x_test.append(np.squeeze(concat).tolist())
        y_test.append(i)
x_test=np.array(x_test)
y_test=np.array(y_test)
np.save('D:/Progetto FVAB/test_data_FERET_sheet4.npy',x_test)
np.save('D:/Progetto FVAB/test_labels_FERET_sheet4.npy',y_test)