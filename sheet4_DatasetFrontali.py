import os
import numpy as np
import tensorflow.keras.backend as K
import tifffile
from tensorflow.keras.applications.imagenet_utils import preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array,save_img
import pickle
import tensorflow as tf
# Prepare Train Data
from main import loadModel
import tensorflow

x_train=[]
y_train=[]
partial=[]
depth_folders=os.listdir('D:/Progetto FVAB/sheet4/depth frontali/')
color_folders=os.listdir('D:/Progetto FVAB/sheet4/log frontali/')
#Train data
from main import loadModel
vgg_face=loadModel()
for i,classi in enumerate(color_folders):
    partial=[]
    for x,y in zip(os.listdir('D:/Progetto FVAB/sheet4/depth frontali/'+classi),os.listdir('D:/Progetto FVAB/sheet4/log frontali/'+classi)):

        print(str(i) +" "+classi)
    ### merge..
        img=load_img('D:/Progetto FVAB/sheet4/log frontali/'+classi+'/'+x,target_size=(224,224))
        depth=load_img('D:/Progetto FVAB/sheet4/depth frontali/'+classi+'/'+y,target_size=(224,224))

        img=img_to_array(img)
        img=np.expand_dims(img,axis=0)
        img=preprocess_input(img)
        img_encode=vgg_face(img) #vgg_face() model outputs (1,2622)

        depth=img_to_array(depth)
        depth=np.expand_dims(depth,axis=0)
        depth_encode=vgg_face(depth) #vgg_face() model outputs (1,2622)



        concat=tf.concat([img_encode,depth_encode],1)
        x_train.append(np.squeeze(concat).tolist())
        y_train.append(i)
x_train=np.array(x_train)
y_train=np.array(y_train)
np.save('D:/Progetto FVAB/train_data_FERET_sheet4.npy',x_train)
np.save('D:/Progetto FVAB/train_labels_FERET_sheet4.npy',y_train)