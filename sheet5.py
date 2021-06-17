import os
import PIL
import numpy as np
import tensorflow.keras.backend as K
from tensorflow.keras.applications.imagenet_utils import preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Prepare Train Data
from main import loadModel

vgg_face = loadModel()
assegnamenti = dict()
x_train=[]
y_train=[]
person_folders=os.listdir('D:/Progetto FVAB/sheet5/train/')
#Train data

for i,classi in enumerate(person_folders):

    for x in (os.listdir('D:/Progetto FVAB/sheet5/train/'+classi)):

        assegnamenti[classi]=i
        print(str(i) +" "+classi)
        img=load_img('D:/Progetto FVAB/sheet5/train/'+classi+'/'+x,target_size=(224,224))
        img=img_to_array(img)
        img=np.expand_dims(img,axis=0)
        img=preprocess_input(img)
        img_encode=vgg_face(img)
        x_train.append(np.squeeze(K.eval(img_encode)).tolist())
        y_train.append(i)


# Prepare Test Data
person_folders=os.listdir('D:/Progetto FVAB/sheet5/test/')
x_test=[]
y_test=[]
z=0
for i,classi in enumerate(person_folders):
    z=z+1
    for x in (os.listdir('D:/Progetto FVAB/sheet5/test/'+classi)):

        if z<=1000:

            if classi in assegnamenti:
                classe= assegnamenti[classi]

            else:
                print("no classe")
                continue
                #classe=assegnamenti[classi]=max(assegnamenti.values())+1

            print(str(i) +" "+classi)
            img=load_img('D:/Progetto FVAB/sheet5/test/'+classi+'/'+x,target_size=(224,224))
            img=img_to_array(img)
            img=np.expand_dims(img,axis=0)
            img=preprocess_input(img)
            img_encode=vgg_face(img) #vgg_face() model outputs (1,2622) dimensional Tensor, it is converted into list and append to train and test data
            x_test.append(np.squeeze(K.eval(img_encode)).tolist())
            y_test.append(classe)


x_test=np.array(x_test)
y_test=np.array(y_test)
x_train=np.array(x_train)
y_train=np.array(y_train)

np.save('D:/Progetto FVAB/train_data_FERET_sheet5.npy',x_train)
np.save('D:/Progetto FVAB/train_labels_FERET_sheet5.npy',y_train)
np.save('D:/Progetto FVAB/test_data_FERET_sheet5.npy',x_test)
np.save('D:/Progetto FVAB/test_labels_FERET_sheet5.npy',y_test)