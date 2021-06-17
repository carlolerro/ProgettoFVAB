import cv2
import dlib
import os
# Load cnn_face_detector with 'mmod_face_detector'
dnnFaceDetector=dlib.cnn_face_detection_model_v1("mmod_human_face_detector.dat")
color_folders=os.listdir('D:/Progetto FVAB/sheet5/log laterali/')

for classi in color_folders:

    for x in os.listdir('D:/Progetto FVAB/sheet5/log laterali/'+classi):
        print(x)
        # Load image
        img=cv2.imread('D:/Progetto FVAB/sheet5/log laterali/'+classi+'/'+x)

        # Convert to gray scale
        gray=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find faces in image
        rects=dnnFaceDetector(gray,1)

        left,top,right,bottom=0,0,0,0

        # For each face 'rect' provides face location in image as pixel loaction
        for (i,rect) in enumerate(rects):

            left=rect.rect.left() #x1
            top=rect.rect.top() #y1
            right=rect.rect.right() #x2
            bottom=rect.rect.bottom() #y2
            width=right-left
            height=bottom-top
            print(left,top,right,bottom,width,height)
            # Crop image
            img_crop=img[top:top+height,left:left+width]

        #save crop image with person name as image name
        cv2.imwrite('D:/Progetto FVAB/sheet5/test/'+x,img_crop)