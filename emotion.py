from keras.models import load_model #for loading our pre-trained model
from keras.preprocessing.image import img_to_array
from keras.preprocessing import image
import cv2
import numpy as np


face_classifier = cv2.CascadeClassifier(r'C:\Users\USER\Desktop\FR\haarcascade_frontalface_default.xml')
model = load_model(r'C:\Users\USER\Desktop\FR\Emotion_little_vgg.h5')

class_labels = ['Angry','Happy','Sad','Neutral','Surprise','Disgust','Fear']

cap = cv2.VideoCapture(0)       #0 for inbuild camera and 1 for external cameras

while True:
    ret, frame = cap.read()     #store a single frame of camera
    labels = []
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)       #converting frame to gray for easy detection
    faces = face_classifier.detectMultiScale(gray,1.3,5)        #if 5(minNeighbor) datapoints indicate we are happy, and less indicate we are sad, then happy is the emotion

    for (x,y,w,h) in faces:     #for drawing rectangles around faces
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)        #position of rectangle with border color in BGR and thickness 2
        roi_gray = gray[y:y+h,x:x+w]        #creating our region of interest
        roi_gray = cv2.resize(roi_gray,(48,48),interpolation=cv2.INTER_AREA)       # resizing the rectangle


        if np.sum([roi_gray])!=0:       #if atleast one face is present
            roi = roi_gray.astype('float')/255.0        #dividing by 255 to reduce the pixel size
            roi = img_to_array(roi)     #converting the img to array for mathematical calculations
            roi = np.expand_dims(roi,axis=0)


            preds = model.predict(roi)[0]       #need the first index value of the prediction
            labels = class_labels[preds.argmax()]
            label_position = (x,y)      #position of labels
            cv2.putText(frame,labels,label_position,cv2.FONT_HERSHEY_SIMPLEX,2,(0,255,0),3)

        else:
            cv2.putText(frame,'No Face Found',(20,60),cv2.FONT_HERSHEY_SIMPLEX,2,(0,255,0),3)

    cv2.imshow('Emotion Detector',frame)    #for showing the frame
    if cv2.waitKey(1) & 0xFF == ord('q'):       #without this line the frame will get stuck
        break

cap.release()
cv2.destroyAllWindows()





