
from random import randrange
from matplotlib.pyplot import close
import cv2

trainedface=cv2.CascadeClassifier('D:\OPENCV\opencv\data\haarcascades\haarcascade_frontalface_default.xml')
webcam=cv2.VideoCapture(0)

while True:
    successful_frame_rate,frame=webcam.read()
    grayscaled_img=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    
    facecoordinates=trainedface.detectMultiScale(grayscaled_img)
    
    for (x,y,w,h) in facecoordinates:
        
        #cv2.rectangle(frame, (x,y), (x+w,y+h), (randrange(256), randrange(256), randrange(256)), 5)
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),5)
    cv2.imshow('Face Detector',frame)
    k=cv2.waitKey(1)
    if (k==81 or k==113):
        break
webcam.releace()
    