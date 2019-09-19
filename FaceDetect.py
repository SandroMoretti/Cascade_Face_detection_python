# -*- coding: UTF-8 -*-
import numpy as np
import cv2
from matplotlib import pyplot as plt



def calcHistograma(img):
    histg = cv2.calcHist([img],[0],None,[256],[0,256])
    return histg


plt.rcParams['figure.figsize'] = (224, 224)

face_cascade = cv2.CascadeClassifier('modelo/haarcascade_frontalface_default.xml')
#eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')


cap = cv2.VideoCapture('imagem/video.mp4')


cont = 0
flag = 0

fileHist = open(r"Hist\\hist.txt","w+") 
fileHistFace = open(r"HistFace\\histFace.txt","w+") 

while(cap.isOpened()):
    flag = flag+1
    ret, frame = cap.read()
    if ret==True:
        img = frame
        hist = calcHistograma(img)
        fileHist.write("Frame " + str(flag) + ":\n\n\n")
        for histI in hist:
            fileHist.write(str(histI[0])+"\n")
        fileHist.write("\n\n")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        

        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
        )
        for (x,y,w,h) in faces:
            crop_img = img[y:y+h, x:x+w]
            hist = calcHistograma(crop_img)
            fileHistFace.write("Face " + str(cont) + ":\n\n\n")
            for histI in hist:
                fileHistFace.write(str(histI[0])+"\n")
            fileHistFace.write("\n\n")
            cv2.imwrite("Pessoas/" + str(cont) + ".jpg", crop_img)
            cont=cont+1
    else:
        break

cv2.destroyAllWindows()
fileHist.close()
fileHistFace.close()