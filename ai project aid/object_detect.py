import cv2 #C:\Users\AI LAB\Desktop\CS PD
import cvzone
cap=cv2.VideoCapture(0)
classifier=cvzone.Classifier("keras_model.h5","labels.txt")
fpsreader = cvzone.FPS()
while True:
    ab,img = cap.read()
    predictions,index=classifier.getPrediction(img,scale =1)
    fps ,img = fpsreader .update(img,pos=(450,50))
    cv2.imshow("image",img)
    cv2.waitKey(1)