import numpy as np 
import cv2 as cv 
import os


capture = cv.VideoCapture(0)

while True: 
    _, img = capture.read()
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    haar_cascade = cv.CascadeClassifier(os.path.join(cv.data.haarcascades, 'haarcascade_frontalface_default.xml')) 
    faces_rect = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=1)
    for (x,y,w,h) in faces_rect: 
        mask = np.zeros(img.shape[:2], dtype="uint8")
        cv.rectangle(mask, (x,y), (x+w, y+h), (255,255,255), -1)  

        masked = cv.bitwise_and(img, img, mask=mask)
        blur_mask = cv.GaussianBlur(masked, (101, 101), 0)

        background_mask = cv.bitwise_not(mask)
        background = cv.bitwise_and(img,img, mask=background_mask)

        result = cv.add(background, blur_mask)
        cv.rectangle(result, (x,y), (x+w, y+h), (0,255,0), 2) #boundary line

    cv.imshow('video', result)

    if cv.waitKey(20) & 0xFF==ord('d'): 
        break  

capture.release()
cv.destroyAllWindows()

cv.waitKey(0)