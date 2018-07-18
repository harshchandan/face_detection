import numpy as np
import cv2

face_classifier = cv2.CascadeClassifier('frontface.xml')
eye_classifier = cv2.CascadeClassifier('eyecascade.xml')

def face(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
        cv2.rectangle(image, (x,y), (x+w, y+h), (127, 0, 255), 2)
        eyes = eye_classifier.detectMultiScale(gray)
        for (ex,ey,ew,eh) in eyes:
            cv2.rectangle(image, (ex,ey), (ex+ew, ey+eh), (0, 0, 255), 2)
        
    return image


cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    cv2.imshow('Face Detection', face(frame))
    if cv2.waitKey(1) == 13:
        break
        
cap.release()
cv2.destroyAllWindows()
