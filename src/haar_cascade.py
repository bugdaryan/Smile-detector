import cv2
import numpy as np
from typing import Tuple, List

face_cascade = cv2.CascadeClassifier('../haar_cascades/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('../haar_cascades/haarcascade_eye_tree_eyeglasses.xml')
smile_cascade = cv2.CascadeClassifier('../haar_cascades/haarcascade_smile.xml')
teeth_cascade = cv2.CascadeClassifier('../haar_cascades/haarcascade_teeth.xml')


def detect_faces(img:np.ndarray)-> List:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces_detected = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    return faces_detected

def detect_eyes(img:np.ndarray, face:Tuple) -> List:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    (x,y,w,h) = face

    roi_gray = gray[y:y+h, x:x+w]

    eyes=eye_cascade.detectMultiScale(roi_gray, 1.05, 20)

    return eyes

def detect_mouth(img:np.ndarray, face:Tuple)->List:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    (x,y,w,h) = face

    roi_gray = gray[y:y+h, x:x+w]

    mouth = smile_cascade.detectMultiScale(roi_gray, 1.8,25)

    return mouth

def detect_teeth(img:np.ndarray, mouth:Touple)->List:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    (x,y,w,h) = mouth

    roi_gray = gray[y:y+h, x:x+w]

    teeth = teeth_cascade.detectMultiScale(roi_gray, 1.1, 1)

    return teeth