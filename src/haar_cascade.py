import cv2
import numpy as np

face_cascade = cv2.CascadeClassifier('../haar_cascades/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('../haar_cascades/haarcascade_eye_tree_eyeglasses.xml')
smile_cascade = cv2.CascadeClassifier('../haar_cascades/haarcascade_smile.xml')
teeth_cascade = cv2.CascadeClassifier('../haar_cascades/haarcascade_teeth.xml')

def detect_smile(img:np.ndarray, check_mouth_open:bool = False)->bool:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    face_detected = len(faces) != 0
    eyes_detected = False
    mouth_detected = False
    teeth_detected = not check_mouth_open
    
    for (x,y,w,h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        eyes=eye_cascade.detectMultiScale(roi_gray, 1.05, 20)
        if len(eyes) > 1:
            eyes_detected = True
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex+ew,ey+eh), (0, 255,0), 2)
        smiles = smile_cascade.detectMultiScale(roi_gray, 1.8,25)
        if len(smiles) > 0:
            mouth_detected = True
        for (ex, ey, ew, eh) in smiles:
            cv2.rectangle(roi_color, (ex, ey), (ex+ew,ey+eh), (0, 0,255), 2)
            
            if check_mouth_open:
                roi_gray_mouth = roi_gray[ey:ey+eh, ex:ex+ew]
                roi_color_mouth = roi_color[ey:ey+eh, ex:ex+ew]

                teeth = teeth_cascade.detectMultiScale(roi_gray_mouth, 1.1, 1)
                if len(teeth) != 0:
                    teeth_detected = True
                    
                    for (tx,ty,tw,th) in teeth:
                        cv2.rectangle(roi_gray_mouth, (tx, ty), (tx+tw,ty+th), (0, 255,255), 2)
    
    return face_detected and eyes_detected and mouth_detected and teeth_detected