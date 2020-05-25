import cv2
import numpy as np
import time
import sys
import haar_cascade as cascade
from datetime import datetime

class SmileDetectStatus:
    def __init__(self):
        self.begin_take_photo = False
        self.face_found = False
        self.smile_detected = False
        self.restart = False
        self.completed = False
        self.no_smile_detect = 0
        self.smile_detect = 0

class Image:
    def __init__(self, cap):
        self.cap = cap
    
    def capture_image(self):
        ret, img = self.cap.read()
        self.captured = cv2.flip(img, 1)
        self.annotated = np.copy(self.captured)

class Detector:
    def __init__(self, image, status):
        self.image = image
        self.status = status
    
    def detect_smiles(self):
        faces = cascade.detect_faces(self.image.captured)

        eyes_detected = False
        mouth_detected = False
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        for (x,y,w,h) in faces:
            eyes = cascade.detect_eyes(self.image.captured, (x,y,w,h))
            if len(eyes) == 2:
                eyes_detected = True

            mouth = cascade.detect_mouth(self.image.captured, (x,y,w,h))
            if len(mouth) == 1:
                mouth_detected = True

            if self.status.smile_detected:
                color = (0, 255, 0)
            elif self.status.face_found:
                color = (0, 255, 255)
            else:
                color = (0,0,255)

            face = self.image.annotated[y:y+h, x:x+w]
            cv2.rectangle(self.image.annotated, (x, y), (x+w,y+h), color, 2)
            for (ex, ey, ew, eh) in eyes:
                cv2.rectangle(face, (ex,ey), (ex+ew, ey+eh), color)
            for (ex, ey, ew, eh) in mouth:
                cv2.rectangle(face, (ex,ey), (ex+ew, ey+eh), color)

        if self.status.begin_take_photo:
            print('Taking image')
            cv2.imwrite(f'../images/img_{now_str}.jpg', self.image.captured)
            self.status.completed = True
            self.status.restart = True
        
        if eyes_detected and mouth_detected:
            self.status.smile_detect += 1
            self.status.no_smile_detect = 0

            if self.status.smile_detect >= 25:
                self.status.face_found = True
            if self.status.smile_detect >= 50:
                self.status.smile_detected = True
            if self.status.smile_detect >= 100:
                print("Smile detected")
                self.status.begin_take_photo = True
        else:
            self.status.no_smile_detect += 1
            if self.status.no_smile_detect == 20:
                print("No smile was detected")
            if self.status.no_smile_detect > 50:
                self.status.restart = True
            
        if not self.status.begin_take_photo or len(faces) == 0:
            cv2.imshow('Smile detector :)', self.image.annotated)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.image.cap.release()
                cv2.destroyAllWindows()
                sys.exit()




def main():
    cap = cv2.VideoCapture(0)

    while True:
        status = SmileDetectStatus()
        while not status.begin_take_photo:
            status = SmileDetectStatus()
            image = Image(cap)
            detector = Detector(image, status)

            while not status.smile_detected:
                image.capture_image()
                detector.detect_smiles()
                if status.restart:
                    print("Restarting...")
                    break
            while status.smile_detected and not status.begin_take_photo:
                image.capture_image()
                detector.detect_smiles()
                if status.restart:
                    print("Restarting...")
                    break
        while not status.completed:
            image.capture_image()
            detector.detect_smiles()
            
            if status.restart:
                    print("Restarting...")
                    break



if __name__ == '__main__':
    main()
