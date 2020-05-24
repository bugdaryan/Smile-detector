import cv2
import numpy as np
import time
from haar_cascade import detect_smile

class SmileDetectStatus:
    def __init__(self):
        self.begin_detect = False
        self.face_detected = False
        self.smile_detected = False
        self.restart = False
        self.completed = False
        self.no_detect = 0
        self.detect = 0

class Image:
    def __init__(self, cap):
        self.cap = cap
    
    def capture_image(self):
        #capture image
        raise NotImplementedError()

class Detector:
    def __init__(self, image, status):
        self.image = image
        self.status = status

    def detect_face(self):
        #detect face
        #update status
        raise NotImplementedError()
    
    def detect_smile(self):
        #detect smile in faces
        #update status
        raise NotImplementedError()




def main():
    cap = cv2.VideoCapture(0)

    while True:
        status = SmileDetectStatus()
        while not status.begin_detect:
            status = SmileDetectStatus()
            image = Image(cap)
            detector = Detector(image, status)

            while not status.face_detected:
                image.capture_image(status)
                detector.detect_face()
                if status.restart:
                    break
            while status.face_detected and not status.smile_detected:
                image.capture_image()
                detector.detect_smile()
                if status.restart:
                    print("Restarting...")
                    break
                else:
                    print("Recheking for smile...")

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

if __name__ == '__main__':
    main()
