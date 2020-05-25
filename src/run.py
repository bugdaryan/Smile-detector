import cv2
import numpy as np
import time
import sys
from haar_cascade import detect_faces,detect_smiles

class SmileDetectStatus:
    def __init__(self):
        self.begin_detect = False
        self.face_detected = False
        self.smile_detected = False
        self.restart = False
        self.completed = False
        self.no_face_detect = 0
        self.face_detect = 0
        self.no_smile_detect = 0
        self.smile_detect = 0

class Image:
    def __init__(self, cap):
        self.cap = cap
    
    def capture_image(self):
        ret, img = self.cap.read()
        self.captured = cv2.flip(img, 1)

class Detector:
    def __init__(self, image, status):
        self.image = image
        self.status = status

    def detect_faces(self):
        faces = detect_faces(self.image.captured)

        if len(faces) > 0:
            self.image.faces = faces
            self.status.face_detect += 1
            self.status.no_face_detect = 0

            if self.status.completed:
                cv2.imwrite(f'../images/img_{time.time()}', self.status.captured)
            elif self.status.face_detected:
                for (x,y,w,h) in self.image.faces:
                    cv2.rectangle(self.image.captured, (x, y), (x+w,y+h), (0, 255,0), 2)
            else:
                for (x,y,w,h) in self.image.faces:
                    cv2.rectangle(self.image.captured, (x, y), (x+w,y+h), (0, 255, 255), 2)
        else:
            self.status.no_face_detect += 1
            if self.status.no_face_detect == 20:
                print("No face was detected")
            if self.status.no_face_detect > 50:
                self.status.restart = True
        
        if self.status.face_detect == 25:
            self.status.face_detected = True
        if not self.status.begin_detect or len(faces) == 0:
            cv2.imshow('Smile detector :)', self.image.captured)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                sys.exit()
    
    def detect_smiles(self):
        smiles = detect_smiles(self.image.captured, self.image.faces)
        if len(smiles) > 0:
            self.image.smiles = smiles
            self.status.smile_detect += 1
            self.status.no_smile_detect = 0

            if self.status.completed:
                cv2.imwrite(f'../images/img_{time.time()}', self.image.captured)
            elif self.status.smile_detected:
                for (x,y,w,h) in self.image.faces:
                    cv2.rectangle(self.image.captured, (x, y), (x+w,y+h), (0, 255,0), 2)
            else:
                for (x,y,w,h) in self.image.faces:
                    cv2.rectangle(self.image.captured, (x, y), (x+w,y+h), (0, 255, 255), 2)
        else:
            self.status.no_face_detect += 1
            if self.status.no_face_detect == 20:
                print("No face was detected")
            if self.status.no_face_detect > 50:
                self.status.restart = True
        
        if self.status.face_detect == 25:
            self.status.face_detected = True

        if not self.status.begin_detect or len(smiles) == 0:
            cv2.imshow('Smile detector :)', self.image.captured)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                sys.exit()
        
        return
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
                image.capture_image()
                detector.detect_faces()
                if status.restart:
                    break
            while status.face_detected and not status.smile_detected:
                image.capture_image()
                detector.detect_faces()
                detector.detect_smiles()
                if status.restart:
                    print("Restarting...")
                    break
                else:
                    print("Recheking for smile...")

        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break

        # print("Starting detection...")
        # start_time = time.time()



    #     ret, img = cap.read()
    #     img = cv2.flip(img, 1)

    #     annotated = np.copy(img)
    #     smile_detected = detect_smile(annotated)

    #     if smile_detected:
    #     # smile_detected = True
    #         cv2.putText(img,"SMILE DETECTED", (int(0.1*img.shape[1]),int(0.9*img.shape[0])), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,0,0),4)

    #     cv2.imshow('annotated',annotated)
    #     cv2.imshow('img',img)
    #     if cv2.waitKey(1) & 0xFF == ord('q'):
    #         break

    # cap.release()
    # cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
