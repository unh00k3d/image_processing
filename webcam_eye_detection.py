import cv2
import numpy as np

# video_stream= "http://192.168.1.137:8080/video"
video_stream = 0

def main():
    capture = cv2.VideoCapture(video_stream)

    cascade_classifier = cv2.CascadeClassifier("assets/cascades/haarcascade_eye.xml")

    while True:
        ret, frame = capture.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = cascade_classifier.detectMultiScale(gray, 1.1, 6)

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        cv2.imshow("Frame", frame)
        if cv2.waitKey(1)== ord('q'):
            break


if __name__ == '__main__':
    main()
    cv2.destroyAllWindows()