import numpy as np
import cv2


labels = ['vladimir putin', 'kim jong un', 'donald trump', 'erdogan', 'joe biden']

cascade_classifier = cv2.CascadeClassifier('assets/cascades/haarcascade_frontalface_default.xml')
image_path = 'assets/validation/erdogan-5.jpeg'
image = cv2.imread(image_path)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
faces = cascade_classifier.detectMultiScale(gray, 1.3, 5)
face_recognizer = cv2.face.LBPHFaceRecognizer_create()
face_recognizer.read('assets/trained_model.yml')
if len(faces) == 0:
    print("No faces found")
    exit()

for (x,y,w,h) in faces:
    face = gray[y:y+h, x:x+w]
    label , confidence = face_recognizer.predict(face)
    print(f"Label = {label} with confidence of {round(confidence,ndigits=2)}%")
    cv2.putText(image ,labels[label],(20,20),cv2.FONT_HERSHEY_COMPLEX,1.0,(0,255,0),thickness= 2)
    cv2.rectangle(image, (x,y), (x+w, y+h), (0,255,0), 2)


cv2.imshow('Detected faces', image)

cv2.waitKey(0)
cv2.destroyAllWindows()





# the model isn't really good and should be improved later with tensorflow or somehting else