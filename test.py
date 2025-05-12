import cv2
import pyttsx3
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
classifier = Classifier("Model/keras_model.h5", "Model/labels.txt")
labels = ["0","1","2","3","A", "B", "C","D"]

engine = pyttsx3.init()

while True:
    success, img = cap.read()
    hands, img = detector.findHands(img)
    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']
        prediction, index = classifier.getPrediction(img, draw=False)
        k = cv2.waitKey(1)
        if k == ord("s"):
            engine.say(labels[index])
            engine.runAndWait()
        cv2.putText(img, labels[index], (x + 70, y - 20), cv2.FONT_HERSHEY_COMPLEX, 2, (255, 0, 255), 2)
        #cv2.putText(prediction, labels[index], (x + 100, y - 20), cv2.FONT_HERSHEY_COMPLEX, 2, (255, 0, 255), 2)

    cv2.imshow("image", img)
    key = cv2.waitKey(1)
    if key == ord("q"):
        break
