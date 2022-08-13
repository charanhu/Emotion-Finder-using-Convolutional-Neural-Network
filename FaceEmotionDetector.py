import cv2
import numpy as np
from keras.models import model_from_json

emotions_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}

#load json and create model
json_file = open('savedModel/EmotionFinderModel.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
emotion_finder_model = model_from_json(loaded_model_json)

#load weights into new model
emotion_finder_model.load_weights("savedModel/EmotionFinderModel.h5")
print("Loaded model from disk")

# testing by live video
#cap = cv2.VideoCapture(0)

# test by passing video
cap=cv2.VideoCapture('emotion_sample_video.mp4')

while True:
    ret, frame = cap.read()
    frame = cv2.resize(frame, (1280, 720))    #resize frame (optional)
    if not ret:
        break
    face_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # detect faces in the grayscale frame
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    # loop over the faces and draw a rectangle around each
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y-50), (x + w, y + h+10), (0, 255, 0), 4)
        roi_gray = gray[y:y + h, x:x + w]
        cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray, (48, 48)), -1), 0)

        # predict the emotion
        prediction = emotion_finder_model.predict(cropped_img)
        maxindex = int(np.argmax(prediction))
        cv2.putText(frame, emotions_dict[maxindex], (x+5, y-20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

    cv2.imshow('Emotion Finder', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()