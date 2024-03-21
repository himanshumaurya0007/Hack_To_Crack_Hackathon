from flask import Flask, render_template, request, redirect, url_for, flash, Response
from flask_sqlalchemy import SQLAlchemy
import cv2
from tensorflow import keras
from keras.models import model_from_json
from keras.preprocessing.image import img_to_array
import numpy as np
import time

app = Flask(__name__)

# ********************************** emotion detection **********************************
# Load model
emotion_dict = {0: 'angry', 1: 'happy', 2: 'neutral', 3: 'sad', 4: 'surprise'}
json_file = open('static/emotion_detection/emotion_model1.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
classifier = model_from_json(loaded_model_json)
classifier.load_weights("static/emotion_detection/emotion_model1.h5")

# Load face
try:
    face_cascade = cv2.CascadeClassifier('static/emotion_detection/haarcascade_frontalface_default.xml')
except Exception:
    print("Error loading cascade classifiers")

class VideoCamera(object):
    def __init__(self):
        self.video = cv2.VideoCapture(0)
        self.start_time = time.time()

    def __del__(self):
        self.video.release()

    def get_frame(self):
        current_time = time.time()
        elapsed_time = current_time - self.start_time
        if elapsed_time >= 5:
            return None

        success, frame = self.video.read()
        if not success:
            return None

        img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(img_gray, scaleFactor=1.3, minNeighbors=5)

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            roi_gray = img_gray[y:y+h, x:x+w]
            roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)

            if np.sum([roi_gray]) != 0:
                roi = roi_gray.astype('float') / 255.0
                roi = img_to_array(roi)
                roi = np.expand_dims(roi, axis=0)
                prediction = classifier.predict(roi)[0]
                maxindex = int(np.argmax(prediction))
                finalout = emotion_dict[maxindex]
                output = str(finalout)

            label_position = (x, y)
            cv2.putText(frame, output, label_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        ret, jpeg = cv2.imencode('.jpg', frame)
        return jpeg.tobytes()


# Index page
@app.route('/')
def index():
    return render_template('index.html')

# Sign in page
@app.route('/signin')
def signin():
    return render_template('signin.html')

# Sign up page
@app.route('/signup')
def signup():
    return render_template('signup.html')

# Forgot password page
@app.route('/forgotpassword')
def forgotpassword():
    return render_template('forgotpassword.html')

# Home page
@app.route('/home')
def home():
    return render_template('home.html')

# Chatbot page
@app.route('/chatbot')
def chatbot():
    return render_template('chatbot.html')

# ********************************** Due to emotion detection **********************************
def gen(camera):
    while True:
        frame = camera.get_frame()
        if frame is None:
            break
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

# ********************************** Due to emotion detection **********************************
@app.route('/video_feed_emotion')
def video_feed_emotion():
    return Response(gen(VideoCamera()), mimetype='multipart/x-mixed-replace; boundary=frame')

# Emotion detection page
@app.route('/emotiondetection')
def emotiondetection():
    
    return render_template('emotiondetection.html')

if __name__ == '__main__':
    app.run(debug=True)
