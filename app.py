import os
from uuid import uuid1
import jwt
from flask_cors import CORS, cross_origin
import time
from huggingface_hub.utils import BadRequestError


def current_milli_time():
    return round(time.time() * 1000)


secret_key = "798a2278-7baa-11ee-b962-0242ac120002"

import mysql.connector

from flask import Flask, render_template, request, redirect, url_for, make_response
import cv2
import numpy as np
from keras.models import model_from_json
from google.oauth2 import id_token
from google.auth.transport import requests
from flask import send_from_directory

app = Flask(__name__)

cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}

# load json and create models
json_file = open('models/emotion_model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
emotion_model = model_from_json(loaded_model_json)

# load weights into new models
emotion_model.load_weights("models/emotion_model.h5")
print("Loaded models from disk")

# diskhog = cv2.HOGDescriptor()
# hog.setSVMDetector(cv2.HOGDescriptor.getDefaultPeopleDetector())

face_detector = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')

connection = mysql.connector.connect(
    host="localhost",
    user="root",
    password="hello@123",
    database="face_detection"
)


@app.route('/auth', methods=['POST'])
@cross_origin()
def auth():
    try:
        cursor = connection.cursor(buffered=True)
        idInfo = id_token.verify_oauth2_token(request.json['response']['credential'], requests.Request(),
                                              "534042040906-qff5ftkdcklm1sfrs99ct4s794gkcvbv.apps.googleusercontent.com")
        userid = str(uuid1())
        email = idInfo['email']
        name = idInfo['name']
        cursor.execute("SELECT * FROM Users WHERE email = %s", (email,))
        if cursor.rowcount > 0:
            print("user exists")
        else:
            sql = """INSERT INTO Users (user_id, email, name) VALUES (%s, %s, %s)"""
            cursor.execute(sql, (userid, email, name))
            connection.commit()
        cursor.close()
        encoded_token = jwt.encode(payload={"email": email, "exp": current_milli_time() + 1800000},
                                   key=secret_key,
                                   algorithm="HS256")
        response = {"token": encoded_token}
        return response
    except ValueError:
        pass
    except Exception as e:
        print(e)
    # Invalid token


# Emotion detection route
@app.route('/', methods=['POST'])
def index():
        if request.method == 'POST':
            if 'my-image-file' not in request.files:
                raise BadRequestError("Invalid request")

            file = request.files['my-image-file']

            if file.filename == '':
                return redirect(request.url)

            if file:
                image_path = os.path.join("static/uploads", file.filename)
                file.save(image_path)
                output_path = predict_emotion(image_path)
                response = {"url": "http://localhost:5000/" + output_path}
                return response


def predict_emotion(path):
    frame = cv2.imread(path, cv2.IMREAD_COLOR)
    frame = cv2.resize(frame, (1280, 720))

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # detect faces available on camera
    num_faces = face_detector.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)

    # take each face available on the camera and Preprocess it
    for (x, y, w, h) in num_faces:
        cv2.rectangle(frame, (x, y - 50), (x + w, y + h + 10), (0, 255, 0), 4)
        roi_gray_frame = gray_frame[y:y + h, x:x + w]
        cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray_frame, (48, 48)), -1), 0)

        # predict the emotions
        emotion_prediction = emotion_model.predict(cropped_img)
        maxindex = int(np.argmax(emotion_prediction))
        cv2.putText(frame, emotion_dict[maxindex], (x + 5, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2,
                    cv2.LINE_AA)
    output_path = os.path.join('static/uploads', str(uuid1()) + ".png")
    cv2.imwrite(output_path, frame)

    return output_path


if __name__ == '__main__':
    app.run(debug=True)
