import jwt
import numpy as np
import cv2
from keras.models import model_from_json
from flask import Flask, request, redirect
from flask_cors import CORS, cross_origin
from google.auth.transport import requests
from google.oauth2 import id_token
from huggingface_hub.utils import BadRequestError
import mysql.connector
import smtplib
import time
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from uuid import uuid1
import redis
import json
import os

haar_file = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(haar_file)

server_email = "emotiondetectionserver@gmail.com"
server_password = "iynztbhliitypkxn"
secret_key = "798a2278-7baa-11ee-b962-0242ac120002"

app = Flask(__name__)
CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

labels = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}
json_file = open('models/emotion_model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
emotion_model = model_from_json(loaded_model_json)

emotion_model.load_weights("models/emotion_model.h5")
print("Loaded models from disk")

connection = mysql.connector.connect(
    host="localhost",
    user="root",
    password="hello@123",
    database="face_detection"
)


def current_milli_time():
    return round(time.time() * 1000)


# redis_client = redis.StrictRedis(host='localhost', port=6379, db=0, password='your_password')
r = redis.Redis()


def get_data_from_database(email):
    # Check if data exists in cache
    cached_data = r.get(email)
    if cached_data:
        return json.loads(cached_data)
        # return cached_data.decode('utf-8')  # Data found in cache

    # If data is not cached, fetch it from the database
    data = fetch_data_from_database(email)
    cachedData = json.dumps(data)
    # Cache the fetched data in Redis
    r.set(email, cachedData)

    return data


def fetch_data_from_database(email):
    cursor = connection.cursor(buffered=True)
    cursor.execute("SELECT * FROM Users WHERE email = %s", (email,))
    cursor.close()
    return cursor.fetchone();


def send_email(server_email, server_password, receiver_email, subject, body):
    message = MIMEMultipart()
    message['From'] = server_email
    message['To'] = receiver_email
    message['Subject'] = subject

    message.attach(MIMEText(body, 'plain'))

    # Connect to the SMTP server
    smtp_server = 'smtp.gmail.com'
    smtp_port = 587  # TLS port
    server = smtplib.SMTP(smtp_server, smtp_port)
    server.starttls()  # Start TLS encryption

    # Log in to the SMTP server
    server.login(server_email, server_password)

    # Send the email
    server.sendmail(server_email, receiver_email, message.as_string())

    # Quit the SMTP session
    server.quit()


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
@cross_origin()
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
            decoded_token = jwt.decode(request.headers["token"], options={"verify_signature": False})
            email = decoded_token["email"]
            user_name = get_data_from_database(email)[3]
            response = {"url": "http://localhost:5000/" + output_path}
            subject = "Emotion Detection Success:" + user_name
            body = "The emotion of your face has been determined successfully.Please find attached the link to detected emotion on image  " + "http://localhost:5000/" + output_path
            send_email(server_email, server_password, email, subject, body)
            return response


# def predict_emotion(path):
#     frame = cv2.imread(path, cv2.IMREAD_COLOR)
#     # frame = cv2.resize(frame, (1280, 720))
#
#     gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#
#     # detect faces available on camera
#     num_faces = face_detector.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)
#     # num_faces = detector(gray_frame)
#
#     # num_faces = face_recognition.face_locations(gray_frame)
#
#     # take each face available on the camera and Preprocess it
#     for (x, y, w, h) in num_faces:
#         cv2.rectangle(frame, (x, y - 50), (x + w, y + h + 10), (0, 255, 0), 4)
#         roi_gray_frame = gray_frame[y:y + h, x:x + w]
#         cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray_frame, (48, 48)), -1), 0)
#
#         # predict the emotions
#         emotion_prediction = emotion_model.predict(cropped_img)
#         maxindex = int(np.argmax(emotion_prediction))
#         cv2.putText(frame, emotion_dict[maxindex], (x + 5, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2,
#                     cv2.LINE_AA)
#     output_path = os.path.join('static/uploads', str(uuid1()) + ".png")
#     cv2.imwrite(output_path, frame)
#
#     return output_path


def extract_features(image):
    feature = np.array(image)
    feature = feature.reshape(1, 48, 48, 1)
    return feature / 255.0


def predict_emotion(path):
    frame = cv2.imread(path, cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    try:
        for (p, q, r, s) in faces:
            image = gray[q:q + s, p:p + r]
            cv2.rectangle(frame, (p, q), (p + r, q + s), (255, 0, 0), 2)
            image = cv2.resize(image, (48, 48))
            img = extract_features(image)
            pred = emotion_model.predict(img)
            prediction_label = labels[pred.argmax()]
            # print("Predicted Output:", prediction_label)
            # cv2.putText(im,prediction_label)
            cv2.putText(frame, '% s' % (prediction_label), (p - 10, q - 10), cv2.FONT_HERSHEY_COMPLEX_SMALL, 3,
                        (0, 0, 255))
        # cv2.imshow("Output",frame)
        output_path = os.path.join('static/uploads', str(uuid1()) + ".png")
        cv2.imwrite(output_path, frame)
        cv2.waitKey(27)
        return output_path
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


if __name__ == '__main__':
    app.run(debug=True)
