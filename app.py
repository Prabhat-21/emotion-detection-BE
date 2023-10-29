import os
from flask import Flask, render_template, request, redirect, url_for
import cv2
import numpy as np
from keras.models import model_from_json
from flask import send_from_directory

app = Flask(__name__)

emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}

# load json and create models
json_file = open('models/emotion_model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
emotion_model = model_from_json(loaded_model_json)

# load weights into new models
emotion_model.load_weights("models/emotion_model.h5")
print("Loaded models from disk")

face_detector = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')


# Emotion detection route
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)

        file = request.files['file']

        if file.filename == '':
            return redirect(request.url)

        if file:
            image_path = os.path.join("static/uploads", file.filename)
            file.save(image_path)
            output_path = predict_emotion(image_path)

            return render_template('index.html', image_path=output_path, emotion="Detected Success")

    return render_template('index.html', image_path=None, emotion=None)


def predict_emotion(path):
    frame = cv2.imread(path, cv2.IMREAD_COLOR)
    # if not ret:
    #     break
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
    output_path = os.path.join('static/uploads', 'output.png')
    cv2.imwrite(output_path, frame)

    return output_path


if __name__ == '__main__':
    app.run(debug=True)
