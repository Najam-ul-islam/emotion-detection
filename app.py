from flask import Flask, request, jsonify
import cv2
import numpy as np
import base64
from tensorflow.keras.models import load_model
app = Flask(__name__)

app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 1
label_map = {"Emotions":['Anger', 'Neutral', 'Fear', 'Happy', 'Sad', 'Surprise']}


@app.route('/response', methods=['POST'])
def response():
    json1 = request.get_json()
    data = json1['content']
    with open("imageToSave.png", "wb") as fh:
        fh.write(base64.b64decode(data))
    img = cv2.imread('imageToSave.png')
    grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(grey, (48, 48))
    img = np.reshape(img, (1, 48, 48, 1))
    model = load_model('model.h5')
    pred = model.predict(img)[0]
    pred = np.argmax(pred)
    return jsonify({"Detected Emotion": np.float(pred.__str__()), "emotions": label_map})


if __name__ == "__main__":
    app.run(debug=True)