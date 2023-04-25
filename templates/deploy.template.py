import tensorflow as tf
import numpy as np
from PIL import Image
from flask import Flask, request
from flask_cors import CORS
import os, io

# Set env or mount to container
model = tf.keras.models.load_model(os.getenv('MODEL_DIR'))
print('>>> Model loaded successfully')

# example: 'ants, bees' => ['ants', 'bees']
CLASSES = os.getenv('CLASSES').split(' ')
target_size = int(os.getenv('TARGET_SIZE'))
IMAGE_SIZE = [target_size, target_size]

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

@app.route("/")
def hello_world():
    return "<p>Hello from predictor!!!</p>"

@app.route('/cors', methods=['POST'])
def check_cors():
    return { 'status': 'OK' }

@app.route('/predict', methods=['POST'])
def predict():
    predictions = []
    for key in request.files:
        image = request.files[key]
        img = Image.open(io.BytesIO(image.read()))
        img = np.array(img) / 255.0
        img = tf.image.resize(img, [*IMAGE_SIZE])
        img = np.expand_dims(img, (0,)).astype("float32")
        pred = model.predict(img)

        # Get the predicted class and confidence
        pred_class = CLASSES[np.argmax(pred)]
        pred_conf = np.max(pred)

        # Add the prediction to the list
        predictions.append({'key': key, 'class': pred_class, 'confidence': float(pred_conf)})
    return {'predictions': predictions}

if __name__ == '__main__':
    app.run(port=4000,host='0.0.0.0')