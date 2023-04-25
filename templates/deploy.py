import tensorflow as tf
import numpy as np
from PIL import Image
from flask import Flask, request
import os

os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = './service-account-gcs.json'

model = tf.keras.models.load_model(
    'gs://uet-mlflow/example_experiment/06d964a0070f45709edcfabb9854fd62/models/ckpt_epoch_7')


def predict_image(path):
  CLASSES = ['daisy', 'dandelion', 'roses', 'sunflowers', 'tulips']
  IMAGE_SIZE = [224, 224]

  img = Image.open(path)
  img = np.array(img)/255.0
  img = tf.image.resize(img, [*IMAGE_SIZE])
  img = np.expand_dims(img, (0,)).astype("float32")

  # model = create_model()
  y_pred = model.predict(img, verbose=0)
  # print(y_pred)
  conf = np.max(y_pred)

  return CLASSES[np.argmax(y_pred)], conf


folder = './img'
for path in os.listdir(folder):
  label, conf = predict_image(f"{folder}/{path}")
  print(path, label, conf)
