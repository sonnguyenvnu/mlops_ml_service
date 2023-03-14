# -*- coding: utf-8 -*-
import tensorflow as tf
import tensorflow_datasets as tfds
import argparse

parser = argparse.ArgumentParser(description='TrainingContainer')
parser.add_argument('--model_name', type=str, default="MobileNetV2", metavar='N',
                        help='name of pre-trained model to fine tuning')
parser.add_argument('--num_epochs', type=int, default=6, metavar='N',
                        help='numer of training epochs')
parser.add_argument('--dataset_url', type=str, default="", metavar='N',
                        help='url of dataset (GCS pattern, public visible)')

args = parser.parse_args()
model_name = args.model_name.replace("\'", "\"")
print('Model name: ', model_name)

num_epochs = args.num_epochs
print('Num epochs: ', num_epochs)

dataset_url = args.dataset_url.replace("\'", "\"")
print('Dataset URL: ', dataset_url)

(ds_train, ds_test), ds_info = tfds.load(
    'mnist',
    split=['train', 'test'],
    shuffle_files=True,
    as_supervised=True,
    with_info=True,
)

def normalize_img(image, label):
  """Normalizes images: `uint8` -> `float32`."""
  return tf.cast(image, tf.float32) / 255., label

ds_train = ds_train.map(
    normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
ds_train = ds_train.cache()
ds_train = ds_train.shuffle(ds_info.splits['train'].num_examples)
ds_train = ds_train.batch(128)
ds_train = ds_train.prefetch(tf.data.AUTOTUNE)

ds_test = ds_test.map(
    normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
ds_test = ds_test.batch(128)
ds_test = ds_test.cache()
ds_test = ds_test.prefetch(tf.data.AUTOTUNE)

model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dense(10)
])
model.compile(
    optimizer=tf.keras.optimizers.Adam(0.001),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
)

model.fit(
    ds_train,
    epochs=num_epochs,
    validation_data=ds_test,
)
