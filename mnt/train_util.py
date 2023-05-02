import json
import os
import time
import argparse
import re
import tensorflow as tf
import numpy as np

AUTOTUNE = tf.data.AUTOTUNE

def count_data_items(filenames):
    n = [int(re.compile(r"-([0-9]*)\.").search(filename).group(1))
         for filename in filenames]
    return np.sum(n)


def read_tfrecord(example, num_classes):
    features = {
        "image": tf.io.FixedLenFeature([], tf.string),
        "class": tf.io.FixedLenFeature([], tf.int64),
        "one_hot_class": tf.io.VarLenFeature(tf.float32),
    }
    example = tf.io.parse_single_example(example, features)
    image = tf.io.decode_jpeg(example['image'], channels=3)
    image = tf.cast(image, tf.float32) / 255.0
    class_label = tf.cast(example['class'], tf.int32)
    one_hot_class = tf.sparse.to_dense(example['one_hot_class'])
    one_hot_class = tf.reshape(one_hot_class, [num_classes])
    return image, one_hot_class


def force_image_sizes(dataset, image_size):
    def reshape_images(image, label): return (
        tf.reshape(image, [*image_size, 3]), label)
    dataset = dataset.map(reshape_images, num_parallel_calls=AUTOTUNE)
    return dataset


def load_dataset(filenames, num_classes, image_size):
    opt = tf.data.Options()
    opt.experimental_deterministic = False

    dataset = tf.data.Dataset.from_tensor_slices(filenames)
    dataset = dataset.with_options(opt)
    dataset = tf.data.TFRecordDataset(filenames, num_parallel_reads=AUTOTUNE)
    dataset = dataset.map(lambda example: (example, num_classes))
    dataset = dataset.map(read_tfrecord, num_parallel_calls=AUTOTUNE)
    dataset = force_image_sizes(dataset, image_size)

    return dataset


def data_augment(image, one_hot_class):
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_brightness(image, max_delta=0.2)
    image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
    image = tf.image.random_hue(image, max_delta=0.2)
    image = tf.image.random_saturation(image, 0, 2)
    return image, one_hot_class


def calculate_batch_size(num_images, automl=False):
    batch_size = 16
    if not automl:
        if num_images > 100 and num_images < 500:
            batch_size = 32
        if num_images > 500 and num_images < 1000:
            batch_size = 64
        if num_images > 1000:
            batch_size = 128
    steps_per_epoch = int(num_images // batch_size) + 1
    print("{} images will be trained with batch size = {}, steps = {}".format(
        num_images, batch_size, steps_per_epoch))
    return batch_size, steps_per_epoch

def load_datasets(train_config, automl=False):
    dataset_url = train_config.get('dataset_url')
    validation_split = train_config.get('validation_split')
    num_classes = train_config.get('num_classes')
    image_size = train_config.get('image_size')

    filenames = tf.io.gfile.glob(dataset_url)
    num_images = count_data_items(filenames)
    train_images = int(num_images * (1 - validation_split))
    train_config['num_images'] = num_images
    train_config['train_images'] = train_images
    print('>>> Total images: {}, Train images: {}'.format(num_images, train_images))
    batch_size, steps_per_epoch = calculate_batch_size(train_images, automl=automl)

    dataset = load_dataset(filenames, num_classes, image_size)
    train_dataset = dataset.take(train_images)
    val_dataset = dataset.skip(train_images)

    train_dataset = train_dataset.map(
        data_augment, num_parallel_calls=AUTOTUNE)
    train_dataset = train_dataset.repeat()
    # train_dataset = train_dataset.shuffle(2048)
    train_dataset = train_dataset.batch(batch_size)
    train_dataset = train_dataset.prefetch(AUTOTUNE)

    val_dataset = val_dataset.batch(batch_size)
    val_dataset = val_dataset.prefetch(AUTOTUNE)

    opt = tf.data.Options()
    opt.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
    val_dataset = val_dataset.with_options(opt)
    
    return train_dataset, val_dataset, batch_size, steps_per_epoch

# train config for both pre-trained models and automl
def get_train_config(metadata, cached_config={}, automl=False):
    config = cached_config.get(metadata.get('dataset_url'))
    if config is not None:
        print('>>> Received train config from cache, key:',
              metadata.get('dataset_url'))
        config['model_name'] = metadata.get('model_name')

        # for automl
        config['run_name'] = metadata.get('run_name')
        config['architecture'] = metadata.get('architecture')
        config['nn_config'] = metadata.get('nn_config')

        return config, cached_config

    dataset_url = metadata.get('dataset_url')
    target_size = metadata.get('target_size')
    num_classes = metadata.get('num_classes')
    IMAGE_SIZE = [target_size, target_size]

    train_config = {}
    train_config['validation_split'] = 0.2
    train_config['dataset_url'] = dataset_url
    train_config['target_size'] = target_size
    train_config['num_classes'] = num_classes
    train_config['num_epochs'] = metadata.get('num_epochs')
    train_config['model_name'] = metadata.get('model_name')
    train_config['image_size'] = IMAGE_SIZE

    # for automl
    train_config['run_name'] = metadata.get('run_name')
    train_config['architecture'] = metadata.get('architecture')
    train_config['nn_config'] = metadata.get('nn_config')

    cached_config[dataset_url] = train_config
    return train_config, cached_config