import re
import time
import os
import argparse
import json
import tensorflow as tf
import numpy as np
import rediswq
import mlflow
from tensorflow.keras.callbacks import ModelCheckpoint
from ModelConstructor import ModelConstructor


def find_argmax_by_attr(lst, attr):
    max_val = None
    max_obj = None
    max_idx = None
    for i, obj in enumerate(lst):
        val = getattr(obj, attr)
        if max_val is None or val > max_val:
            max_val = val
            max_obj = obj
            max_idx = i
    return max_idx, max_val, max_obj


def create_model(arc_json, nn_json):
    constructor = ModelConstructor(arc_json, nn_json)
    model = constructor.build_model()
    model.compile(loss=tf.keras.losses.categorical_crossentropy,
                  optimizer='adam',
                  metrics=['accuracy'])

    return model


def train_model(train_config):
    experiment_name = os.getenv('MLFLOW_EXPERIMENT_NAME')
    model_dir = os.getenv('MODEL_DIR')

    strategy = train_config.get('strategy')
    train_dataset = train_config.get('train_dataset')
    val_dataset = train_config.get('val_dataset')
    batch_size = train_config.get('batch_size')
    steps_per_epoch = train_config.get('steps_per_epoch')
    image_size = train_config.get('image_size')
    num_classes = train_config.get('num_classes')
    num_epochs = train_config.get('num_epochs')
    patience = num_epochs // 5

    run_name = train_config.get('run_name')
    architecture = train_config.get('architecture')
    nn_config = train_config.get('nn_config')
    print(
        f">>> Running AutoML with params: architecture: {architecture}, nn_config: {nn_config}")

    with strategy.scope():
        model = create_model(architecture, nn_config)
        model.summary()

        client = mlflow.MlflowClient()
        mlflow.start_run()

        # mlflow.log_param('arch', architecture)
        # mlflow.log_param('nn_config', nn_config)
        # Set run name
        run_id = mlflow.active_run().info.run_id
        mlflow.set_tag("mlflow.runName", run_name)

        checkpoint_path = f"gs://{model_dir}/{experiment_name}/{run_id}"
        checkpoint_callback = ModelCheckpoint(
            checkpoint_path, verbose=1, monitor='val_accuracy', save_best_only=True, mode='max')

        best_epoch = 0
        for e in range(num_epochs):
            print("\nEpoch {}/{}".format(e + 1, num_epochs))
            history = model.fit(train_dataset,
                                steps_per_epoch=steps_per_epoch,
                                epochs=1, verbose=1, batch_size=32,
                                validation_data=val_dataset, callbacks=[checkpoint_callback])

            metrics = {
                'train_loss': history.history['loss'][0],
                'train_accuracy': history.history['accuracy'][0],
                'val_loss': history.history['val_loss'][0],
                'val_accuracy': history.history['val_accuracy'][0],
            }
            mlflow.log_metrics(metrics)

            metric_history = client.get_metric_history(run_id, "val_accuracy")
            if history.history['val_accuracy'][0] > metric_history[best_epoch].value:
                best_epoch = e
            if e >= patience:
                monitor_history = metric_history[-patience:]
                max_idx, max_value, max_metric = find_argmax_by_attr(
                    monitor_history, 'value')
                if max_idx == 0:
                    print("Early stopping at epoch {}".format(e))
                    break
        mlflow.end_run()
        print('>>> Best epoch: ', best_epoch + 1)
