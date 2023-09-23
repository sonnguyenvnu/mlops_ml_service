import os

import mlflow
import requests
import tensorflow as tf

from train_util import load_datasets


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


def create_model(pretrained_model_name, num_classes, image_size):
    model_fn = getattr(tf.keras.applications, pretrained_model_name)
    pretrained_model = model_fn(weights='imagenet', input_shape=[
        *image_size, 3], include_top=False)

    pretrained_model.trainable = True

    model = tf.keras.Sequential([
        pretrained_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(
            num_classes, activation='softmax', dtype=tf.float32)
    ])

    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model


def save_model_temporary(model):
    save_locally = tf.saved_model.SaveOptions(
        experimental_io_device='/job:localhost')
    model.save('./best_model', options=save_locally)


def load_best_model():
    load_locally = tf.saved_model.LoadOptions(
        experimental_io_device='/job:localhost')
    model = tf.keras.models.load_model('./best_model', options=load_locally)
    return model


def save_run_info(run_info):
    url = f"{os.getenv('BACKEND_SERVICE_HOST')}/v1/runs"
    print('Payload ====>', run_info)

    response = requests.post(url, data=run_info)
    print('Response status code: ', response.status_code)


def train_model(train_config):
    # clear old session
    tf.keras.backend.clear_session()

    experiment_name = os.getenv('MLFLOW_EXPERIMENT_NAME')
    model_dir = os.getenv('MODEL_DIR')

    model_name = train_config.get('model_name')
    image_size = train_config.get('image_size')
    num_classes = train_config.get('num_classes')
    num_epochs = train_config.get('num_epochs')
    patience = num_epochs // 5

    try:
        tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
        print('Running on TPU ', tpu.cluster_spec().as_dict()['worker'])

        tf.config.experimental_connect_to_cluster(tpu)
        tf.tpu.experimental.initialize_tpu_system(tpu)
        strategy = tf.distribute.TPUStrategy(tpu)
    except ValueError:
        strategy = tf.distribute.MirroredStrategy()
    print("Number of accelerators: ", strategy.num_replicas_in_sync)

    print(
        f">>> Transfer learning with pretrained model: {model_name}, {num_epochs} epochs, {num_classes} classes, dataset_url: {train_config.get('dataset_url')}")

    with strategy.scope():
        print(train_config)
        train_dataset, val_dataset, batch_size, steps_per_epoch = load_datasets(train_config)

        model = create_model(model_name, num_classes, image_size)
        model.summary()

        client = mlflow.MlflowClient()
        mlflow.start_run()
        # Set run name
        run_id = mlflow.active_run().info.run_id
        run_name = f"{experiment_name}_pretrained_{model_name}"
        mlflow.set_tag("mlflow.runName", run_name)

        best_epoch = 0
        best_val_accuracy = 0
        # path to save model
        path = f"gs://{model_dir}/{experiment_name}/{run_id}"
        # dict: { epoch: metrics }
        history_data = {}
        for e in range(num_epochs):
            print("\nEpoch {}/{}".format(e + 1, num_epochs))
            history = model.fit(train_dataset,
                                steps_per_epoch=steps_per_epoch,
                                epochs=1, verbose=1, batch_size=batch_size,
                                validation_data=val_dataset)
            val_accuracy = history.history['val_accuracy'][0]
            metrics = {
                'train_loss': history.history['loss'][0],
                'train_accuracy': history.history['accuracy'][0],
                'val_loss': history.history['val_loss'][0],
                'val_accuracy': val_accuracy,
            }
            history_data[e] = metrics
            mlflow.log_metrics(metrics)

            metric_history = client.get_metric_history(run_id, "val_accuracy")
            if val_accuracy > best_val_accuracy:
                print(
                    f">>> {val_accuracy} > {best_val_accuracy}, save_model_temporary")
                save_model_temporary(model)
                best_val_accuracy = val_accuracy

            if history.history['val_accuracy'][0] > metric_history[best_epoch].value:
                best_epoch = e

            if e % 10 == 9:
                # save model
                model = load_best_model()
                print(f"Saving best model to {path}...")
                model.save(path)

                # save run info
                run_info = history_data.get(best_epoch)
                run_info['experiment_name'] = experiment_name
                run_info['run_id'] = run_id
                run_info['name'] = run_name
                run_info['best_model_url'] = path
                save_run_info(run_info)
            if e >= patience:
                monitor_history = metric_history[-patience:]
                max_idx, max_value, max_metric = find_argmax_by_attr(
                    monitor_history, 'value')
                if max_idx == 0:
                    print("Early stopping at epoch {}".format(e))
                    break

        mlflow.end_run()
        print('>>> Best epoch: ', best_epoch + 1)
