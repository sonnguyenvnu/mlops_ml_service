import argparse
import os
import re
import json
import time
import mlflow
import rediswq

# Main
parser = argparse.ArgumentParser(description='TrainingContainer')
parser.add_argument('--architecture', type=str, default="", metavar='N',
                    help='architecture of the neural network')
parser.add_argument('--nn_config', type=str, default="", metavar='N',
                    help='configurations and search space embeddings')
parser.add_argument('--num_epochs', type=int, default=100, metavar='N',
                    help='number of epoches that each child will be trained')
parser.add_argument('--target_size', type=int, default=224, metavar='N',
                    help='target size of image in dataset')
parser.add_argument('--dataset_url', type=str, default=1, metavar='N',
                    help='dataset url')
parser.add_argument('--num_classes', type=int, default=2, metavar='N',
                    help='numer of classes')

args = parser.parse_args()

# Convert args
# architecture, nn_config is JSON format
arch = args.architecture.replace("\'", "\"")
print(">>> arch received by trial")
print(arch)
arch_value = json.loads(arch)

nn_config = args.nn_config.replace("\'", "\"")
print(">>> nn_config received by trial")
print(nn_config)
nn_config_value = json.loads(nn_config)

redis_host = os.getenv('REDIS_SERVICE_HOST')
queue_name = os.getenv('QUEUE_NAME')
experiment_name = os.getenv('MLFLOW_EXPERIMENT_NAME')

# push one job to queue
q = rediswq.RedisWQ(name=queue_name, host=redis_host)
print("Worker with sessionID: " + q.sessionID())
print("Pushing job to queue...")

time_str = str(int(time.time()))
run_name = f"{experiment_name}_automl_{time_str}"
job = json.dumps({
    'run_name': run_name,
    'architecture': arch_value,
    'nn_config': nn_config_value,
    'num_epochs': args.num_epochs,
    'target_size': args.target_size,
    'dataset_url': args.dataset_url,
    'num_classes': args.num_classes,
})

q.put(job)
print('>>> Pushed job to queue. Waiting for train result...')

filter_string = f"tags.mlflow.runName = '{run_name}'"
client = mlflow.MlflowClient(tracking_uri=os.getenv('MLFLOW_TRACKING_URI'))


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


while True:
    runs = mlflow.search_runs(
        experiment_names=[experiment_name],
        filter_string=filter_string,
        order_by=["metrics.val_accuracy DESC"],
        run_view_type=mlflow.entities.ViewType.ACTIVE_ONLY,
        max_results=1,
    )
    if len(runs) > 0:
        run = runs.iloc[0]
        if run.status != 'RUNNING':
            run_id = run.run_id
            val_accuracy_history = client.get_metric_history(
                run_id, "val_accuracy")
            accuracy_history = client.get_metric_history(
                run_id, "train_accuracy")
            val_loss_history = client.get_metric_history(run_id, "val_loss")
            loss_history = client.get_metric_history(run_id, "train_loss")

            max_idx, best_val_accuracy, best_val_accuracy_metric = find_argmax_by_attr(
                val_accuracy_history, 'value')
            best_epoch = best_val_accuracy_metric.step

            print(
                "Training-Accuracy={}".format(accuracy_history[best_epoch].value))
            print("Training-Loss={}".format(loss_history[best_epoch].value))
            # Objective metric, used to estimate performace
            print("Validation-Accuracy={}".format(best_val_accuracy))
            print(
                "Validation-Loss={}".format(val_loss_history[best_epoch].value))
            break
    time.sleep(10)
