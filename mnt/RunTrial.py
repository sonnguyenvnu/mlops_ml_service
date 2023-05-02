import argparse
import os
import re
import json
import time
import mlflow
import rediswq
import requests

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

time.sleep(300) # 5 mins

url = f"{os.getenv('BACKEND_SERVICE_HOST')}/v1/runs?name={run_name}"
while True:
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        print("Training-Accuracy={}".format(data.get('train_accuracy')))
        print("Training-Loss={}".format(data.get('train_loss')))
        # Objective metric, used to estimate performace
        print("Validation-Accuracy={}".format(data.get('val_accuracy')))
        print("Validation-Loss={}".format(data.get('val_loss')))
        break
    time.sleep(10)
