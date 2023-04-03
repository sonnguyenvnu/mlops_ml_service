import json
import os
import time
import argparse
import re
import rediswq
from pretrained import train_model
from train_util import get_train_config
import tensorflow as tf
import numpy as np

AUTOTUNE = tf.data.AUTOTUNE
parser = argparse.ArgumentParser(description='Training container')
parser.add_argument('--redis_host', type=str, default="redis", metavar='N',
                    help='redis host contains job queue')
parser.add_argument('--queue_name', type=str, default="job2", metavar='N',
                    help='queue name')
args = parser.parse_args()

# MLFlow
print('MLFLOW_TRACKING_URI:', os.getenv('MLFLOW_TRACKING_URI'))
print('MLFLOW_TRACKING_USERNAME:', os.getenv('MLFLOW_TRACKING_USERNAME'))
print('MLFLOW_TRACKING_PASSWORD:', os.getenv('MLFLOW_TRACKING_PASSWORD'))
print('GOOGLE_APPLICATION_CREDENTIALS:',
      os.getenv('GOOGLE_APPLICATION_CREDENTIALS'))

q = rediswq.RedisWQ(name=args.queue_name, host=args.redis_host)
print("Worker with sessionID: " + q.sessionID())
print("Initial queue state: empty=" + str(q.empty()))

start_time = time.time()
# wait for jobs up to 5 min
while q.empty():
  wait_time = time.time() - start_time
  if wait_time > 300:  # 300s
    break

cached_config = {}
while not q.empty():
  item = q.lease(lease_secs=10, block=True, timeout=2)
  if item is not None:
    itemstr = item.decode("utf-8")
    job = json.loads(itemstr)
    print('==================================')
    print("\n>>> Working on " + job.get('model_name'))
    train_config, cached_config = get_train_config(job, cached_config)
    train_model(train_config)
    # cmd = f"python3 ./train.py --model_name={job['model_name']} --num_epochs={job['num_epochs']} --dataset_url={job['dataset_url']} --num_classes={job['num_classes']}"
    # os.system(cmd)
    q.complete(item)
  else:
    print("Waiting for work")
print("Queue empty, exiting")
