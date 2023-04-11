import json
import os
import time
import argparse
import re
import rediswq
import mlflow

parser = argparse.ArgumentParser(description='Training container')
parser.add_argument('--redis_host', type=str, default="redis", metavar='N',
                    help='redis host contains job queue')
parser.add_argument('--queue_name', type=str, default="automl", metavar='N',
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

while not q.empty():
  item = q.lease(lease_secs=10, block=True, timeout=2)
  if item is not None:
    itemstr = item.decode("utf-8")
    job = json.loads(itemstr)
    print('Job:', itemstr)
    print('==================================')
    print(">>> Working on: " + job.get('run_name'))
    run_name = job.get('run_name')

    mlflow.start_run()
    mlflow.set_tag("mlflow.runName", run_name)
    time.sleep(20)
    metrics = {
        'train_loss': 0.0123,
        'train_accuracy': 0.9612,
        'val_loss': 0.932211,
        'val_accuracy': 0.76832,
    }
    mlflow.log_metrics(metrics)
    mlflow.log_metrics(metrics)
    mlflow.log_metrics(metrics)
    mlflow.end_run()

    q.complete(item)
  else:
    print("Waiting for work")
print("Queue empty, exiting")
