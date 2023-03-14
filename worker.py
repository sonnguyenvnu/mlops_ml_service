import json, os
import rediswq
import argparse

parser = argparse.ArgumentParser(description='TrainingContainer')
parser.add_argument('--redis_host', type=str, default="", metavar='N',
                        help='redis host contains job queue')
parser.add_argument('--queue_name', type=str, default="", metavar='N',
                        help='queue name')

args = parser.parse_args()
print(args)

host=args.redis_host
queue_name = args.queue_name
# Uncomment next two lines if you do not have Kube-DNS working.
# import os
# host = os.getenv("REDIS_SERVICE_HOST")

q = rediswq.RedisWQ(name=queue_name, host=host)
print("Worker with sessionID: " +  q.sessionID())
print("Initial queue state: empty=" + str(q.empty()))
while not q.empty():
  item = q.lease(lease_secs=10, block=True, timeout=2) 
  if item is not None:
    itemstr = item.decode("utf-8")
    job = json.loads(itemstr)
    print("Working on " + job['model_name'])
    cmd = f"python3 train.py --model_name={job['model_name']} --num_epochs={job['num_epochs']} --dataset_url={job['dataset_url']}"
    # time.sleep(10) # Put your actual work here instead of sleep.
    os.system(cmd)
    q.complete(item)
  else:
    print("Waiting for work")
print("Queue empty, exiting")
