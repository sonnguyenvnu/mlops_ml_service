import json
import argparse
import rediswq

parser = argparse.ArgumentParser(description='Simple parser')

parser.add_argument('--redis_host', type=str, default='redis', metavar='N',
                    help='Redis host')
parser.add_argument('--queue_name', type=str, default='job2', metavar='N',
                    help='Queue name')
args = parser.parse_args()
print('Redis host:', args.redis_host)
print('Queue name:', args.queue_name)

pretrained_models = ['Xception', 'MobileNetV2', 'VGG16', 'VGG19', 'ResNet50', 'EfficientNetB0', 'EfficientNetB1',
                     'EfficientNetB2', 'EfficientNetB3']


def create_train_jobs(host, queue_name):
    q = rediswq.RedisWQ(name=queue_name, host=host)
    for model_name in pretrained_models:
        job = json.dumps(
            {
                "model_name": model_name,
                "num_epochs": 100,
                "num_classes": 2,
                "dataset_url": "gs://uet-mlops/antsbees/*.tfrec",
                'target_size': 224,
            },)
        q.put(job)
    print('Done, exiting...')


create_train_jobs(host=args.redis_host, queue_name=args.queue_name)
