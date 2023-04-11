import json
import argparse
import rediswq

parser = argparse.ArgumentParser(description='Simple parser')

parser.add_argument('--redis_host', type=str, default='redis', metavar='N',
                    help='redis host')
parser.add_argument('--queue_name', type=str, default='job2', metavar='N',
                    help='queue name')
parser.add_argument('--dataset_url', type=str, default='', metavar='N',
                    help='dataset url')
parser.add_argument('--num_epochs', type=int, default=100, metavar='N',
                    help='number of epochs')
parser.add_argument('--num_classes', type=int, default=2, metavar='N',
                    help='number of classes')
parser.add_argument('--target_size', type=int, default=224, metavar='N',
                    help='size of images')
args = parser.parse_args()
print('>>> Args:', args)

pretrained_models = ['Xception', 'MobileNetV2', 'VGG16', 'VGG19', 'ResNet50', 'EfficientNetB0', 'EfficientNetB1',
                     'EfficientNetB2', 'EfficientNetB3']


def create_train_jobs(args):
    q = rediswq.RedisWQ(name=args.queue_name, host=args.redis_host)
    for model_name in pretrained_models:
        job = json.dumps(
            {
                "model_name": model_name,
                "num_epochs": args.num_epochs,
                "num_classes": args.num_classes,
                "dataset_url": args.dataset_url,
                'target_size': args.target_size,
            },)
        q.put(job)
    print('Done, exiting...')


create_train_jobs(args)
