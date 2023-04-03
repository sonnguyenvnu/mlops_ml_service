import os, json
from flask import Flask, request
from flask_cors import CORS
from dotenv import load_dotenv
from dataset import to_tfrecord_dataset
import rediswq

app = Flask(__name__)
CORS(app)
load_dotenv()

@app.route("/")
def hello_world():
    return "<p>Hello, World!</p>"

@app.route('/clf/dataset', methods=['POST'])
def create_clf_dataset():
    data = request.get_json(force=True)
    print(data, type(data))
    to_tfrecord_dataset(data)
    # without leading space
    # labels = ['ants', 'bees']
    # flag = ' '.join(labels)
    # print(flag)
    # {'name': 'Test', 'classes': ['ants', 'bees']} <class 'dict'>
    return data

pretrained_models = ['Xception', 'VGG16', 'MobileNetV2', 'ResNet50']
def create_train_jobs(host, queue_name):
    q = rediswq.RedisWQ(name=queue_name, host=host)
    for model_name in pretrained_models:
        job = json.dumps({ 'model_name': model_name, 'num_epochs': 100, 'dataset_url': 'gs://uet-mlops/antsbees/*.tfrec', 'num_classes': 2})
        q.put(job)

@app.route('/clf/train', methods=['POST'])
def train_model():
    host = os.getenv('REDIS_ADDR')
    queue_name = "job2"
    create_train_jobs(host, queue_name)
    os.system(f"kubectl apply -f ./yaml/job.yaml")
    return {'message': 'Training in progress'}, 202

if __name__ == '__main__':
    app.run(port=4000)
