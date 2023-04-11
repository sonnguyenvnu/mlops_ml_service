import os
import json
import time
from flask import Flask, request
from flask_cors import CORS
import rediswq
from jinja2 import Template

app = Flask(__name__)
CORS(app)

@app.route("/")
def hello_world():
    return "<p>Hello, World!</p>"

@app.route('/clf/train', methods=['GET'])
def train_model():
    # host = os.getenv('REDIS_ADDR')
    # host = 'redis'
    # queue_name = "job2"
    # create_train_jobs(host, queue_name)
    os.system(f"kubectl apply -f ./yaml/push.yaml &")
    os.system(f"kubectl apply -f ./yaml/job_pretrained.yaml &")
    return {'message': 'Training in progress'}, 202

def get_template_file(path):
    with open(path, 'r', encoding='UTF-8') as file:
        return file.read()

def save_file(content, dst_path):
    with open(dst_path, 'w', encoding='UTF-8') as file:
        file.write(content)

@app.route('/train', methods=['POST'])
def render():
    data = request.json
    classes_str = ' '.join(data.get('classes'))
    data['classes'] = classes_str

    dataset_template = get_template_file('./templates/dataset.template.yaml')
    train_template = get_template_file('./templates/train.template.yaml')
    push_job_template = get_template_file(
        './templates/push_job_pretrained.template.yaml')

    jinja2_template = Template(dataset_template)
    content = jinja2_template.render(data)
    dst_path = f"./dataset_{data.get('experiment_name')}.yaml"
    save_file(content, dst_path)

    jinja2_template = Template(train_template)
    content = jinja2_template.render(data)
    dst_path = f"./train_{data.get('experiment_name')}.yaml"
    save_file(content, dst_path)

    jinja2_template = Template(push_job_template)
    content = jinja2_template.render(data)
    dst_path = f"./push_job_pretrained_{data.get('experiment_name')}.yaml"
    save_file(content, dst_path)

    return 'Done'

@app.route('/deploy', methods=['POST'])
def deploy():
    data = request.json
    classes_str = ' '.join(data.get('classes'))
    data['classes'] = classes_str

    deploy_template = get_template_file(
        './templates/docker-compose.template.yaml')
    jinja2_template = Template(deploy_template)
    content = jinja2_template.render(data)
    dst_path = f"./deploy_{data.get('experiment_name')}.yaml"
    save_file(content, dst_path)
    os.system(f"docker-compose -f {dst_path} up -d --force-recreate --build")

    return {'status': 'OK'}

if __name__ == '__main__':
    app.run(port=4000)
