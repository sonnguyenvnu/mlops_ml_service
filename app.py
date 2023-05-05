import os
import json
import time
from flask import Flask, request
from flask_cors import CORS
from jinja2 import Template
from mlflow_client import best_accuracy

app = Flask(__name__)
CORS(app)


@app.route("/")
def hello_world():
    return "<p>Hello, World!</p>"

# @app.route('/clf/train', methods=['GET'])
# def train_model():
#     # host = os.getenv('REDIS_ADDR')
#     # host = 'redis'
#     # queue_name = "job2"
#     # create_train_jobs(host, queue_name)
#     os.system(f"kubectl apply -f ./yaml/redis.yaml &")
#     os.system(f"kubectl apply -f ./yaml/push.yaml &")
#     os.system(f"kubectl apply -f ./yaml/job_pretrained.yaml &")
#     os.system(f"kubectl apply -f ./yaml/enas.yaml &")
#     os.system(f"kubectl apply -f ./yaml/job_automl.yaml &")
#     return {'message': 'Training in progress'}, 202

def get_template_file(path):
    with open(path, 'r', encoding='UTF-8') as file:
        return file.read()


def save_file(content, dst_path):
    with open(dst_path, 'w', encoding='UTF-8') as file:
        file.write(content)


@app.route('/clf/train', methods=['POST'])
def train():
    data = request.json
    classes_str = ' '.join(data.get('classes'))
    data['classes'] = classes_str
    print(data)

    pretrained_template = get_template_file(
        './templates/pretrained.template.yaml')
    enas_template = get_template_file('./templates/enas.template.yaml')
    automl_template = get_template_file('./templates/automl.template.yaml')
    push_job_template = get_template_file(
        './templates/push_job_pretrained.template.yaml')

    # jinja2_template = Template(push_job_template)
    # content = jinja2_template.render(data)
    # dst_path = f"./push_job_pretrained_{data.get('experiment_name')}.yaml"
    # save_file(content, dst_path)
    # os.system(f"kubectl apply -f {dst_path} &")

    # template = Template(pretrained_template)
    # content = template.render(data)

    content = ""
    template = Template(enas_template)
    # 5 to 20 layers
    for i in range(4, 5):
        data['num_layers'] = i
        content += template.render(data)
    dst_path = f"./train_{data.get('experiment_name')}.yaml"

    template = Template(automl_template)
    content += template.render(data)

    save_file(content, dst_path)
    os.system(f"kubectl apply -f {dst_path} &")
    return {'status': 'OK'}

@app.route('/clf/stop', methods=['GET'])
def stop_train():
    experiment_name = request.args.get('experiment_name')
    os.system(f"kubectl -n kubeflow delete -f ./dataset_{experiment_name}.yaml")
    os.system(f"kubectl -n kubeflow delete -f ./train_{experiment_name}.yaml")
    os.system(f"rm -rf ./dataset_{experiment_name}.yaml")
    os.system(f"rm -rf ./train_{experiment_name}.yaml")
    return {'message': 'Training stopped'}

@app.route('/clf/dataset', methods=['POST'])
def write_dataset():
    data = request.json
    data['classes'] = ' '.join(data.get('classes'))

    dataset_template = get_template_file('./templates/dataset.template.yaml')
    jinja2_template = Template(dataset_template)
    content = jinja2_template.render(data)
    dst_path = f"./dataset_{data.get('experiment_name')}.yaml"
    save_file(content, dst_path)
    os.system(f"kubectl apply -f {dst_path} &")
    return {'status': 'OK'}


@app.route('/clf/deploy', methods=['GET'])
def deploy():
    # data = request.json
    # classes_str = ' '.join(data.get('classes'))
    # data['classes'] = classes_str
    data = {
        'classes': 'daisy dandelion roses sunflowers tulips',
        'target_size': 224,
        'model_dir': 'gs://uet-mlflow/example_experiment/06d964a0070f45709edcfabb9854fd62/models/ckpt_epoch_7',
        'experiment_name': 'kashuiyweiruiw',
    }

    deploy_template = get_template_file(
        './templates/docker-compose.template.yaml')
    jinja2_template = Template(deploy_template)
    content = jinja2_template.render(data)
    dst_path = f"./deploy_{data.get('experiment_name')}.yaml"
    save_file(content, dst_path)
    os.system(f"docker-compose -f {dst_path} up -d --force-recreate --build")

    return {'status': 'OK'}

@app.route('/accuracy/best', methods=['GET'])
def best_experiment_model():
    experiment_name = request.args.get('experiment_name')
    for_deployment = request.args.get('for_deployment')
    data = best_accuracy(experiment_name, for_deployment)
    return data

if __name__ == '__main__':
    app.run(port=4000)
