FROM tensorflow/tensorflow:2.11.0

RUN apt-get update && apt-get install git -y
RUN pip install redis cloud-tpu-client google-cloud-storage mlflow==2.2.2

COPY . /tf/notebooks

CMD sleep 10000
