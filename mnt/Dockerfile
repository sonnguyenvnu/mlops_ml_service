FROM python:3.10.8-slim

RUN mkdir /app
WORKDIR /app

RUN apt-get update && apt-get install git -y
RUN pip3 install redis mlflow==2.2.2 argparse

VOLUME /app

COPY . .

CMD sleep 10000
