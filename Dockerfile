FROM python:3.10.8-slim

RUN pip3 install flask redis flask_cors mlflow

RUN mkdir /app
WORKDIR /app

COPY ./service-account-gcs.json /app/service-account-gcs.json

RUN apt update && apt-get update && apt-get install curl -y
RUN apt-get install apt-transport-https ca-certificates gnupg -y
RUN echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" | tee -a /etc/apt/sources.list.d/google-cloud-sdk.list
RUN curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key --keyring /usr/share/keyrings/cloud.google.gpg add - && echo "OK"
RUN apt-get update && apt-get install google-cloud-cli -y
RUN apt-get install kubectl -y
RUN apt-get install google-cloud-sdk-gke-gcloud-auth-plugin -y
RUN gcloud auth activate-service-account owner-160@pixelbrain.iam.gserviceaccount.com --key-file=/app/service-account-gcs.json --project=pixelbrain

RUN gcloud container clusters get-credentials mlops-cluster --region=us-central1-f && kubectl get node

CMD ["sh", "-c", "flask run --host=0.0.0.0 --port=5000"]
EXPOSE 5000
