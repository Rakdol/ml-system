FROM python:3.9-buster

ENV PROJECT_DIR=/mlflow/projects
ENV CODE_DIR=/mlflow/projects/code
WORKDIR /${PROJECT_DIR}

COPY requirements.txt /${PROJECT_DIR}/requirements.txt

RUN apt-get -y update && \
    apt-get -y install apt-utils gcc && \
    pip install --upgrade pip setuptools wheel && \
    pip install  --no-cache-dir -r requirements.txt

WORKDIR /${CODE_DIR}