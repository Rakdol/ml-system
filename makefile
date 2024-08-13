ABSOLUTE_PATH := $(shell pwd)
BASE_IMAGE_NAME := ml-system
TRAINING_PATTERN := training
TRAINING_PROJECT := housing
IMAGE_VERSION := 0.0.1

DOCKERFILE := Dockerfile
include .env

.PHONY: dev
dev:
	pip install -r requirements.txt

.PHONY: d_build
d_build:
	docker build \
		-t $(BASE_IMAGE_NAME):$(TRAINING_PATTERN)_$(TRAINING_PROJECT)_$(IMAGE_VERSION) \
		-f $(DOCKERFILE) .

.PHONY: train
train:
	export MLFLOW_S3_ENDPOINT_URL=${MLFLOW_S3_ENDPOINT_URL} && \
	export MLFLOW_TRACKING_URI=${MLFLOW_TRACKING_URI} && \
	export AWS_ACCESS_KEY_ID=${AWS_ACCESS_KEY_ID} && \
	export AWS_SECRET_ACCESS_KEY=${AWS_SECRET_ACCESS_KEY} && \
	mlflow run . --env-manager=local
	
.PHONY: ui
ui:
	mlflow ui
