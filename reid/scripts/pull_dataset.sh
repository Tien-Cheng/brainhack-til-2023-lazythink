#!/bin/bash

# Authenticate to GCP
gcloud auth login

# Pull the dataset from GCS
mkdir -p /tmp/cv
gsutil cp gs://cloud-ai-platform-e8edc327-855c-4911-bb8e-205517f8c899/cv/Train.zip /tmp/cv/Train.zip
gsutil cp gs://cloud-ai-platform-e8edc327-855c-4911-bb8e-205517f8c899/cv/Validation.zip /tmp/cv/Validation.zip
gsutil cp gs://cloud-ai-platform-e8edc327-855c-4911-bb8e-205517f8c899/cv/Test.zip /tmp/cv/Test.zip
gsutil cp gs://cloud-ai-platform-e8edc327-855c-4911-bb8e-205517f8c899/cv/train_labels.zip /tmp/cv/train_labels.zip
gsutil cp gs://cloud-ai-platform-e8edc327-855c-4911-bb8e-205517f8c899/cv/val_labels.zip /tmp/cv/val_labels.zip

# Unzip the dataset
mkdir -p data/images
mkdir -p data/labels/yolo
unzip -q -o /tmp/cv/Train.zip -d data/images/train
unzip -q -o /tmp/cv/Validation.zip -d data/images/validation
unzip -q -o /tmp/cv/Test.zip -d data/images/test

unzip -q -o /tmp/cv/train_labels.zip -d data/labels/yolo/train_labels
unzip -q -o /tmp/cv/val_labels.zip -d data/labels/yolo/val_labels
