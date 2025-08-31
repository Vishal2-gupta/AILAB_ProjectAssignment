# AILab_Project

# MLOps Pipeline for Titanic Survival Prediction

This repository contains a robust MLOps pipeline for predicting survival on the Titanic dataset, implemented as part of a project assignment. The pipeline includes data preprocessing, model training with Spark MLlib, experiment tracking with MLflow, model deployment via FastAPI, and drift detection with automated retraining using DVC for version control.

## Overview

- **Dataset**: Titanic dataset (train.csv, test.csv) from Kaggle.
- **Tools**: Apache Spark, MLflow, FastAPI, DVC, Python.
- **Goal**: Build an end-to-end MLOps workflow with data versioning, model management, deployment, and robustness features.
- **Tasks**:
  - Task 1: Data Pipeline and Versioning
  - Task 2: Model Development and Training
  - Task 3: Experiment and Model Management
  - Task 4: Model Deployment and Testing
  - Task 5: Future-Proofing and Robustness (Drift Detection, Retraining)

## Features

- Automated pipeline using DVC.
- Distributed data preprocessing and model training with Spark.
- Experiment tracking and model registry with MLflow.
- REST API for real-time predictions.
- Drift detection and automated retraining to handle data shifts.

## Prerequisites

- **Python 3.11+**
- **Installed Packages**: Install dependencies via:
  ```bash
  pip install -r requirements.txt

  
## System Requirements:

- Java 8 or 11 (for Spark).
- Approximately 6GB of RAM (recommended for Spark local mode).
- Git for version control.



## Installation

- Clone the Repository:
- bash:
- git clone https://github.com/your-username/mlops-titanic.git
- cd mlops-titanic

- Install Dependencies:
- pip install -r requirements.txt

## Initialize DVC and Git:
- bash:
- git init
- dvc init

## Add Titanic Data:

- Download train.csv and test.csv from Kaggle.
- Place them in the data/raw/ directory.
- Version the raw data with DVC:
- bash:
- mkdir -p data/raw data/processed
- mv train.csv test.csv data/raw/
- dvc add data/raw
- git add data/raw.dvc .gitignore
- git commit -m "Add raw data with DVC"


## Usage
- Run the Full Pipeline

- Start MLflow Tracking Server (in a separate terminal):
- bash:
- mlflow ui --host 127.0.0.1 --port 5000 or mlflow ui

Access the UI at http://127.0.0.1:5000 to monitor experiments.


## Execute the DVC Pipeline:
- bash:
- dvc repro

This automates the following stages:

- preprocess: Preprocesses raw data and saves to data/processed/.
- train: Trains a RandomForest model, logs to MLflow, and saves to models/.
- deploy: Starts a FastAPI server for predictions at http://localhost:8000.
- drift: Detects data drift and triggers retraining if needed.


Note: The API runs in the background; stop it manually (e.g., Ctrl+C) when done testing.


# Test the API:

- Run the test script:
- bash:
- python src/api_test.py

Expected output: A prediction (e.g., Prediction: {'survived': 0}).
Alternatively, use the Swagger UI at http://localhost:8000/docs.



# Manual Script Execution (for Debugging)

Preprocess Data:
- bash:
- python src/data_preprocessing.py
- dvc add data/processed
- git add data/processed.dvc
- git commit -m "Add processed data with DVC"

- Train Model:
- bash:
- python src/model_training.py

- Deploy API:
- bash:
- python src/api_deployment.py

- Detect Drift:
- bash:
- python src/drift_detection.py


## Transition Model to Production

- After training, use the MLflow UI or a script to set the best model version to "Production":
- pythonfrom mlflow.tracking import MlflowClient
- client = MlflowClient(tracking_uri="http://127.0.0.1:5000")
- client.transition_model_version_stage(name="TitanicModel", version=<version>, stage="Production")


## Project Structure

- data/raw/: Raw Titanic data (versioned with DVC).
- data/processed/: Preprocessed data (versioned with DVC).
- src/: Python scripts for preprocessing, training, deployment, and drift detection.
- models/: Saved machine learning models.
- reports/figures/: Generated plots (e.g., confusion matrix).
- dvc.yaml: DVC pipeline configuration.
- requirements.txt: Python dependencies.
- README.md: This file.
- docs/: Project report and diagrams.