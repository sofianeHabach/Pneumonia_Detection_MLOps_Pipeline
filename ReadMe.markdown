# Pneumonia Detection MLOps Pipeline

## Overview
This project implements an end-to-end MLOps pipeline for detecting pneumonia from chest X-ray images using the [Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) dataset from Kaggle. The pipeline integrates modern MLOps practices, leveraging **GitHub Actions** for CI/CD orchestration, **DVC** for data versioning, **MLflow** for experiment tracking, **Docker** for containerization, **FastAPI** for serving predictions, and **React** for a user-friendly interface. The system achieves a test accuracy of **97.56%** and a recall of **99%** using an **EfficientNetB0** model.

## Features
- **Data Management**: Dataset versioning using DVC.
- **Experiment Tracking**: MLflow for logging parameters, metrics, and artifacts.
- **CI/CD Automation**: GitHub Actions for automated training, validation, and deployment.
- **Model Serving**: FastAPI-based API for real-time pneumonia predictions.
- **Frontend Interface**: React-based UI for uploading images and viewing predictions.
- **Monitoring**: Structured logging of predictions with request IDs and timestamps.
- **Containerization**: Dockerized services (FastAPI, MLflow, Nginx, React) for reproducibility.
- **Public Access**: Ngrok for exposing services during testing.

## Project Structure
```
├── Dockerfile.fastapi          # Dockerfile for FastAPI backend
├── Dockerfile.frontend         # Dockerfile for React frontend
├── Dockerfile.mlflow           # Dockerfile for MLflow server
├── Dockerfile.nginx            # Dockerfile for Nginx reverse proxy
├── app/
│   ├── build.py               # Model building script
│   ├── main.py                # FastAPI application for predictions
│   ├── model/
│   │   ├── class_mapping.json # Class labels mapping
│   │   └── pneumonia_model.py # Prediction logic
├── build.py                   # Model architecture definition
├── data_preprocessing.py      # Data preprocessing script
├── docker-compose.yml         # Docker Compose configuration
├── frontend/
│   ├── index.html             # Frontend entry point
│   ├── package.json           # Node.js dependencies
│   ├── src/
│   │   ├── App.css           # Frontend styling
│   │   ├── App.jsx           # Main React component
│   │   └── main.jsx          # React entry point
│   └── vite.config.js         # Vite configuration for frontend
├── model_evaluation.py        # Model evaluation script
├── model_training.py          # Model training script
├── nginx/
│   ├── default.conf           # Nginx configuration for backend
│   ├── frontend.conf          # Nginx configuration for frontend
│   └── nginx.conf             # Main Nginx configuration
├── requirements.txt           # Python dependencies for training
├── requirements_deployement.txt # Python dependencies for deployment
└── train_test_split.py        # Script for splitting dataset
```

## Prerequisites
- **GitHub Account** with a repository containing this project.
- **GitHub Secrets** configured:
  - `NGROK_AUTH_TOKEN`
  - `GITHUB_TOKEN`
- **Kaggle Account** to access the [Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) dataset.
- *(Optional for local testing)*: **Python 3.9.20**, **Docker**, **Docker Compose**, **Node.js 18**

## Setup Instructions

1. **Fork or Clone the Repository**:
   ```bash
   git clone <repository-url>
   cd <repository-name>
   ```

2. **Configure GitHub Secrets**:
   - In your GitHub repository, go to `Settings` > `Secrets and variables` > `Actions`, then add:
     - `NGROK_AUTH_TOKEN`
     - `GITHUB_TOKEN`

3. **Download the Dataset**:
   - Download the [Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) dataset from Kaggle.
   - Place the dataset in the chest_xray/ directory

3. **Trigger the CI/CD Pipeline**:
   - Push to the `main` branch or manually run the workflow via the GitHub Actions interface.
   - The pipeline automatically handles:
     - Dataset versioning via DVC
     - Data preprocessing and splitting
     - Model training and tracking with MLflow
     - Docker container building
     - Deployment via Ngrok

4. **Access Deployed Services**:
   - Once the pipeline completes, Ngrok URLs are visible in the GitHub Actions logs.
   - Available services:
     - 🌐 **React Frontend**: `https://<ngrok-url>` (corresponds to port 80 of the React application)
     - ⚙️ **FastAPI API**: `https://<ngrok-url>/predict/`
     - 📊 **MLflow UI**: `https://<ngrok-url>/mlflow`

## Pipeline Workflow
1. **Data Preprocessing** (`data_preprocessing.py`):
   - Resize images to 224x224 pixels and convert to grayscale.
   - Handle class imbalance using weighted loss (50% higher weight for NORMAL class).

2. **Data Splitting** (`train_test_split.py`):
   - Split dataset into training (65%), validation (21%), and test (14%) sets.

3. **Model Training** (`model_training.py`):
   - Uses EfficientNetB0 with fine-tuning, BatchNormalization, Dropout (0.3), and Dense layers.
   - Training stops early if validation accuracy plateaus (EarlyStopping, ModelCheckpoint, ReduceLROnPlateau).

4. **Model Evaluation** (`model_evaluation.py`):
   - Evaluates on test set, generating accuracy, loss, F1-score, and confusion matrix.
   - Validates model if F1-scores for both classes exceed 0.9.

5. **CI/CD with GitHub Actions**:
   - **ML Pipeline** (`ml_pipeline.yml`):
     - Triggered on push to `main` or manual dispatch.
     - Handles preprocessing, splitting, training, evaluation, and artifact upload.
   - **Deploy Pipeline** (`deploy.yaml`):
     - Triggered on successful ML pipeline completion.
     - Downloads artifacts, builds Docker containers, and exposes services via Ngrok.

6. **Prediction Service** (`main.py`):
   - FastAPI endpoint (`/predict/`) accepts image uploads and returns predictions.
   - Logs predictions in JSON format with timestamps, request IDs, and durations.

7. **Monitoring**:
   - Prediction logs stored in `logs/predictions.log` for drift detection and performance analysis.

## Results
- **Model Performance**:
  - Test Accuracy: **97.56%**
  - Test Recall (PNEUMONIA): **99%**
  - F1-score (NORMAL): **0.95**
  - F1-score (PNEUMONIA): **0.98**
- **Prediction Latency**: 94ms to 1.08s per image.
- **Deployment**: Fully automated via GitHub Actions, with services accessible via Ngrok.

## Usage
1. **Via Frontend**:
   - Open the React interface at `https://<ngrok-url>` (port 80).
   - Upload a chest X-ray image to receive a prediction (e.g., "Normal" with 98.8% confidence).

2. **Via API**:
   - Send a POST request to `https://<ngrok-url>/predict/` with an image file:
     ```bash
     curl -X POST -F "file=@image.jpg" https://<ngrok-url>/predict/
     ```