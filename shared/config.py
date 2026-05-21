"""
Shared configuration for FedCare system
"""
import os
from dotenv import load_dotenv

load_dotenv()

# Server Configuration
MAIN_SERVER_HOST = os.getenv("MAIN_SERVER_HOST", "localhost")
MAIN_SERVER_PORT = int(os.getenv("MAIN_SERVER_PORT", 5000))
MAIN_SERVER_URL = os.getenv("MAIN_SERVER_URL", "").strip()
HOSPITAL_SERVER_PORT = int(os.getenv("HOSPITAL_SERVER_PORT", 5001))

# Federated Learning Configuration
NUM_ROUNDS = int(os.getenv("NUM_ROUNDS", 10))
NUM_HOSPITALS = int(os.getenv("NUM_HOSPITALS", 3))
EPOCHS_PER_ROUND = int(os.getenv("EPOCHS_PER_ROUND", 5))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", 32))
TEST_SIZE = float(os.getenv("TEST_SIZE", 0.2))

# Model Configuration
INPUT_SHAPE = (28, 28, 1)  # MNIST-like data
NUM_CLASSES = 10
LEARNING_RATE = float(os.getenv("LEARNING_RATE", 0.001))

# Security and Privacy
ENCRYPTION_ENABLED = os.getenv("ENCRYPTION_ENABLED", "False").lower() == "true"
SECURE_AGGREGATION = os.getenv("SECURE_AGGREGATION", "False").lower() == "true"
DIFFERENTIAL_PRIVACY = os.getenv("DIFFERENTIAL_PRIVACY", "False").lower() == "true"

# AWS Configuration
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
SAGEMAKER_ROLE_ARN = os.getenv("SAGEMAKER_ROLE_ARN", "")
BUCKET_NAME = os.getenv("BUCKET_NAME", "fedcare-models")

# Logging
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_DIR = os.getenv("LOG_DIR", "./logs")

# Database
MONGODB_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
DB_NAME = "fedcare"

# API Configuration
API_TIMEOUT = int(os.getenv("API_TIMEOUT", 300))
MAX_RETRIES = int(os.getenv("MAX_RETRIES", 3))
RETRY_BACKOFF = float(os.getenv("RETRY_BACKOFF", 1.0))
