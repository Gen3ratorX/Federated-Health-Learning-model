"""
Shared constants for Federated Learning POC
Used by both coordinator and hospital nodes
"""

import torch

# --- MODEL CONFIGURATION ---
MODEL_INPUT_DIM = 10  # Matches the length of FEATURE_COLUMNS below
MODEL_HIDDEN_DIM = 64
MODEL_OUTPUT_DIM = 2  # Binary classification (Healthy vs At-Risk)

# Feature columns (The inputs)
# NOTE: We EXCLUDE 'patient_id' (irrelevant) and 'risk_score' (data leakage)
FEATURE_COLUMNS = [
    'age',
    'bmi',
    'bp_systolic',
    'bp_diastolic',
    'cholesterol',
    'glucose',
    'heart_rate',
    'smoking',
    'diabetes',
    'family_history'
]

# The target column (The prediction)
TARGET_COLUMN = 'at_risk'

# --- TRAINING CONFIGURATION ---
BATCH_SIZE = 32
LOCAL_EPOCHS = 2     # How many loops the hospital trains locally per round
LEARNING_RATE = 0.001
# Automatically detect GPU
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# --- FEDERATED LEARNING CONFIGURATION ---
NUM_FL_ROUNDS = 5             # Total rounds of global training
MIN_AVAILABLE_CLIENTS = 2     # Wait for 2 hospitals before starting
MIN_FIT_CLIENTS = 2           # Train with 2 hospitals
MIN_EVAL_CLIENTS = 2          # Validate with 2 hospitals
FRACTION_FIT = 1.0            # Use 100% of connected clients
FRACTION_EVALUATE = 1.0       # Evaluate on 100% of connected clients

# --- SERVER CONFIGURATION ---
COORDINATOR_HOST = '0.0.0.0'
COORDINATOR_PORT = 8080
API_PORT = 8000

# --- PATHS & SYSTEM ---
DATA_DIR = 'data'
# This template helps the client find the right file based on ID
HOSPITAL_DATA_TEMPLATE = 'data/hospital_{hospital_id}/patient_data.csv'
CHECKPOINTS_DIR = 'checkpoints'
GLOBAL_MODEL_PATH = 'checkpoints/global_model.pth'

# --- REPRODUCIBILITY ---
RANDOM_SEED = 42
TEST_SIZE = 0.2