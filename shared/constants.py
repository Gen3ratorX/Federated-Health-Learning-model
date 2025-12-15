"""
Shared constants for Federated Learning POC
Used by both coordinator and hospital nodes
"""
import torch

# --- MODEL CONFIGURATION ---
MODEL_INPUT_DIM = 10  # Must match the length of FEATURE_COLUMNS
MODEL_HIDDEN_DIM = 64
MODEL_OUTPUT_DIM = 2  # 2 Classes: [Healthy, At-Risk]

# --- COLUMN MAPPING ---
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
TARGET_COLUMN = 'at_risk'

# --- TRAINING CONFIGURATION ---
BATCH_SIZE = 32
LOCAL_EPOCHS = 2
LEARNING_RATE = 0.0001
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# --- FEDERATED LEARNING CONFIGURATION ---
NUM_FL_ROUNDS = 5             
MIN_AVAILABLE_CLIENTS = 2     
MIN_FIT_CLIENTS = 2           
MIN_EVAL_CLIENTS = 2          

# 👇 THESE WERE MISSING - ADD THEM BACK
FRACTION_FIT = 1.0            # Use 100% of available clients for training
FRACTION_EVALUATE = 1.0       # Use 100% of available clients for evaluation

# --- SERVER CONFIGURATION ---
COORDINATOR_HOST = '0.0.0.0'
COORDINATOR_PORT = 8080
API_PORT = 8000

# --- PATHS ---
DATA_DIR = 'data'
HOSPITAL_DATA_TEMPLATE = 'data/hospital_{hospital_id}/patient_data.csv'
CHECKPOINTS_DIR = 'checkpoints'
GLOBAL_MODEL_PATH = 'checkpoints/global_model.pth' 
RANDOM_SEED = 42
TEST_SIZE = 0.2

SCALING_FACTORS = [
    100.0,  # age (Max expected ~100)
    50.0,   # bmi (Max ~50)
    200.0,  # bp_systolic (Max ~200)
    120.0,  # bp_diastolic
    400.0,  # cholesterol
    250.0,  # glucose
    150.0,  # heart_rate
    1.0,    # smoking (Already 0-1)
    1.0,    # diabetes
    1.0     # family_history
]