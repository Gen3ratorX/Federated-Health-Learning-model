"""
Flower Client Implementation for Hospital Nodes
Handles federated learning participation
"""

import flwr as fl
from flwr.common import (
    Code,
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    GetParametersIns,
    GetParametersRes,
    Parameters,
    Status,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from typing import Dict, Tuple, List
import torch
import numpy as np
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

from shared.models.base_model import create_model
from shared.constants import (
    BATCH_SIZE,
    LOCAL_EPOCHS,
    LEARNING_RATE,
    DEVICE,
    FEATURE_COLUMNS
)
from hospital.app.data.loader import HospitalDataLoader
from hospital.app.training.trainer import HospitalTrainer


class HealthFlowerClient(fl.client.NumPyClient):
    """
    Flower Client for federated learning at hospital nodes
    
    This client:
    1. Receives global model parameters from coordinator
    2. Trains on local hospital data
    3. Sends updated parameters back to coordinator
    4. Never shares raw patient data
    """
    
    def __init__(
        self,
        hospital_id: int,
        data_path: str = None,
        batch_size: int = BATCH_SIZE,
        local_epochs: int = LOCAL_EPOCHS,
        learning_rate: float = LEARNING_RATE,
        device: str = DEVICE
    ):
        """
        Args:
            hospital_id: Unique hospital identifier
            data_path: Path to hospital's CSV data file
            batch_size: Batch size for training
            local_epochs: Number of local training epochs per FL round
            learning_rate: Learning rate for local training
            device: 'cpu' or 'cuda'
        """
        self.hospital_id = hospital_id
        self.batch_size = batch_size
        self.local_epochs = local_epochs
        self.learning_rate = learning_rate
        self.device = device
        
        print(f"\n{'='*60}")
        print(f"🏥 INITIALIZING HOSPITAL {hospital_id} CLIENT")
        print(f"{'='*60}")
        
        # Load data
        print(f"\n📂 Loading hospital data...")
        self.data_loader = HospitalDataLoader(
            hospital_id=hospital_id,
            data_path=data_path,
            normalize=True
        )
        self.data_loader.prepare_data()
        
        self.train_loader = self.data_loader.get_train_loader(batch_size=batch_size)
        self.test_loader = self.data_loader.get_test_loader(batch_size=batch_size)
        
        # Get class weights for imbalanced data handling
        self.class_weights = self.data_loader.get_class_weights()
        
        # Create model
        print(f"\n🤖 Creating model...")
        self.model = create_model(device=device)
        
        # Create trainer
        self.trainer = HospitalTrainer(
            model=self.model,
            device=device,
            learning_rate=learning_rate,
            class_weights=self.class_weights
        )
        
        # Statistics
        self.stats = self.data_loader.get_data_statistics()
        self.num_examples = {
            "trainset": self.stats['train_samples'],
            "testset": self.stats['test_samples']
        }
        
        print(f"\n✅ Client initialized successfully!")
        print(f"   Train samples: {self.num_examples['trainset']}")
        print(f"   Test samples: {self.num_examples['testset']}")
        print(f"{'='*60}\n")
    
    def get_parameters(self, config: Dict[str, any]) -> List[np.ndarray]:
        """
        Return current model parameters
        
        Args:
            config: Configuration dictionary (not used here)
        
        Returns:
            List of numpy arrays containing model parameters
        """
        print(f"🔄 Hospital {self.hospital_id}: Getting parameters...")
        return self.trainer.get_model_parameters()
    
    def fit(self, parameters: List[np.ndarray], config: Dict[str, any]) -> Tuple[List[np.ndarray], int, Dict]:
        """
        Train model on local data
        
        Args:
            parameters: Global model parameters from coordinator
            config: Configuration dictionary with training settings
        
        Returns:
            Updated parameters, number of training examples, metrics
        """
        print(f"\n{'='*60}")
        print(f"🏥 Hospital {self.hospital_id}: Starting Local Training")
        print(f"{'='*60}")
        
        # Set model parameters from coordinator
        self.trainer.set_model_parameters(parameters)
        
        # Get training configuration (use defaults if not provided)
        epochs = config.get("local_epochs", self.local_epochs)
        
        print(f"\n📊 Training Configuration:")
        print(f"   Epochs: {epochs}")
        print(f"   Batch size: {self.batch_size}")
        print(f"   Learning rate: {self.learning_rate}")
        print(f"   Training samples: {self.num_examples['trainset']}")
        
        # Train on local data
        history = self.trainer.train(
            train_loader=self.train_loader,
            epochs=epochs,
            verbose=True
        )
        
        # Get updated parameters
        updated_parameters = self.trainer.get_model_parameters()
        
        # Prepare metrics to send back
        metrics = {
            "hospital_id": self.hospital_id,
            "train_loss": float(history['loss'][-1]),
            "train_accuracy": float(history['accuracy'][-1]),
            "num_examples": self.num_examples['trainset']
        }
        
        print(f"\n✅ Training completed!")
        print(f"   Final loss: {metrics['train_loss']:.4f}")
        print(f"   Final accuracy: {metrics['train_accuracy']:.4f}")
        print(f"{'='*60}\n")
        
        return updated_parameters, self.num_examples['trainset'], metrics
    
    def evaluate(self, parameters: List[np.ndarray], config: Dict[str, any]) -> Tuple[float, int, Dict]:
        """
        Evaluate model on local test data
        
        Args:
            parameters: Global model parameters from coordinator
            config: Configuration dictionary
        
        Returns:
            Loss, number of test examples, metrics
        """
        print(f"\n🔍 Hospital {self.hospital_id}: Evaluating model...")
        
        # Set model parameters from coordinator
        self.trainer.set_model_parameters(parameters)
        
        # Evaluate on local test data
        loss, accuracy, detailed_metrics = self.trainer.evaluate(self.test_loader)
        
        # Prepare metrics
        metrics = {
            "hospital_id": self.hospital_id,
            "accuracy": float(accuracy),
            "precision": float(detailed_metrics['precision']),
            "recall": float(detailed_metrics['recall']),
            "f1_score": float(detailed_metrics['f1_score']),
            "num_examples": self.num_examples['testset']
        }
        
        print(f"   Loss: {loss:.4f}")
        print(f"   Accuracy: {accuracy:.4f}")
        print(f"   F1 Score: {detailed_metrics['f1_score']:.4f}\n")
        
        return float(loss), self.num_examples['testset'], metrics


def create_client(hospital_id: int) -> HealthFlowerClient:
    """
    Factory function to create a Flower client for a hospital
    
    Args:
        hospital_id: Hospital identifier (1, 2, or 3)
    
    Returns:
        Initialized HealthFlowerClient
    """
    return HealthFlowerClient(
        hospital_id=hospital_id,
        batch_size=BATCH_SIZE,
        local_epochs=LOCAL_EPOCHS,
        learning_rate=LEARNING_RATE,
        device=DEVICE
    )


def start_client(hospital_id: int, server_address: str = "localhost:8080"):
    """
    Start Flower client and connect to coordinator
    
    Args:
        hospital_id: Hospital identifier
        server_address: Coordinator address (host:port)
    """
    print(f"\n{'='*60}")
    print(f"🚀 STARTING FLOWER CLIENT FOR HOSPITAL {hospital_id}")
    print(f"{'='*60}")
    print(f"Server address: {server_address}\n")
    
    # Create client
    client = create_client(hospital_id)
    
    # Connect to coordinator
    try:
        fl.client.start_client(
            server_address=server_address,
            client=client.to_client()
        )
    except Exception as e:
        print(f"\n❌ Error connecting to coordinator: {e}")
        print("Make sure the coordinator is running!")
        raise


if __name__ == "__main__":
    """
    Run this script to start a hospital client
    Usage: python hospital/app/client.py <hospital_id>
    Example: python hospital/app/client.py 1
    """
    import argparse
    
    parser = argparse.ArgumentParser(description="Start Flower client for hospital")
    parser.add_argument(
        "hospital_id",
        type=int,
        help="Hospital ID (1, 2, or 3)"
    )
    parser.add_argument(
        "--server",
        type=str,
        default="localhost:8080",
        help="Coordinator server address (default: localhost:8080)"
    )
    
    args = parser.parse_args()
    
    if args.hospital_id not in [1, 2, 3]:
        print("❌ Error: hospital_id must be 1, 2, or 3")
        exit(1)
    
    # Start client
    start_client(
        hospital_id=args.hospital_id,
        server_address=args.server
    )