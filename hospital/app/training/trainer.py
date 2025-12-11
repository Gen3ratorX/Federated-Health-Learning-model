"""
Local training logic for hospital nodes
Handles model training and evaluation on private data
"""

# Fix for Windows Unicode/Emoji support
import sys
import io
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Tuple, Dict
import numpy as np
from tqdm import tqdm
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from shared.constants import LOCAL_EPOCHS, LEARNING_RATE, DEVICE


class HospitalTrainer:
    """
    Manages local training for a hospital node
    """
    
    def __init__(
        self,
        model: nn.Module,
        device: str = DEVICE,
        learning_rate: float = LEARNING_RATE,
        class_weights: torch.Tensor = None
    ):
        """
        Args:
            model: PyTorch model to train
            device: 'cpu' or 'cuda'
            learning_rate: Learning rate for optimizer
            class_weights: Weights for handling class imbalance
        """
        self.model = model.to(device)
        self.device = device
        self.learning_rate = learning_rate
        
        # Loss function (with optional class weights for imbalanced data)
        if class_weights is not None:
            self.criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
        else:
            self.criterion = nn.CrossEntropyLoss()
        
        # Optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=1e-5  # L2 regularization
        )
        
        # Learning rate scheduler (optional but recommended)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=2,
            verbose=False
        )
    
    def train_epoch(self, train_loader: DataLoader) -> Tuple[float, float]:
        """
        Train for one epoch
        
        Args:
            train_loader: DataLoader for training data
        
        Returns:
            Average loss and accuracy for the epoch
        """
        self.model.train()
        
        total_loss = 0.0
        correct = 0
        total = 0
        
        for features, labels in train_loader:
            # Move data to device
            features = features.to(self.device)
            labels = labels.to(self.device)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            outputs = self.model(features)
            loss = self.criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping (helps with stability)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            # Update weights
            self.optimizer.step()
            
            # Track metrics
            total_loss += loss.item() * features.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        avg_loss = total_loss / total
        accuracy = correct / total
        
        return avg_loss, accuracy
    
    def train(
        self,
        train_loader: DataLoader,
        epochs: int = LOCAL_EPOCHS,
        verbose: bool = True
    ) -> Dict[str, list]:
        """
        Train the model for multiple epochs
        
        Args:
            train_loader: DataLoader for training data
            epochs: Number of epochs to train
            verbose: Whether to print progress
        
        Returns:
            Dictionary with training history
        """
        history = {
            'loss': [],
            'accuracy': []
        }
        
        if verbose:
            print(f"\n🔄 Training for {epochs} epochs...")
            iterator = tqdm(range(epochs), desc="Training")
        else:
            iterator = range(epochs)
        
        for epoch in iterator:
            loss, accuracy = self.train_epoch(train_loader)
            
            history['loss'].append(loss)
            history['accuracy'].append(accuracy)
            
            if verbose:
                iterator.set_postfix({
                    'loss': f'{loss:.4f}',
                    'acc': f'{accuracy:.4f}'
                })
        
        return history
    
    def evaluate(self, test_loader: DataLoader) -> Tuple[float, float, Dict[str, float]]:
        """
        Evaluate the model
        
        Args:
            test_loader: DataLoader for test data
        
        Returns:
            loss, accuracy, and detailed metrics
        """
        self.model.eval()
        
        total_loss = 0.0
        correct = 0
        total = 0
        
        # For calculating precision, recall, F1
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for features, labels in test_loader:
                features = features.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(features)
                loss = self.criterion(outputs, labels)
                
                total_loss += loss.item() * features.size(0)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        avg_loss = total_loss / total
        accuracy = correct / total
        
        # Calculate additional metrics
        all_predictions = np.array(all_predictions)
        all_labels = np.array(all_labels)
        
        # For binary classification
        true_positives = np.sum((all_predictions == 1) & (all_labels == 1))
        false_positives = np.sum((all_predictions == 1) & (all_labels == 0))
        false_negatives = np.sum((all_predictions == 0) & (all_labels == 1))
        true_negatives = np.sum((all_predictions == 0) & (all_labels == 0))
        
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        metrics = {
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'true_positives': int(true_positives),
            'false_positives': int(false_positives),
            'true_negatives': int(true_negatives),
            'false_negatives': int(false_negatives)
        }
        
        return avg_loss, accuracy, metrics
    
    def get_model_parameters(self) -> list:
        """
        Get model parameters as a list of numpy arrays (for FL)
        
        Returns:
            List of numpy arrays
        """
        return [param.cpu().detach().numpy() for param in self.model.parameters()]
    
    def set_model_parameters(self, parameters: list):
        """
        Set model parameters from a list of numpy arrays (for FL)
        
        Args:
            parameters: List of numpy arrays
        """
        params_dict = zip(self.model.parameters(), parameters)
        for param, new_param in params_dict:
            param.data = torch.tensor(new_param, dtype=param.dtype).to(self.device)


def print_evaluation_results(loss: float, accuracy: float, metrics: Dict[str, float]):
    """
    Pretty print evaluation results
    
    Args:
        loss: Evaluation loss
        accuracy: Evaluation accuracy
        metrics: Dictionary of additional metrics
    """
    print(f"\n{'='*60}")
    print("EVALUATION RESULTS")
    print(f"{'='*60}")
    print(f"Loss: {loss:.4f}")
    print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"\nDetailed Metrics:")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall: {metrics['recall']:.4f}")
    print(f"  F1 Score: {metrics['f1_score']:.4f}")
    print(f"\nConfusion Matrix:")
    print(f"  True Positives:  {metrics['true_positives']}")
    print(f"  False Positives: {metrics['false_positives']}")
    print(f"  True Negatives:  {metrics['true_negatives']}")
    print(f"  False Negatives: {metrics['false_negatives']}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    """Test the trainer"""
    print("Testing HospitalTrainer...")
    
    # Import dependencies
    from shared.models.base_model import create_model
    from hospital.app.data.loader import create_data_loaders
    
    # Create model and data
    model = create_model(device='cpu')
    train_loader, test_loader, stats = create_data_loaders(
        hospital_id=1,
        batch_size=32
    )
    
    # Create trainer
    trainer = HospitalTrainer(
        model=model,
        device='cpu',
        learning_rate=0.001
    )
    
    # Train for 2 epochs (quick test)
    print("\n📊 Training test...")
    history = trainer.train(train_loader, epochs=2, verbose=True)
    
    # Evaluate
    print("\n📊 Evaluation test...")
    loss, accuracy, metrics = trainer.evaluate(test_loader)
    print_evaluation_results(loss, accuracy, metrics)
    
    print("✅ Trainer test completed successfully!")