"""
PyTorch Neural Network for Health Risk Prediction
Shared model architecture used by all hospitals and coordinator
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class HealthRiskModel(nn.Module):
    """
    Neural Network for Binary Classification of Patient Health Risk
    
    Architecture:
    - Input Layer: 10 features
    - Hidden Layer 1: 64 neurons + BatchNorm + ReLU + Dropout
    - Hidden Layer 2: 32 neurons + BatchNorm + ReLU + Dropout
    - Output Layer: 2 classes (Healthy, At-Risk)
    """
    
    def __init__(self, input_dim: int = 10, hidden_dim: int = 64, output_dim: int = 2, dropout: float = 0.3):
        super(HealthRiskModel, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # Layer 1: Input -> Hidden
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.dropout1 = nn.Dropout(dropout)
        
        # Layer 2: Hidden -> Hidden
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.bn2 = nn.BatchNorm1d(hidden_dim // 2)
        self.dropout2 = nn.Dropout(dropout)
        
        # Layer 3: Hidden -> Output
        self.fc3 = nn.Linear(hidden_dim // 2, output_dim)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Xavier initialization for better convergence"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
        
        Returns:
            Output logits of shape (batch_size, output_dim)
        """
        # Layer 1
        x = self.fc1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout1(x)
        
        # Layer 2
        x = self.fc2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.dropout2(x)
        
        # Output layer (no activation - will use CrossEntropyLoss)
        x = self.fc3(x)
        
        return x
    
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get predicted class (0 or 1)
        
        Args:
            x: Input tensor
        
        Returns:
            Predicted class indices
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            predictions = torch.argmax(logits, dim=1)
        return predictions
    
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get class probabilities
        
        Args:
            x: Input tensor
        
        Returns:
            Probability distribution over classes
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            probabilities = F.softmax(logits, dim=1)
        return probabilities
    
    def get_num_parameters(self) -> int:
        """Get total number of trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_weights(self) -> dict:
        """Extract model weights as a dictionary (for FL aggregation)"""
        return {name: param.cpu().detach().numpy() for name, param in self.state_dict().items()}
    
    def set_weights(self, weights: dict):
        """Set model weights from dictionary (for FL aggregation)"""
        state_dict = {name: torch.tensor(weight) for name, weight in weights.items()}
        self.load_state_dict(state_dict, strict=True)


def create_model(device: str = 'cpu') -> HealthRiskModel:
    """
    Factory function to create and initialize model
    
    Args:
        device: 'cpu' or 'cuda'
    
    Returns:
        Initialized HealthRiskModel
    """
    model = HealthRiskModel(
        input_dim=10,
        hidden_dim=64,
        output_dim=2,
        dropout=0.3
    )
    model = model.to(device)
    
    # Print model summary
    print(f"\n{'='*60}")
    print("MODEL ARCHITECTURE")
    print(f"{'='*60}")
    print(model)
    print(f"\nTotal Parameters: {model.get_num_parameters():,}")
    print(f"Device: {device}")
    print(f"{'='*60}\n")
    
    return model


def get_model_size_mb(model: nn.Module) -> float:
    """Calculate model size in megabytes"""
    param_size = 0
    buffer_size = 0
    
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    
    size_mb = (param_size + buffer_size) / (1024 ** 2)
    return size_mb


if __name__ == "__main__":
    """Test the model"""
    print("Testing HealthRiskModel...")
    
    # Create model
    model = create_model(device='cpu')
    
    # Test forward pass with dummy data
    batch_size = 32
    dummy_input = torch.randn(batch_size, 10)
    
    print(f"\nTest Forward Pass:")
    print(f"  Input shape: {dummy_input.shape}")
    
    output = model(dummy_input)
    print(f"  Output shape: {output.shape}")
    
    # Test prediction
    predictions = model.predict(dummy_input)
    print(f"  Predictions shape: {predictions.shape}")
    print(f"  Sample predictions: {predictions[:5].tolist()}")
    
    # Test probability prediction
    probabilities = model.predict_proba(dummy_input)
    print(f"  Probabilities shape: {probabilities.shape}")
    print(f"  Sample probabilities: {probabilities[:3].tolist()}")
    
    # Model size
    print(f"\nModel Size: {get_model_size_mb(model):.2f} MB")
    
    print("\n✅ Model test completed successfully!")