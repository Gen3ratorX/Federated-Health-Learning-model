"""
Data Loader for Hospital Patient Data
Handles CSV loading, preprocessing, and PyTorch dataset creation
"""

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from pathlib import Path
from typing import Tuple, Optional
import sys
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from shared.constants import FEATURE_COLUMNS, TARGET_COLUMN


class HealthDataset(Dataset):
    """PyTorch Dataset for patient health data"""
    
    def __init__(self, features: np.ndarray, labels: np.ndarray):
        """
        Args:
            features: Numpy array of shape (n_samples, n_features)
            labels: Numpy array of shape (n_samples,)
        """
        self.features = torch.FloatTensor(features)
        self.labels = torch.LongTensor(labels)
    
    def __len__(self) -> int:
        return len(self.labels)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.features[idx], self.labels[idx]


class HospitalDataLoader:
    """
    Manages data loading and preprocessing for a hospital
    """
    
    def __init__(
        self,
        hospital_id: int,
        data_path: Optional[str] = None,
        test_size: float = 0.2,
        random_state: int = 42,
        normalize: bool = True
    ):
        """
        Args:
            hospital_id: Hospital identifier (1, 2, or 3)
            data_path: Path to CSV file (if None, uses default path)
            test_size: Proportion of data for testing
            random_state: Random seed for reproducibility
            normalize: Whether to normalize features
        """
        self.hospital_id = hospital_id
        self.test_size = test_size
        self.random_state = random_state
        self.normalize = normalize
        
        # Set data path
        if data_path is None:
            self.data_path = Path(f'data/hospital_{hospital_id}/patient_data.csv')
        else:
            self.data_path = Path(data_path)
        
        # Initialize scaler
        self.scaler = StandardScaler() if normalize else None
        
        # Storage for data
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.df = None
        
        print(f"\n🏥 Initializing data loader for Hospital {hospital_id}")
        print(f"   Data path: {self.data_path}")
    
    def load_data(self) -> pd.DataFrame:
        """Load CSV data"""
        if not self.data_path.exists():
            raise FileNotFoundError(f"Data file not found: {self.data_path}")
        
        self.df = pd.read_csv(self.data_path)
        print(f"   ✅ Loaded {len(self.df)} patient records")
        
        return self.df
    
    def validate_data(self) -> bool:
        """Validate that all required columns exist"""
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        missing_features = set(FEATURE_COLUMNS) - set(self.df.columns)
        if missing_features:
            raise ValueError(f"Missing feature columns: {missing_features}")
        
        if TARGET_COLUMN not in self.df.columns:
            raise ValueError(f"Missing target column: {TARGET_COLUMN}")
        
        print(f"   ✅ Data validation passed")
        return True
    
    def prepare_data(self):
        """
        Prepare data for training:
        1. Extract features and labels
        2. Train/test split
        3. Normalize features (if enabled)
        """
        if self.df is None:
            self.load_data()
        
        self.validate_data()
        
        # Extract features and labels
        X = self.df[FEATURE_COLUMNS].values
        y = self.df[TARGET_COLUMN].values
        
        # Train/test split
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=y  # Maintain class distribution
        )
        
        # Normalize features
        if self.normalize:
            self.X_train = self.scaler.fit_transform(self.X_train)
            self.X_test = self.scaler.transform(self.X_test)
        
        # Print statistics
        train_at_risk = np.sum(self.y_train)
        test_at_risk = np.sum(self.y_test)
        
        print(f"   📊 Train set: {len(self.y_train)} samples ({train_at_risk} at-risk, {len(self.y_train) - train_at_risk} healthy)")
        print(f"   📊 Test set: {len(self.y_test)} samples ({test_at_risk} at-risk, {len(self.y_test) - test_at_risk} healthy)")
        
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def get_train_loader(self, batch_size: int = 32, shuffle: bool = True) -> DataLoader:
        """
        Create PyTorch DataLoader for training data
        
        Args:
            batch_size: Batch size for training
            shuffle: Whether to shuffle data
        
        Returns:
            PyTorch DataLoader
        """
        if self.X_train is None:
            self.prepare_data()
        
        dataset = HealthDataset(self.X_train, self.y_train)
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=0,  # Use 0 for Windows compatibility
            pin_memory=False
        )
        
        return loader
    
    def get_test_loader(self, batch_size: int = 32, shuffle: bool = False) -> DataLoader:
        """
        Create PyTorch DataLoader for test data
        
        Args:
            batch_size: Batch size for testing
            shuffle: Whether to shuffle data
        
        Returns:
            PyTorch DataLoader
        """
        if self.X_test is None:
            self.prepare_data()
        
        dataset = HealthDataset(self.X_test, self.y_test)
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=0,
            pin_memory=False
        )
        
        return loader
    
    def get_data_statistics(self) -> dict:
        """Get statistics about the dataset"""
        if self.df is None:
            self.load_data()
        
        stats = {
            'hospital_id': self.hospital_id,
            'total_samples': len(self.df),
            'at_risk_count': int(self.df[TARGET_COLUMN].sum()),
            'at_risk_percentage': float((self.df[TARGET_COLUMN].sum() / len(self.df)) * 100),
            'feature_means': self.df[FEATURE_COLUMNS].mean().to_dict(),
            'feature_stds': self.df[FEATURE_COLUMNS].std().to_dict(),
        }
        
        if self.X_train is not None:
            stats['train_samples'] = len(self.y_train)
            stats['test_samples'] = len(self.y_test)
            stats['train_at_risk'] = int(np.sum(self.y_train))
            stats['test_at_risk'] = int(np.sum(self.y_test))
        
        return stats
    
    def get_class_weights(self) -> torch.Tensor:
        """
        Calculate class weights for handling imbalanced data
        Useful for weighted loss functions
        
        Returns:
            Tensor of class weights [weight_class_0, weight_class_1]
        """
        if self.y_train is None:
            self.prepare_data()
        
        class_counts = np.bincount(self.y_train)
        total_samples = len(self.y_train)
        
        # Inverse frequency weighting
        weights = total_samples / (len(class_counts) * class_counts)
        
        return torch.FloatTensor(weights)


def create_data_loaders(
    hospital_id: int,
    batch_size: int = 32,
    test_size: float = 0.2
) -> Tuple[DataLoader, DataLoader, dict]:
    """
    Convenience function to create train and test loaders
    
    Args:
        hospital_id: Hospital identifier
        batch_size: Batch size
        test_size: Test set proportion
    
    Returns:
        train_loader, test_loader, statistics
    """
    loader = HospitalDataLoader(
        hospital_id=hospital_id,
        test_size=test_size,
        normalize=True
    )
    
    loader.prepare_data()
    
    train_loader = loader.get_train_loader(batch_size=batch_size)
    test_loader = loader.get_test_loader(batch_size=batch_size)
    stats = loader.get_data_statistics()
    
    return train_loader, test_loader, stats


if __name__ == "__main__":
    """Test the data loader"""
    print("="*60)
    print("Testing Hospital Data Loader")
    print("="*60)
    
    # Test for Hospital 1
    try:
        train_loader, test_loader, stats = create_data_loaders(
            hospital_id=1,
            batch_size=32,
            test_size=0.2
        )
        
        print("\n📊 Dataset Statistics:")
        for key, value in stats.items():
            if isinstance(value, dict):
                print(f"   {key}:")
                for k, v in value.items():
                    print(f"      {k}: {v:.2f}" if isinstance(v, float) else f"      {k}: {v}")
            else:
                print(f"   {key}: {value:.2f}" if isinstance(value, float) else f"   {key}: {value}")
        
        # Test batch iteration
        print("\n🔄 Testing batch iteration:")
        for batch_idx, (features, labels) in enumerate(train_loader):
            print(f"   Batch {batch_idx + 1}:")
            print(f"      Features shape: {features.shape}")
            print(f"      Labels shape: {labels.shape}")
            print(f"      Sample labels: {labels[:5].tolist()}")
            
            if batch_idx >= 2:  # Test only first 3 batches
                break
        
        print("\n✅ Data loader test completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nMake sure you've generated the data by running:")
        print("   python scripts/generate_data.py")