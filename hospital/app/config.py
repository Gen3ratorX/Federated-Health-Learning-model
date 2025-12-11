"""
Configuration for hospital nodes
"""

from pydantic_settings import BaseSettings
from typing import Optional


class HospitalConfig(BaseSettings):
    """Hospital client configuration"""
    
    # Hospital Identity
    hospital_id: int = 1
    hospital_name: str = "Hospital 1"
    
    # Server Connection
    coordinator_host: str = "localhost"
    coordinator_port: int = 8080
    
    # Data Configuration
    data_path: Optional[str] = None  # If None, uses default path
    batch_size: int = 32
    test_size: float = 0.2
    
    # Training Configuration
    local_epochs: int = 5
    learning_rate: float = 0.001
    device: str = "cpu"
    
    # Privacy Configuration (optional)
    use_differential_privacy: bool = False
    epsilon: float = 1.0
    delta: float = 1e-5
    
    # Logging
    log_level: str = "INFO"
    log_file: Optional[str] = None
    
    class Config:
        env_file = ".env"
        env_prefix = "HOSPITAL_"
    
    @property
    def server_address(self) -> str:
        """Get full server address"""
        return f"{self.coordinator_host}:{self.coordinator_port}"
    
    @property
    def data_file_path(self) -> str:
        """Get data file path"""
        if self.data_path:
            return self.data_path
        return f"data/hospital_{self.hospital_id}/patient_data.csv"


# Create global config instance
config = HospitalConfig()


if __name__ == "__main__":
    """Test configuration"""
    print("Hospital Configuration:")
    print(f"  Hospital ID: {config.hospital_id}")
    print(f"  Server Address: {config.server_address}")
    print(f"  Data Path: {config.data_file_path}")
    print(f"  Batch Size: {config.batch_size}")
    print(f"  Local Epochs: {config.local_epochs}")
    print(f"  Device: {config.device}")