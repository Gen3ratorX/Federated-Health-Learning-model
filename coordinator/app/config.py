"""
Configuration for the Federated Learning Coordinator
"""

from pydantic_settings import BaseSettings
from typing import Optional


class CoordinatorConfig(BaseSettings):
    """Coordinator server configuration"""
    
    # Server Configuration
    host: str = "0.0.0.0"
    port: int = 8080
    api_port: int = 8000
    
    # Federated Learning Configuration
    num_rounds: int = 10
    min_available_clients: int = 2
    min_fit_clients: int = 2
    min_eval_clients: int = 2
    fraction_fit: float = 1.0
    fraction_evaluate: float = 1.0
    
    # Model Configuration
    device: str = "cpu"
    
    # Checkpoint Configuration
    save_checkpoints: bool = True
    checkpoint_dir: str = "checkpoints"
    
    # Logging Configuration
    log_level: str = "INFO"
    log_file: Optional[str] = "monitoring/logs/coordinator.log"
    
    # MLflow Configuration (optional)
    use_mlflow: bool = False
    mlflow_tracking_uri: str = "http://localhost:5000"
    mlflow_experiment_name: str = "federated_health_poc"
    
    class Config:
        env_file = ".env"
        env_prefix = "COORDINATOR_"
    
    @property
    def server_address(self) -> str:
        """Get full server address"""
        return f"{self.host}:{self.port}"


# Create global config instance
config = CoordinatorConfig()


if __name__ == "__main__":
    """Test configuration"""
    print("Coordinator Configuration:")
    print(f"  Server Address: {config.server_address}")
    print(f"  API Port: {config.api_port}")
    print(f"  Number of Rounds: {config.num_rounds}")
    print(f"  Min Clients: {config.min_available_clients}")
    print(f"  Checkpoint Dir: {config.checkpoint_dir}")
    print(f"  Use MLflow: {config.use_mlflow}")