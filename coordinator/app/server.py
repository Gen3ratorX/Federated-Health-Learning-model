"""
Flower Server Implementation for Federated Learning Coordinator
Manages aggregation and orchestration of federated learning rounds
"""

import flwr as fl
from flwr.common import (
    FitRes,
    Parameters,
    Scalar,
    parameters_to_ndarrays,
    ndarrays_to_parameters,
)
from flwr.server.client_manager import ClientManager
from flwr.server.strategy import FedAvg
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import torch
import json
from pathlib import Path
from datetime import datetime
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from shared.models.base_model import create_model
from shared.constants import (
    NUM_FL_ROUNDS,
    MIN_AVAILABLE_CLIENTS,
    MIN_FIT_CLIENTS,
    MIN_EVAL_CLIENTS,
    FRACTION_FIT,
    FRACTION_EVALUATE,
    DEVICE,
    CHECKPOINTS_DIR,
    GLOBAL_MODEL_PATH
)


class HealthFederatedStrategy(FedAvg):
    """
    Custom Federated Averaging Strategy for Health Risk Prediction
    
    Extends Flower's FedAvg with:
    - Custom model initialization
    - Metrics aggregation and logging
    - Model checkpointing
    - Per-round statistics
    """
    
    def __init__(
        self,
        fraction_fit: float = FRACTION_FIT,
        fraction_evaluate: float = FRACTION_EVALUATE,
        min_fit_clients: int = MIN_FIT_CLIENTS,
        min_evaluate_clients: int = MIN_EVAL_CLIENTS,
        min_available_clients: int = MIN_AVAILABLE_CLIENTS,
        save_checkpoints: bool = True,
        checkpoint_dir: str = CHECKPOINTS_DIR
    ):
        """
        Args:
            fraction_fit: Fraction of clients to sample for training
            fraction_evaluate: Fraction of clients to sample for evaluation
            min_fit_clients: Minimum number of clients for training
            min_evaluate_clients: Minimum number of clients for evaluation
            min_available_clients: Minimum number of clients that need to connect
            save_checkpoints: Whether to save model checkpoints
            checkpoint_dir: Directory to save checkpoints
        """
        
        # Initialize global model
        self.model = create_model(device=DEVICE)
        initial_parameters = ndarrays_to_parameters(
            [param.cpu().detach().numpy() for param in self.model.parameters()]
        )
        
        super().__init__(
            fraction_fit=fraction_fit,
            fraction_evaluate=fraction_evaluate,
            min_fit_clients=min_fit_clients,
            min_evaluate_clients=min_evaluate_clients,
            min_available_clients=min_available_clients,
            initial_parameters=initial_parameters,
        )
        
        self.save_checkpoints = save_checkpoints
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Tracking
        self.current_round = 0
        self.history = {
            'rounds': [],
            'train_loss': [],
            'train_accuracy': [],
            'eval_loss': [],
            'eval_accuracy': [],
            'eval_f1_score': [],
            'num_clients': []
        }
        
        print(f"\n{'='*60}")
        print("FEDERATED LEARNING STRATEGY INITIALIZED")
        print(f"{'='*60}")
        print(f"Strategy: FedAvg (Federated Averaging)")
        print(f"Minimum clients: {min_available_clients}")
        print(f"Clients per training round: {min_fit_clients}")
        print(f"Clients per evaluation round: {min_evaluate_clients}")
        print(f"Checkpoint directory: {self.checkpoint_dir}")
        print(f"{'='*60}\n")
    
    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, FitRes]],
        failures: List[Union[Tuple[fl.server.client_proxy.ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """
        Aggregate training results from multiple hospitals
        
        Args:
            server_round: Current FL round number
            results: List of (client, FitRes) tuples with training results
            failures: List of failed clients
        
        Returns:
            Aggregated parameters and metrics
        """
        
        self.current_round = server_round
        
        print(f"\n{'='*60}")
        print(f"ROUND {server_round}: AGGREGATING TRAINING RESULTS")
        print(f"{'='*60}")
        print(f"Successful hospitals: {len(results)}")
        print(f"Failed hospitals: {len(failures)}")
        
        if not results:
            print("❌ No results to aggregate!")
            return None, {}
        
        # Aggregate parameters using FedAvg (weighted by number of examples)
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(
            server_round, results, failures
        )
        
        # Extract and aggregate custom metrics
        total_examples = 0
        weighted_loss = 0.0
        weighted_accuracy = 0.0
        hospital_metrics = []
        
        for client, fit_res in results:
            num_examples = fit_res.num_examples
            metrics = fit_res.metrics
            
            total_examples += num_examples
            weighted_loss += metrics.get('train_loss', 0) * num_examples
            weighted_accuracy += metrics.get('train_accuracy', 0) * num_examples
            
            hospital_metrics.append({
                'hospital_id': metrics.get('hospital_id', 'unknown'),
                'num_examples': num_examples,
                'train_loss': metrics.get('train_loss', 0),
                'train_accuracy': metrics.get('train_accuracy', 0)
            })
        
        # Calculate weighted averages
        avg_loss = weighted_loss / total_examples if total_examples > 0 else 0
        avg_accuracy = weighted_accuracy / total_examples if total_examples > 0 else 0
        
        # Print per-hospital results
        print(f"\n📊 Per-Hospital Training Results:")
        for hm in hospital_metrics:
            print(f"   Hospital {hm['hospital_id']}:")
            print(f"      Samples: {hm['num_examples']}")
            print(f"      Loss: {hm['train_loss']:.4f}")
            print(f"      Accuracy: {hm['train_accuracy']:.4f}")
        
        print(f"\n📊 Aggregated Training Metrics:")
        print(f"   Total samples: {total_examples}")
        print(f"   Weighted avg loss: {avg_loss:.4f}")
        print(f"   Weighted avg accuracy: {avg_accuracy:.4f}")
        
        # Store metrics
        self.history['rounds'].append(server_round)
        self.history['train_loss'].append(avg_loss)
        self.history['train_accuracy'].append(avg_accuracy)
        self.history['num_clients'].append(len(results))
        
        # Update global model for checkpointing
        if aggregated_parameters:
            self._update_global_model(aggregated_parameters)
        
        # Save checkpoint
        if self.save_checkpoints and aggregated_parameters:
            self._save_checkpoint(server_round, avg_loss, avg_accuracy)
        
        print(f"{'='*60}\n")
        
        # Add metrics to return dictionary
        aggregated_metrics['train_loss_avg'] = avg_loss
        aggregated_metrics['train_accuracy_avg'] = avg_accuracy
        
        return aggregated_parameters, aggregated_metrics
    
    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.EvaluateRes]],
        failures: List[Union[Tuple[fl.server.client_proxy.ClientProxy, fl.common.EvaluateRes], BaseException]],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        """
        Aggregate evaluation results from multiple hospitals
        
        Args:
            server_round: Current FL round number
            results: List of evaluation results
            failures: List of failed evaluations
        
        Returns:
            Aggregated loss and metrics
        """
        
        print(f"\n{'='*60}")
        print(f"ROUND {server_round}: AGGREGATING EVALUATION RESULTS")
        print(f"{'='*60}")
        
        if not results:
            print("❌ No evaluation results!")
            return None, {}
        
        # Aggregate losses (weighted by number of examples)
        total_examples = 0
        weighted_loss = 0.0
        weighted_accuracy = 0.0
        weighted_f1 = 0.0
        hospital_evals = []
        
        for client, evaluate_res in results:
            num_examples = evaluate_res.num_examples
            loss = evaluate_res.loss
            metrics = evaluate_res.metrics
            
            total_examples += num_examples
            weighted_loss += loss * num_examples
            weighted_accuracy += metrics.get('accuracy', 0) * num_examples
            weighted_f1 += metrics.get('f1_score', 0) * num_examples
            
            hospital_evals.append({
                'hospital_id': metrics.get('hospital_id', 'unknown'),
                'num_examples': num_examples,
                'loss': loss,
                'accuracy': metrics.get('accuracy', 0),
                'precision': metrics.get('precision', 0),
                'recall': metrics.get('recall', 0),
                'f1_score': metrics.get('f1_score', 0)
            })
        
        # Calculate weighted averages
        avg_loss = weighted_loss / total_examples if total_examples > 0 else 0
        avg_accuracy = weighted_accuracy / total_examples if total_examples > 0 else 0
        avg_f1 = weighted_f1 / total_examples if total_examples > 0 else 0
        
        # Print per-hospital results
        print(f"\n📊 Per-Hospital Evaluation Results:")
        for he in hospital_evals:
            print(f"   Hospital {he['hospital_id']}:")
            print(f"      Samples: {he['num_examples']}")
            print(f"      Loss: {he['loss']:.4f}")
            print(f"      Accuracy: {he['accuracy']:.4f}")
            print(f"      F1 Score: {he['f1_score']:.4f}")
        
        print(f"\n📊 Aggregated Evaluation Metrics:")
        print(f"   Total samples: {total_examples}")
        print(f"   Weighted avg loss: {avg_loss:.4f}")
        print(f"   Weighted avg accuracy: {avg_accuracy:.4f}")
        print(f"   Weighted avg F1 score: {avg_f1:.4f}")
        print(f"{'='*60}\n")
        
        # Store metrics
        self.history['eval_loss'].append(avg_loss)
        self.history['eval_accuracy'].append(avg_accuracy)
        self.history['eval_f1_score'].append(avg_f1)
        
        # Prepare aggregated metrics
        aggregated_metrics = {
            'accuracy': avg_accuracy,
            'f1_score': avg_f1,
            'num_hospitals': len(results)
        }
        
        return avg_loss, aggregated_metrics
    
    def _update_global_model(self, parameters: Parameters):
        """Update the global model with new parameters"""
        params_list = parameters_to_ndarrays(parameters)
        params_dict = zip(self.model.parameters(), params_list)
        for param, new_param in params_dict:
            param.data = torch.tensor(new_param, dtype=param.dtype).to(DEVICE)
    
    def _save_checkpoint(self, round_num: int, loss: float, accuracy: float):
        """Save model checkpoint"""
        checkpoint_path = self.checkpoint_dir / f"global_model_round_{round_num}.pth"
        
        checkpoint = {
            'round': round_num,
            'model_state_dict': self.model.state_dict(),
            'loss': loss,
            'accuracy': accuracy,
            'timestamp': datetime.now().isoformat(),
            'history': self.history
        }
        
        torch.save(checkpoint, checkpoint_path)
        
        # Also save as "latest"
        latest_path = self.checkpoint_dir / "global_model_latest.pth"
        torch.save(checkpoint, latest_path)
        
        print(f"💾 Checkpoint saved: {checkpoint_path}")
    
    def save_final_results(self):
        """Save final training history and results"""
        results_path = self.checkpoint_dir / "training_history.json"
        
        with open(results_path, 'w') as f:
            json.dump(self.history, f, indent=2)
        
        print(f"\n💾 Training history saved: {results_path}")
        
        # Print final summary
        print(f"\n{'='*60}")
        print("FINAL RESULTS SUMMARY")
        print(f"{'='*60}")
        print(f"Total rounds: {len(self.history['rounds'])}")
        print(f"Final train loss: {self.history['train_loss'][-1]:.4f}")
        print(f"Final train accuracy: {self.history['train_accuracy'][-1]:.4f}")
        if self.history['eval_loss']:
            print(f"Final eval loss: {self.history['eval_loss'][-1]:.4f}")
            print(f"Final eval accuracy: {self.history['eval_accuracy'][-1]:.4f}")
            print(f"Final eval F1 score: {self.history['eval_f1_score'][-1]:.4f}")
        print(f"{'='*60}\n")


def create_strategy() -> HealthFederatedStrategy:
    """
    Factory function to create the FL strategy
    
    Returns:
        Initialized HealthFederatedStrategy
    """
    return HealthFederatedStrategy(
        fraction_fit=FRACTION_FIT,
        fraction_evaluate=FRACTION_EVALUATE,
        min_fit_clients=MIN_FIT_CLIENTS,
        min_evaluate_clients=MIN_EVAL_CLIENTS,
        min_available_clients=MIN_AVAILABLE_CLIENTS,
        save_checkpoints=True
    )


def start_server(
    server_address: str = "0.0.0.0:8080",
    num_rounds: int = NUM_FL_ROUNDS,
    strategy: Optional[HealthFederatedStrategy] = None
):
    """
    Start the Flower server (coordinator)
    
    Args:
        server_address: Address to bind server (host:port)
        num_rounds: Number of FL rounds to run
        strategy: FL strategy (if None, creates default)
    """
    
    print(f"\n{'='*60}")
    print("STARTING FEDERATED LEARNING COORDINATOR")
    print(f"{'='*60}")
    print(f"Server address: {server_address}")
    print(f"Number of rounds: {num_rounds}")
    print(f"Waiting for hospitals to connect...")
    print(f"{'='*60}\n")
    
    # Create strategy if not provided
    if strategy is None:
        strategy = create_strategy()
    
    # Configure server
    config = fl.server.ServerConfig(num_rounds=num_rounds)
    
    try:
        # Start Flower server
        history = fl.server.start_server(
            server_address=server_address,
            config=config,
            strategy=strategy
        )
        
        # Save final results
        strategy.save_final_results()
        
        print("\n✅ Federated learning completed successfully!")
        
        return history
        
    except Exception as e:
        print(f"\n❌ Error starting server: {e}")
        raise


if __name__ == "__main__":
    """
    Run this script to start the coordinator
    Usage: python coordinator/app/server.py
    """
    import argparse
    
    parser = argparse.ArgumentParser(description="Start Flower server (coordinator)")
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Server host address (default: 0.0.0.0)"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8080,
        help="Server port (default: 8080)"
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=NUM_FL_ROUNDS,
        help=f"Number of FL rounds (default: {NUM_FL_ROUNDS})"
    )
    
    args = parser.parse_args()
    
    server_address = f"{args.host}:{args.port}"
    
    # Start server
    start_server(
        server_address=server_address,
        num_rounds=args.rounds
    )