"""
Federated Learning Simulation Runner
Runs coordinator and multiple hospital clients locally for testing
"""

import subprocess
import time
import sys
import signal
from pathlib import Path
from typing import List

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from shared.constants import NUM_FL_ROUNDS, COORDINATOR_PORT


class FederatedLearningSimulation:
    """
    Manages local FL simulation with multiple processes
    """
    
    def __init__(
        self,
        num_hospitals: int = 3,
        num_rounds: int = NUM_FL_ROUNDS,
        coordinator_host: str = "localhost",
        coordinator_port: int = COORDINATOR_PORT
    ):
        """
        Args:
            num_hospitals: Number of hospital clients to simulate
            num_rounds: Number of FL rounds
            coordinator_host: Coordinator host address
            coordinator_port: Coordinator port
        """
        self.num_hospitals = num_hospitals
        self.num_rounds = num_rounds
        self.coordinator_host = coordinator_host
        self.coordinator_port = coordinator_port
        self.server_address = f"{coordinator_host}:{coordinator_port}"
        
        self.processes: List[subprocess.Popen] = []
        
    def start_coordinator(self):
        """Start the coordinator server"""
        print(f"\n{'='*60}")
        print("🚀 STARTING COORDINATOR SERVER")
        print(f"{'='*60}")
        print(f"Address: {self.server_address}")
        print(f"Rounds: {self.num_rounds}")
        print(f"{'='*60}\n")
        
        # Start coordinator in a subprocess
        coordinator_cmd = [
            sys.executable,
            "coordinator/app/server.py",
            "--host", self.coordinator_host,
            "--port", str(self.coordinator_port),
            "--rounds", str(self.num_rounds)
        ]
        
        coordinator_process = subprocess.Popen(
            coordinator_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )
        
        self.processes.append(coordinator_process)
        
        # Give coordinator time to start
        print("⏳ Waiting for coordinator to initialize...")
        time.sleep(3)
        print("✅ Coordinator started\n")
        
        return coordinator_process
    
    def start_hospital_client(self, hospital_id: int):
        """
        Start a hospital client
        """
        print(f"🏥 Starting Hospital {hospital_id} client...")
        
        # Create a log file for this hospital
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        log_file = open(log_dir / f"hospital_{hospital_id}.log", "w", encoding='utf-8')
        
        client_cmd = [
            sys.executable,
            "hospital/app/client.py",
            str(hospital_id),
            "--server", self.server_address
        ]
        
        # KEY FIX: Write to file, not PIPE. This prevents buffer deadlock.
        client_process = subprocess.Popen(
            client_cmd,
            stdout=log_file,
            stderr=subprocess.STDOUT, 
            text=True,
            bufsize=1
        )
        
        self.processes.append(client_process)
        
        # Give client time to connect
        time.sleep(3) # Increased wait time slightly
        
        return client_process
    
    def start_all_clients(self):
        """Start all hospital clients"""
        print(f"\n{'='*60}")
        print(f"🏥 STARTING {self.num_hospitals} HOSPITAL CLIENTS")
        print(f"{'='*60}\n")
        
        for hospital_id in range(1, self.num_hospitals + 1):
            self.start_hospital_client(hospital_id)
        
        print(f"\n✅ All {self.num_hospitals} hospitals connected\n")
    
    def monitor_processes(self):
        """Monitor and display output from all processes"""
        print(f"\n{'='*60}")
        print("📊 FEDERATED LEARNING IN PROGRESS")
        print(f"{'='*60}")
        print("Monitoring coordinator output...")
        print(f"(Each round takes ~30-60 seconds - please be patient)\n")
        
        try:
            # Monitor coordinator output (first process)
            coordinator = self.processes[0]
            current_round = 0
            
            for line in coordinator.stdout:
                print(line, end='', flush=True)
                
                # Track rounds
                if "[ROUND" in line:
                    try:
                        current_round = int(line.split("ROUND")[1].split("]")[0].strip())
                        print(f"⏳ Round {current_round}/{self.num_rounds} - Training in progress...", flush=True)
                    except:
                        pass
                
                # Check if all processes are still running
                if coordinator.poll() is not None:
                    break
            
            # Wait for coordinator to finish
            coordinator.wait()
            
            print(f"\n{'='*60}")
            print("✅ FEDERATED LEARNING COMPLETED")
            print(f"{'='*60}\n")
            
        except KeyboardInterrupt:
            print("\n\n⚠️  Simulation interrupted by user")
            self.cleanup()
    
    def cleanup(self):
        """Terminate all processes"""
        print("\n🧹 Cleaning up processes...")
        
        for process in self.processes:
            if process.poll() is None:  # Process still running
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
        
        print("✅ Cleanup complete\n")
    
    def run(self):
        """Run the complete FL simulation"""
        try:
            # Start coordinator
            self.start_coordinator()
            
            # Start hospital clients
            self.start_all_clients()
            
            # Monitor progress
            self.monitor_processes()
            
            # Cleanup
            self.cleanup()
            
            print("\n" + "="*60)
            print("📊 SIMULATION SUMMARY")
            print("="*60)
            print(f"Hospitals participated: {self.num_hospitals}")
            print(f"FL rounds completed: {self.num_rounds}")
            print(f"Results saved in: checkpoints/")
            print("="*60 + "\n")
            
        except Exception as e:
            print(f"\n❌ Error during simulation: {e}")
            self.cleanup()
            raise


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Run Federated Learning Simulation"
    )
    parser.add_argument(
        "--hospitals",
        type=int,
        default=3,
        help="Number of hospitals to simulate (default: 3)"
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=NUM_FL_ROUNDS,
        help=f"Number of FL rounds (default: {NUM_FL_ROUNDS})"
    )
    parser.add_argument(
        "--host",
        type=str,
        default="localhost",
        help="Coordinator host (default: localhost)"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=COORDINATOR_PORT,
        help=f"Coordinator port (default: {COORDINATOR_PORT})"
    )
    
    args = parser.parse_args()
    
    # Validate hospitals
    if args.hospitals < 2:
        print("❌ Error: Must have at least 2 hospitals")
        sys.exit(1)
    
    if args.hospitals > 3:
        print("⚠️  Warning: Only 3 hospital datasets available")
        print(f"   Using {min(args.hospitals, 3)} hospitals")
        args.hospitals = 3
    
    # Create and run simulation
    print("\n" + "="*60)
    print("🏥 FEDERATED LEARNING POC - SIMULATION")
    print("="*60)
    print(f"Configuration:")
    print(f"  Hospitals: {args.hospitals}")
    print(f"  FL Rounds: {args.rounds}")
    print(f"  Coordinator: {args.host}:{args.port}")
    print("="*60 + "\n")
    
    simulation = FederatedLearningSimulation(
        num_hospitals=args.hospitals,
        num_rounds=args.rounds,
        coordinator_host=args.host,
        coordinator_port=args.port
    )
    
    simulation.run()


if __name__ == "__main__":
    main()