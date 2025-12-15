🐳 Docker Deployment Guide
Prerequisites
Docker Desktop installed and running

PowerShell (for Windows users) or Terminal

Virtualization Enabled in BIOS

4GB+ RAM allocated to Docker Desktop

🚀 Quick Start
1. Setup Offline Packages (One-Time Setup)
Since PyTorch is large and internet builds can fail, we use an "Air-Gap" strategy.

Create a packages folder:

Bash

mkdir packages
Download the required Linux wheel files manually:

Download Torch (CPU) -> Save to packages/

Download TorchVision (CPU) -> Save to packages/

2. Generate Synthetic Data
Create the patient data folders for the hospitals:

Bash

python scripts/generate_data.py
3. Build & Launch
This will build the containers using the local package files (very fast).

Bash

# Build and start in detached mode
docker-compose up -d --build

# View the logs to ensure startup
docker-compose logs -f
Wait until you see: Uvicorn running on http://0.0.0.0:8000

4. Trigger Training 🔫
The system starts in an IDLE state. You must trigger the training via the API.

Option A: Browser (Easy)

Go to: http://localhost:8000/docs

Open POST /api/training/start

Click Try it out -> Execute.

Option B: PowerShell

PowerShell

Invoke-RestMethod -Uri "http://localhost:8000/api/training/start" `
  -Method Post `
  -ContentType "application/json" `
  -Body '{"num_rounds": 5, "min_clients": 3}'
🏗️ Architecture
Code snippet

graph TD
    User[User / Admin] -->|HTTP API (Port 8000)| Coord[Coordinator Container]
    
    subgraph Docker Network
        Coord -->|gRPC (Port 8080)| H1[Hospital 1]
        Coord -->|gRPC (Port 8080)| H2[Hospital 2]
        Coord -->|gRPC (Port 8080)| H3[Hospital 3]
    end

    H1 -->|Mount| D1[(Data: Urban)]
    H2 -->|Mount| D2[(Data: Suburban)]
    H3 -->|Mount| D3[(Data: Rural)]
Coordinator: Hosts the Flower Server and the FastAPI Control Plane.

Hospitals: Dumb clients that wait for instructions. They use CPU-only PyTorch for efficiency.

🛠️ Management Commands
Check Training Status
PowerShell

Invoke-RestMethod -Uri "http://localhost:8000/api/training/status"
Stop/Abort Training
PowerShell

Invoke-RestMethod -Uri "http://localhost:8000/api/training/stop" -Method Post
View Real-time Logs
Bash

# Follow coordinator logs
docker-compose logs -f coordinator

# Follow a specific hospital
docker-compose logs -f hospital_1
Shutdown System
Bash

# Stop containers and remove network
docker-compose down
🔧 Troubleshooting
"Executable file not found" / "exec: 3"
Cause: Mismatch between Dockerfile CMD and docker-compose command.

Fix: Ensure docker-compose.yml has the full command explicitly:

YAML

command: ["python", "client.py", "1", "--server", "coordinator:8080"]
"ReadTimeoutError" during Build
Cause: Slow internet connection trying to download PyTorch (200MB+).

Fix: Use the Air-Gap Strategy. Download files to packages/ locally and use the updated Dockerfile that COPYs them in.

"Connection Refused"
Cause: The Coordinator container isn't fully ready yet.

Fix: Wait 10 seconds. The hospitals are configured to retry automatically.

System Out of Memory (OOM)
Cause: Running 3 Neural Network trainings simultaneously on one machine.

Fix: Increase Docker Desktop memory limit to 6GB in Settings > Resources.

☁️ Production Notes
This setup is optimized for a Proof of Concept (POC). For production:

GPU Support: Switch the Hospital Dockerfile base image to pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime.

Security: Enable TLS (SSL) on the Flower gRPC connection (currently insecure).

Orchestration: Deploy to Kubernetes (EKS/AKS/GKE) so hospitals run on different physical nodes.