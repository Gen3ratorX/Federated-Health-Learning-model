# 🏥 Federated Learning for Health Risk Prediction

A production-grade Proof of Concept (POC) demonstrating **Privacy-Preserving Machine Learning** using the Flower framework. This system simulates a distributed network of hospitals collaboratively training a Global AI Model to predict cardiovascular health risk **without ever sharing patient data**.

-----

## 🌟 Key Features

  * **🛡️ Privacy-First Architecture:** Raw patient data (CSVs) remains strictly local on "Hospital" nodes. Only model weight updates (gradients) are transmitted to the Coordinator.
  * **🌍 Non-IID Data Simulation:** Simulates realistic real-world heterogeneity. "Urban" hospitals have younger populations, while "Rural" hospitals have older, higher-risk populations.
  * **⚙️ Resilient Aggregation:** Uses the **FedAvg** (Federated Averaging) algorithm to fuse insights from disparate sources into a robust global model.
  * **📊 Automated Visualization:** Includes tooling to track Loss, Accuracy, and F1 Score convergence over training rounds.
  * **🪟 Windows-Optimized:** Custom process management to handle multi-process concurrency without OS-level deadlocks.

-----

## 🏗️ System Architecture

The system follows a **Hub-and-Spoke** microservices pattern:

1.  **Coordinator (Server):** The central orchestrator. It manages the global model state, selects clients for training, and aggregates returned weights.
2.  **Hospital Nodes (Clients):** Independent processes that hold private data. They download the global model, train locally for `k` epochs, and upload the updated weights.
3.  **Shared Protocol:** A common definition of the Neural Network (PyTorch) and hyperparameters ensures compatibility.

-----

## 📂 Project Structure

```text
federated-health-poc/
├── coordinator/           # Central Aggregation Server
│   └── app/server.py      # Flower Strategy & Server Logic
├── hospital/              # Client Node Logic
│   ├── app/client.py      # Flower Client Wrapper
│   └── app/training/      # Local PyTorch Training Loop
├── shared/                # Common Codebase
│   ├── models/            # PyTorch Model Architecture
│   └── constants.py       # Configuration (Ports, Hyperparams)
├── data/                  # Local Data Storage (Simulated)
│   ├── hospital_1/        # Urban Hospital Data
│   ├── hospital_2/        # Suburban Hospital Data
│   └── hospital_3/        # Rural Hospital Data
├── scripts/               # Automation Tools
│   ├── generate_data.py   # Synthetic Data Generator
│   ├── run_simulation.py  # Process Orchestrator
│   └── visualize_results.py # Plotting Tool
├── checkpoints/           # Model Artifacts & History
├── logs/                  # Process Logs
└── requirements.txt       # Dependencies
```

-----

## 🚀 Quick Start Guide

### 1\. Prerequisites

  * Python 3.9+
  * Virtual Environment (Recommended)

### 2\. Installation

```bash
# Clone the repository
git clone <repository-url>
cd federated-health-poc

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt
pip install matplotlib  # For visualization
```

### 3\. Generate Synthetic Data

This script creates unique datasets for 3 hospitals using **Non-IID** distributions (skewed age/disease rates).

```bash
python scripts/generate_data.py
```

### 4\. Run the Simulation

This orchestrator launches 1 Coordinator and 3 Hospital Clients automatically. It handles the handshake and training rounds.

```bash
# Run 3 hospitals for 5 rounds
python scripts/run_simulation.py --hospitals 3 --rounds 5
```

### 5\. Visualize Results

Once the simulation completes, generate the performance charts.

```bash
python scripts/visualize_results.py
```

*Check `checkpoints/training_curves.png` for the output.*

-----

## 📊 Performance & Insights

In a 3-Hospital simulation (Urban, Suburban, Rural), the model demonstrates the power of collaborative learning:

| Metric | Start (Round 1) | End (Round 5) | Improvement |
| :--- | :--- | :--- | :--- |
| **Accuracy** | \~59.6% | **86.7%** | +27.1% |
| **F1 Score** | \~0.36 | **0.60** | +0.24 |
| **Loss** | 0.609 | **0.344** | -43.6% |

### The "Rural Effect"

Hospital 3 (Rural) contains a significantly higher percentage of "At-Risk" patients (22% vs 7%).

  * **Isolated:** An Urban hospital training alone fails to identify high-risk rural patients.
  * **Federated:** By including Hospital 3 in the federation, the Global Model's **F1 Score** (sensitivity to disease) improved drastically, proving that **data diversity \> data quantity**.

-----

## 🛠️ Technical Implementation Details

### The Model

A customized Multi-Layer Perceptron (MLP) optimized for tabular health data:

  * **Input:** 10 Clinical Features (Age, BMI, BP, Glucose, etc.)
  * **Hidden Layers:** 64 -\> 32 Units (with BatchNorm & Dropout)
  * **Output:** Binary Classification (Healthy vs. At-Risk)
  * **Optimizer:** Adam (`lr=0.001`) with CrossEntropyLoss.

### Deadlock Prevention

Running multiple PyTorch processes locally on Windows often causes deadlocks due to pipe buffer limits and worker spawning. This project solves this by:

1.  Redirecting subprocess `stdout` to file logs (`logs/hospital_x.log`).
2.  Forcing `num_workers=0` in PyTorch DataLoaders.

-----

## 🔮 Future Roadmap

  * **Differential Privacy:** Implement Opacus to add noise to gradients, mathematically guaranteeing patient anonymity.
  * **Secure Aggregation:** Encrypt weights so even the Coordinator cannot see individual hospital updates.
  * **Deploy to Cloud:** Dockerize nodes to run on AWS/Azure across different regions.

-----

## 📝 License

Distributed under the MIT License. See `LICENSE` for more information.