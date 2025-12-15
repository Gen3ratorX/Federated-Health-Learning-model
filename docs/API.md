
# 🔌 API Documentation

This project exposes two distinct API layers:

1.  **Doctor's Inference API:** A Python function for local predictions at the hospital.
2.  **Coordinator Admin API:** A REST API to manage the federated training lifecycle.

-----

## 🩺 Part 1: Doctor's Inference API (Client-Side)

This function simulates a doctor's interface. It runs locally on the hospital node using the downloaded global model.

### `predict_risk(patient: PatientData)`

**Location:** `hospital/app/inference.py`

#### 📥 Input: Patient Object

Expects raw, unscaled clinical data.

| Field | Type | Description |
| :--- | :--- | :--- |
| `age` | `int` | Patient age (18-90) |
| `bmi` | `float` | Body Mass Index |
| `bp_systolic` | `int` | Upper blood pressure (mmHg) |
| `bp_diastolic` | `int` | Lower blood pressure (mmHg) |
| `cholesterol` | `int` | Total cholesterol (mg/dL) |
| `glucose` | `int` | Fasting blood glucose (mg/dL) |
| `heart_rate` | `int` | Resting heart rate (bpm) |
| `smoking` | `int` | 1=Yes, 0=No |
| `diabetes` | `int` | 1=Yes, 0=No |
| `family_history`| `int` | 1=Yes, 0=No |

#### 📤 Output: Risk Assessment

```json
{
  "risk_score": 89.45,
  "label": "HIGH RISK",
  "color": "#DC2626",
  "detail": "Inference successful"
}
```

-----

## 🛠️ Part 2: Coordinator Admin API (Server-Side)

**Base URL:** `http://localhost:8000` (Default)
**Docs UI:** `http://localhost:8000/docs`

This REST API manages the central Coordinator, allowing administrators to start/stop training, monitor hospital status, and download models.

### 🖥️ Simulation & System

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/dashboard` | Returns the HTML dashboard for visualizing the federation. |
| `GET` | `/health` | Health check to verify the API is running. |
| `GET` | `/api/system/info` | Returns system resources (CPU/RAM) and Python version. |
| `POST` | `/api/simulation/launch_hospitals` | **Simulation Only:** Spawns hospital client processes automatically. |
| `POST` | `/api/simulation/stop_hospitals` | **Simulation Only:** Terminates all hospital client processes. |
| `GET` | `/api/hospitals/status` | List connected hospitals and their current states. |

### 🚀 Training Control

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `POST` | `/api/training/start` | Initiates the Federated Learning rounds. Accepts `StartTrainingRequest`. |
| `POST` | `/api/training/stop` | Aborts the current training session. |
| `GET` | `/api/training/status` | Returns current round, phase (e.g., `FIT`, `EVAL`), and connection counts. |

### 📊 Metrics & Results

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/api/metrics` | Returns raw metrics history (accuracy/loss) for all rounds. |
| `GET` | `/api/metrics/summary` | Returns aggregated statistics (best accuracy, average loss). |

### 💾 Model Checkpoints

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/api/checkpoints` | Lists all saved global model files. |
| `GET` | `/api/checkpoints/latest/download` | Download the most recent global model (`.pth` file). |
| `GET` | `/api/checkpoints/{round}/download` | Download the model snapshot from a specific round. |

### 📦 Schemas

**StartTrainingRequest**

```json
{
  "min_clients": 2,
  "num_rounds": 5,
  "fraction_fit": 1.0
}
```

**TrainingStatus**

```json
{
  "state": "training",
  "current_round": 3,
  "total_rounds": 5,
  "connected_clients": 3
}
```