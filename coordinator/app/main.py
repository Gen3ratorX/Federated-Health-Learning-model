"""
FastAPI Admin Backend for Federated Learning Coordinator
Provides REST APIs for monitoring and control, and serves the Dashboard UI.
Includes Simulation Mode to auto-launch hospital clients locally.
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse, JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Dict, List
from shared.constants import SCALING_FACTORS
import json
import threading
from pathlib import Path
from datetime import datetime
import sys
import subprocess
import os
import torch
import torch.nn.functional as F
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from coordinator.app.server import start_server, create_strategy
from shared.constants import NUM_FL_ROUNDS, COORDINATOR_PORT, API_PORT

# 👇 UPDATED IMPORT
from shared.models.base_model import HealthRiskModel 

# ============================================================================
# Lifespan Management
# ============================================================================

from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup/shutdown"""
    print("\n" + "="*60)
    print("FEDERATED LEARNING ADMIN API")
    print("="*60)
    print(f"API Server: http://localhost:{API_PORT}")
    print(f"FL Coordinator: 0.0.0.0:{COORDINATOR_PORT}")
    print(f"Documentation: http://localhost:{API_PORT}/docs")
    print(f"Dashboard: http://localhost:{API_PORT}/dashboard")
    print("="*60 + "\n")
    yield
    print("\nAdmin API shutdown complete. Cleaning up simulation processes...")
    stop_hospitals_internal() 
    print("Cleanup done.\n")

app = FastAPI(
    title="Federated Learning Admin API",
    description="REST API for managing federated learning coordinator",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state
fl_state = {
    "status": "idle",
    "current_round": 0,
    "total_rounds": 0,
    "start_time": None,
    "end_time": None,
    "error": None,
    "thread": None
}

# ============================================================================
# Request/Response Models
# ============================================================================

class StartTrainingRequest(BaseModel):
    num_rounds: int = NUM_FL_ROUNDS
    min_clients: int = 2

class TrainingStatus(BaseModel):
    status: str
    current_round: int
    total_rounds: int
    start_time: Optional[str]
    end_time: Optional[str]
    error: Optional[str]

class MetricsResponse(BaseModel):
    rounds: List[int]
    train_loss: List[float]
    train_accuracy: List[float]
    eval_loss: List[float]
    eval_accuracy: List[float]
    eval_f1_score: List[float]

# ============================================================================
# Helper Functions
# ============================================================================

def run_fl_training(num_rounds: int):
    global fl_state
    try:
        fl_state["status"] = "running"
        fl_state["start_time"] = datetime.now().isoformat()
        fl_state["total_rounds"] = num_rounds 
        
        start_server(
            server_address=f"0.0.0.0:{COORDINATOR_PORT}",
            num_rounds=num_rounds
        )
        fl_state["status"] = "completed"
        fl_state["end_time"] = datetime.now().isoformat()
    except Exception as e:
        fl_state["status"] = "error"
        fl_state["error"] = str(e)
        fl_state["end_time"] = datetime.now().isoformat()

def get_training_history() -> Optional[Dict]:
    history_path = Path("checkpoints/training_history.json")
    if history_path.exists():
        with open(history_path) as f:
            return json.load(f)
    return None

def get_available_checkpoints() -> List[Dict]:
    checkpoints_dir = Path("checkpoints")
    if not checkpoints_dir.exists():
        return []
    checkpoints = []
    for file in checkpoints_dir.glob("global_model_round_*.pth"):
        checkpoints.append({
            "filename": file.name,
            "round": int(file.stem.split("_")[-1]),
            "size_kb": file.stat().st_size / 1024,
            "created_at": datetime.fromtimestamp(file.stat().st_mtime).isoformat()
        })
    return sorted(checkpoints, key=lambda x: x["round"])

# ============================================================================
# Dashboard Endpoint
# ============================================================================

@app.get("/dashboard", response_class=HTMLResponse)
async def get_dashboard():
    try:
        dashboard_path = Path("dashboard.html")
        if not dashboard_path.exists():
             dashboard_path = Path("../dashboard.html")
        if not dashboard_path.exists():
             dashboard_path = project_root / "dashboard.html"

        if dashboard_path.exists():
            with open(dashboard_path, "r", encoding="utf-8") as f:
                return f.read()
        else:
            return "<h1>Error: dashboard.html not found. Please place it in the project root.</h1>"
    except Exception as e:
        return f"<h1>Error loading dashboard: {str(e)}</h1>"

# ============================================================================
# SIMULATION CONTROLS
# ============================================================================

hospital_processes = []

@app.post("/api/simulation/launch_hospitals")
async def launch_hospitals(count: int = 3):
    global hospital_processes
    hospital_processes = [p for p in hospital_processes if p.poll() is None]
    
    if hospital_processes:
        return {"message": f"{len(hospital_processes)} hospitals are already running."}

    base_dir = project_root
    client_script = base_dir / "hospital" / "app" / "client.py"

    if not client_script.exists():
        raise HTTPException(status_code=500, detail=f"Client script not found at {client_script}")

    try:
        launched = 0
        for i in range(1, count + 1):
            kwargs = {"cwd": str(base_dir)}
            if sys.platform == "win32":
                kwargs["creationflags"] = subprocess.CREATE_NEW_CONSOLE

            proc = subprocess.Popen(
                [sys.executable, str(client_script), str(i), "--server", f"localhost:{COORDINATOR_PORT}"],
                **kwargs
            )
            hospital_processes.append(proc)
            launched += 1
            
        return {"message": f"Successfully launched {launched} hospital clients!"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def stop_hospitals_internal():
    global hospital_processes
    count = 0
    for proc in hospital_processes:
        try:
            if proc.poll() is None:
                proc.terminate()
                count += 1
        except:
            pass
    hospital_processes = []
    return count

@app.post("/api/simulation/stop_hospitals")
async def stop_hospitals():
    count = stop_hospitals_internal()
    return {"message": f"Stopped {count} hospital clients."}

# ============================================================================
# 🩺 DOCTOR / INFERENCE API (Corrected 2-Class Logic)
# ============================================================================

class PatientData(BaseModel):
    age: float
    bmi: float
    bp_systolic: float
    bp_diastolic: float
    cholesterol: float
    glucose: float
    heart_rate: float
    smoking: int  
    diabetes: int
    family_history: int

@app.post("/api/doctor/predict")

async def predict_risk(patient: PatientData):
    """
    Simulate a doctor running inference on a local patient.
    """
    checkpoint_path = Path("checkpoints/global_model_latest.pth")
    
    if not checkpoint_path.exists():
        return {
            "risk_score": 0, "label": "NO MODEL", "color": "gray", "detail": "Run training first!"
        }

    try:
        # 1. Initialize model
        model = HealthRiskModel(input_dim=10, output_dim=2)
        
        # 2. Load weights
        checkpoint = torch.load(checkpoint_path)
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            model.load_state_dict(checkpoint)
        model.eval()

        # 3. 🛑 PREPROCESSING: Normalize the Input!
        # These values should match the "feature_means" and "feature_stds" 
        # that you saw printed when you ran `loader.py`.
        # I have populated these with approximate medical averages for a demo.
        stats = {
            "age":            {"mean": 50.0, "std": 15.0},
            "bmi":            {"mean": 28.0, "std": 6.0},
            "bp_systolic":    {"mean": 130.0, "std": 20.0},
            "bp_diastolic":   {"mean": 80.0,  "std": 12.0},
            "cholesterol":    {"mean": 200.0, "std": 40.0},
            "glucose":        {"mean": 100.0, "std": 30.0},
            "heart_rate":     {"mean": 75.0,  "std": 12.0},
            # Binary features also get normalized by StandardScaler
            "smoking":        {"mean": 0.4,   "std": 0.5}, 
            "diabetes":       {"mean": 0.3,   "std": 0.45},
            "family_history": {"mean": 0.2,   "std": 0.4}
        }

        def norm(value, key):
            return (value - stats[key]["mean"]) / stats[key]["std"]

        input_tensor = torch.tensor([[
            norm(patient.age, "age"),
            norm(patient.bmi, "bmi"),
            norm(patient.bp_systolic, "bp_systolic"),
            norm(patient.bp_diastolic, "bp_diastolic"),
            norm(patient.cholesterol, "cholesterol"),
            norm(patient.glucose, "glucose"),
            norm(patient.heart_rate, "heart_rate"),
            norm(float(patient.smoking), "smoking"),
            norm(float(patient.diabetes), "diabetes"),
            norm(float(patient.family_history), "family_history")
        ]], dtype=torch.float32)

        # 4. Predict
        with torch.no_grad():
            logits = model(input_tensor)
            probabilities = F.softmax(logits, dim=1)
            risk_probability = probabilities[0][1].item()
            
        risk_score = round(risk_probability * 100, 2)
        
        # 5. Determine Label
        if risk_probability > 0.65:
            label = "HIGH RISK"
            color = "#DC2626" # Red
        elif risk_probability > 0.35:
            label = "MODERATE RISK"
            color = "#D97706" # Amber
        else:
            label = "LOW RISK"
            color = "#16A34A" # Green
        
        return {
            "risk_score": risk_score,
            "label": label,
            "color": color,
            "detail": "Inference successful"
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        return {
            "risk_score": 0, "label": "ERROR", "color": "red", "detail": str(e)
        }
# ============================================================================
# API Endpoints
# ============================================================================

@app.get("/")
async def root():
    return {
        "name": "Federated Learning Admin API",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "dashboard": "/dashboard",
            "predict": "/api/doctor/predict"
        }
    }

@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

@app.get("/api/training/status", response_model=TrainingStatus)
async def get_training_status():
    return TrainingStatus(
        status=fl_state["status"],
        current_round=fl_state["current_round"],
        total_rounds=fl_state["total_rounds"],
        start_time=fl_state["start_time"],
        end_time=fl_state["end_time"],
        error=fl_state["error"]
    )

@app.post("/api/training/start")
async def start_training(request: StartTrainingRequest, background_tasks: BackgroundTasks):
    if fl_state["status"] == "running":
        raise HTTPException(status_code=400, detail="Training is already running")
    
    fl_state["status"] = "starting"
    fl_state["current_round"] = 0
    fl_state["total_rounds"] = request.num_rounds
    fl_state["error"] = None
    fl_state["end_time"] = None
    
    background_tasks.add_task(run_fl_training, request.num_rounds)
    
    return {"message": "Training started"}

@app.post("/api/training/stop")
async def stop_training():
    if fl_state["status"] != "running":
        raise HTTPException(status_code=400, detail="No training is currently running")
    fl_state["status"] = "stopping"
    return {"message": "Stop request received"}

@app.get("/api/metrics", response_model=MetricsResponse)
async def get_metrics():
    history = get_training_history()
    if not history:
        return MetricsResponse(rounds=[], train_loss=[], train_accuracy=[], eval_loss=[], eval_accuracy=[], eval_f1_score=[])
    return MetricsResponse(
        rounds=history.get("rounds", []),
        train_loss=history.get("train_loss", []),
        train_accuracy=history.get("train_accuracy", []),
        eval_loss=history.get("eval_loss", []),
        eval_accuracy=history.get("eval_accuracy", []),
        eval_f1_score=history.get("eval_f1_score", [])
    )

# 👇 FULLY RESTORED SUMMARY ENDPOINT
@app.get("/api/metrics/summary")
async def get_metrics_summary():
    """Get training metrics summary"""
    
    history = get_training_history()
    
    if not history:
        return {
            "total_rounds": 0,
            "final_train_accuracy": 0,
            "accuracy_improvement": 0,
            "final_eval_accuracy": 0,
            "eval_accuracy_improvement": 0,
            "final_f1_score": 0
        }
    
    rounds = history.get("rounds", [])
    train_acc = history.get("train_accuracy", [])
    eval_acc = history.get("eval_accuracy", [])
    
    summary = {
        "total_rounds": len(rounds),
        "initial_train_accuracy": train_acc[0] if train_acc else 0,
        "final_train_accuracy": train_acc[-1] if train_acc else 0,
        "accuracy_improvement": (train_acc[-1] - train_acc[0]) if train_acc else 0,
        "best_train_accuracy": max(train_acc) if train_acc else 0,
    }
    
    if eval_acc:
        summary["initial_eval_accuracy"] = eval_acc[0]
        summary["final_eval_accuracy"] = eval_acc[-1]
        summary["eval_accuracy_improvement"] = eval_acc[-1] - eval_acc[0]
        summary["best_eval_accuracy"] = max(eval_acc)
    else:
        summary["final_eval_accuracy"] = 0
        summary["eval_accuracy_improvement"] = 0
    
    if history.get("eval_f1_score"):
        f1_scores = history["eval_f1_score"]
        summary["final_f1_score"] = f1_scores[-1]
        summary["best_f1_score"] = max(f1_scores)
    else:
        summary["final_f1_score"] = 0
    
    return summary

@app.get("/api/checkpoints")
async def list_checkpoints():
    checkpoints = get_available_checkpoints()
    return {"count": len(checkpoints), "checkpoints": checkpoints}

@app.get("/api/checkpoints/{round_number}/download")
async def download_checkpoint(round_number: int):
    checkpoint_path = Path(f"checkpoints/global_model_round_{round_number}.pth")
    if not checkpoint_path.exists():
        raise HTTPException(status_code=404, detail="Checkpoint not found")
    return FileResponse(path=checkpoint_path, filename=checkpoint_path.name, media_type="application/octet-stream")

@app.get("/api/checkpoints/latest/download")
async def download_latest_checkpoint():
    checkpoint_path = Path("checkpoints/global_model_latest.pth")
    if not checkpoint_path.exists():
        raise HTTPException(status_code=404, detail="No checkpoint found")
    return FileResponse(path=checkpoint_path, filename="global_model_latest.pth", media_type="application/octet-stream")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=API_PORT, log_level="info")