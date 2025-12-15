"""
Additional admin routes for advanced features
"""

from fastapi import APIRouter, HTTPException
from pathlib import Path
import json
from datetime import datetime

router = APIRouter(prefix="/api/admin", tags=["admin"])


@router.get("/logs")
async def get_logs():
    """Get recent coordinator logs"""
    
    log_path = Path("monitoring/logs/coordinator.log")
    
    if not log_path.exists():
        return {"logs": [], "message": "No logs available"}
    
    # Read last 100 lines
    with open(log_path) as f:
        lines = f.readlines()
        recent_logs = lines[-100:]
    
    return {
        "total_lines": len(lines),
        "recent_logs": recent_logs
    }


@router.delete("/checkpoints/{round_number}")
async def delete_checkpoint(round_number: int):
    """Delete a specific checkpoint"""
    
    checkpoint_path = Path(f"checkpoints/global_model_round_{round_number}.pth")
    
    if not checkpoint_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Checkpoint for round {round_number} not found"
        )
    
    checkpoint_path.unlink()
    
    return {
        "message": f"Checkpoint for round {round_number} deleted",
        "deleted_file": str(checkpoint_path)
    }


@router.post("/cleanup")
async def cleanup_old_checkpoints(keep_latest: int = 5):
    """Clean up old checkpoints, keeping only the latest N"""
    
    checkpoints_dir = Path("checkpoints")
    if not checkpoints_dir.exists():
        return {"message": "No checkpoints directory found"}
    
    # Get all checkpoint files
    checkpoints = sorted(
        checkpoints_dir.glob("global_model_round_*.pth"),
        key=lambda x: int(x.stem.split("_")[-1])
    )
    
    # Keep only latest N
    to_delete = checkpoints[:-keep_latest] if len(checkpoints) > keep_latest else []
    
    deleted = []
    for cp in to_delete:
        cp.unlink()
        deleted.append(cp.name)
    
    return {
        "deleted_count": len(deleted),
        "deleted_files": deleted,
        "kept_count": len(checkpoints) - len(deleted)
    }


@router.get("/statistics")
async def get_statistics():
    """Get detailed training statistics"""
    
    history_path = Path("checkpoints/training_history.json")
    
    if not history_path.exists():
        raise HTTPException(
            status_code=404,
            detail="No training history found"
        )
    
    with open(history_path) as f:
        history = json.load(f)
    
    stats = {
        "total_rounds": len(history["rounds"]),
        "per_round": []
    }
    
    for i, round_num in enumerate(history["rounds"]):
        round_stats = {
            "round": round_num,
            "train_loss": history["train_loss"][i],
            "train_accuracy": history["train_accuracy"][i],
        }
        
        if i < len(history["eval_loss"]):
            round_stats["eval_loss"] = history["eval_loss"][i]
            round_stats["eval_accuracy"] = history["eval_accuracy"][i]
            round_stats["f1_score"] = history["eval_f1_score"][i]
        
        stats["per_round"].append(round_stats)
    
    return stats