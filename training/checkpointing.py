"""
Checkpoint management for model saving and loading
"""
import os
import json
from pathlib import Path
from typing import Dict, Optional, Any
from datetime import datetime
import shutil

import torch
import torch.nn as nn


class CheckpointManager:
    """Manages model checkpoints"""
    
    def __init__(
        self,
        checkpoint_dir: str = "checkpoints",
        keep_best: int = 3,
        keep_latest: int = 5,
    ):
        """
        Initialize checkpoint manager
        
        Args:
            checkpoint_dir: Directory to save checkpoints
            keep_best: Number of best checkpoints to keep
            keep_latest: Number of latest checkpoints to keep
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.keep_best = keep_best
        self.keep_latest = keep_latest
        
        self.best_checkpoints = []
        self.latest_checkpoints = []
        
        # Load existing checkpoint info
        self._load_checkpoint_info()
    
    def _load_checkpoint_info(self):
        """Load existing checkpoint info"""
        info_path = self.checkpoint_dir / "checkpoint_info.json"
        if info_path.exists():
            with open(info_path, "r") as f:
                info = json.load(f)
                self.best_checkpoints = info.get("best", [])
                self.latest_checkpoints = info.get("latest", [])
    
    def _save_checkpoint_info(self):
        """Save checkpoint info"""
        info = {
            "best": self.best_checkpoints,
            "latest": self.latest_checkpoints,
        }
        info_path = self.checkpoint_dir / "checkpoint_info.json"
        with open(info_path, "w") as f:
            json.dump(info, f, indent=2)
    
    def save(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        step: int,
        metrics: Dict[str, Any],
        is_best: bool = False,
    ) -> str:
        """
        Save checkpoint
        
        Args:
            model: Model to save
            optimizer: Optimizer to save
            step: Current training step
            metrics: Training metrics
            is_best: Whether this is the best model so far
            
        Returns:
            Path to saved checkpoint
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        checkpoint = {
            "step": step,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "metrics": metrics,
            "timestamp": timestamp,
        }
        
        # Save checkpoint
        if is_best:
            filename = f"best_step{step}_{timestamp}.pt"
            path = self.checkpoint_dir / filename
            torch.save(checkpoint, path)
            
            # Track best checkpoints
            self.best_checkpoints.append({
                "path": str(path),
                "step": step,
                "metrics": metrics,
            })
            
            # Prune old best checkpoints
            while len(self.best_checkpoints) > self.keep_best:
                old = self.best_checkpoints.pop(0)
                if os.path.exists(old["path"]):
                    os.remove(old["path"])
        else:
            filename = f"checkpoint_step{step}_{timestamp}.pt"
            path = self.checkpoint_dir / filename
            torch.save(checkpoint, path)
            
            # Track latest checkpoints
            self.latest_checkpoints.append({
                "path": str(path),
                "step": step,
            })
            
            # Prune old checkpoints
            while len(self.latest_checkpoints) > self.keep_latest:
                old = self.latest_checkpoints.pop(0)
                if os.path.exists(old["path"]):
                    os.remove(old["path"])
        
        self._save_checkpoint_info()
        print(f"Saved checkpoint: {filename}")
        
        return str(path)
    
    def load(
        self,
        path: str,
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        device: str = "cpu",
    ) -> Optional[Dict]:
        """
        Load checkpoint
        
        Args:
            path: Path to checkpoint
            model: Model to load state into
            optimizer: Optional optimizer to load state into
            device: Device to load to
            
        Returns:
            Checkpoint dict or None if failed
        """
        if not os.path.exists(path):
            print(f"Checkpoint not found: {path}")
            return None
        
        checkpoint = torch.load(path, map_location=device)
        
        model.load_state_dict(checkpoint["model_state_dict"])
        
        if optimizer is not None:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        
        print(f"Loaded checkpoint from step {checkpoint['step']}")
        return checkpoint
    
    def load_best(
        self,
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        device: str = "cpu",
    ) -> Optional[Dict]:
        """Load the best checkpoint"""
        if not self.best_checkpoints:
            print("No best checkpoint available")
            return None
        
        best = self.best_checkpoints[-1]
        return self.load(best["path"], model, optimizer, device)
    
    def load_latest(
        self,
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        device: str = "cpu",
    ) -> Optional[Dict]:
        """Load the latest checkpoint"""
        if not self.latest_checkpoints:
            print("No checkpoint available")
            return None
        
        latest = self.latest_checkpoints[-1]
        return self.load(latest["path"], model, optimizer, device)
    
    def get_best_path(self) -> Optional[str]:
        """Get path to best checkpoint"""
        if self.best_checkpoints:
            return self.best_checkpoints[-1]["path"]
        return None
    
    def get_latest_path(self) -> Optional[str]:
        """Get path to latest checkpoint"""
        if self.latest_checkpoints:
            return self.latest_checkpoints[-1]["path"]
        return None
    
    def list_checkpoints(self) -> Dict[str, list]:
        """List all tracked checkpoints"""
        return {
            "best": self.best_checkpoints,
            "latest": self.latest_checkpoints,
        }


if __name__ == "__main__":
    import tempfile
    
    # Test checkpoint manager
    with tempfile.TemporaryDirectory() as tmpdir:
        manager = CheckpointManager(checkpoint_dir=tmpdir, keep_best=2, keep_latest=3)
        
        # Create dummy model and optimizer
        model = torch.nn.Linear(10, 5)
        optimizer = torch.optim.Adam(model.parameters())
        
        # Save some checkpoints
        for i in range(5):
            metrics = {"loss": 1.0 - i * 0.1, "accuracy": 0.5 + i * 0.1}
            is_best = i >= 3  # Last 2 are best
            manager.save(model, optimizer, step=i * 1000, metrics=metrics, is_best=is_best)
        
        print("\nCheckpoints:")
        checkpoints = manager.list_checkpoints()
        print(f"Best: {len(checkpoints['best'])}")
        print(f"Latest: {len(checkpoints['latest'])}")
        
        # Load best
        print("\nLoading best checkpoint...")
        checkpoint = manager.load_best(model, optimizer)
        if checkpoint:
            print(f"  Step: {checkpoint['step']}")
            print(f"  Metrics: {checkpoint['metrics']}")

