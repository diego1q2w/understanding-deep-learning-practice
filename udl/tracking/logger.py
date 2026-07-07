"""Simple experiment logging for deep learning experiments."""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import yaml


class ExperimentLogger:
    """
    Lightweight experiment logger that saves metrics and configs to JSON/YAML files.

    Example:
        >>> logger = ExperimentLogger("resnet_experiment")
        >>> logger.log_config({"lr": 0.01, "batch_size": 64})
        >>> for epoch in range(10):
        ...     logger.log_metrics({"loss": 0.5, "accuracy": 0.92}, step=epoch)
        >>> logger.save_summary({"best_accuracy": 0.95})
    """

    def __init__(
        self,
        experiment_name: str,
        base_dir: str = "experiments",
        timestamp: bool = True,
    ):
        """
        Initialize experiment logger.

        Args:
            experiment_name: Name of the experiment (creates a directory)
            base_dir: Base directory for all experiments
            timestamp: If True, add timestamp to experiment directory
        """
        self.experiment_name = experiment_name
        self.base_dir = Path(base_dir)

        # Create experiment directory
        if timestamp:
            time_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            self.exp_dir = self.base_dir / experiment_name / time_str
        else:
            self.exp_dir = self.base_dir / experiment_name

        self.exp_dir.mkdir(parents=True, exist_ok=True)

        # File paths
        self.config_path = self.exp_dir / "config.yaml"
        self.metrics_path = self.exp_dir / "metrics.jsonl"
        self.summary_path = self.exp_dir / "summary.json"

        self._metrics_file = None

    @property
    def experiment_path(self) -> Path:
        """Return the experiment directory path."""
        return self.exp_dir

    def log_config(self, config: dict) -> None:
        """
        Save experiment configuration to YAML file.

        Args:
            config: Configuration dictionary or dataclass with to_dict() method
        """
        # Handle dataclass-like objects with to_dict method
        if hasattr(config, "to_dict"):
            config = config.to_dict()

        with open(self.config_path, "w") as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    def log_metrics(self, metrics: dict, step: Optional[int] = None) -> None:
        """
        Log metrics to JSONL file (one JSON object per line).

        Args:
            metrics: Dictionary of metric values
            step: Optional step/epoch number
        """
        record = metrics.copy()
        if step is not None:
            record["step"] = step
        record["timestamp"] = datetime.now().isoformat()

        with open(self.metrics_path, "a") as f:
            f.write(json.dumps(record) + "\n")

    def log_artifact(self, name: str, obj: Any, artifact_type: str = "auto") -> Path:
        """
        Save an artifact (model weights, etc).

        Args:
            name: Artifact filename (e.g., "model_best.pt")
            obj: Object to save
            artifact_type: "torch", "pickle", "json", or "auto" (detect from extension)

        Returns:
            Path to saved artifact
        """
        import pickle

        import torch

        artifact_path = self.exp_dir / name

        if artifact_type == "auto":
            ext = artifact_path.suffix.lower()
            if ext in [".pt", ".pth"]:
                artifact_type = "torch"
            elif ext == ".json":
                artifact_type = "json"
            else:
                artifact_type = "pickle"

        if artifact_type == "torch":
            torch.save(obj, artifact_path)
        elif artifact_type == "json":
            with open(artifact_path, "w") as f:
                json.dump(obj, f, indent=2)
        else:
            with open(artifact_path, "wb") as f:
                pickle.dump(obj, f)

        return artifact_path

    def save_summary(self, summary: dict) -> None:
        """
        Save final experiment summary.

        Args:
            summary: Dictionary of final results
        """
        with open(self.summary_path, "w") as f:
            json.dump(summary, f, indent=2)

    def load_metrics(self) -> list[dict]:
        """Load all logged metrics as a list of dictionaries."""
        if not self.metrics_path.exists():
            return []

        metrics = []
        with open(self.metrics_path, "r") as f:
            for line in f:
                if line.strip():
                    metrics.append(json.loads(line))
        return metrics

    @staticmethod
    def load_experiment(exp_path: str) -> dict:
        """
        Load all data from a saved experiment.

        Args:
            exp_path: Path to experiment directory

        Returns:
            Dictionary with config, metrics, and summary
        """
        exp_dir = Path(exp_path)

        result = {}

        config_path = exp_dir / "config.yaml"
        if config_path.exists():
            with open(config_path, "r") as f:
                result["config"] = yaml.safe_load(f)

        metrics_path = exp_dir / "metrics.jsonl"
        if metrics_path.exists():
            metrics = []
            with open(metrics_path, "r") as f:
                for line in f:
                    if line.strip():
                        metrics.append(json.loads(line))
            result["metrics"] = metrics

        summary_path = exp_dir / "summary.json"
        if summary_path.exists():
            with open(summary_path, "r") as f:
                result["summary"] = json.load(f)

        return result

    @staticmethod
    def list_experiments(base_dir: str = "experiments") -> list[str]:
        """
        List all experiment directories.

        Args:
            base_dir: Base experiments directory

        Returns:
            List of experiment paths
        """
        base = Path(base_dir)
        if not base.exists():
            return []

        experiments = []
        for exp_name in sorted(base.iterdir()):
            if exp_name.is_dir():
                for run_dir in sorted(exp_name.iterdir()):
                    if run_dir.is_dir():
                        experiments.append(str(run_dir))
        return experiments

    def __repr__(self) -> str:
        return f"ExperimentLogger('{self.experiment_name}', path='{self.exp_dir}')"
