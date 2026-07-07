import dataclasses
from dataclasses import dataclass, field, asdict, fields
from typing import Optional, Dict, Any
import yaml


@dataclass
class BaseConfig:
    """Base class for all config classes"""

    @classmethod
    def from_dict(cls, data: Dict[str, Any]):
        """Generic from_dict that works for any dataclass"""
        if data is None:
            return cls()

        field_types = {f.name: f.type for f in dataclasses.fields(cls)}
        kwargs = {}

        for field_name, field_type in field_types.items():
            if field_name not in data:
                continue

            value = data[field_name]

            if dataclasses.is_dataclass(field_type):
                kwargs[field_name] = field_type.from_dict(value)
            else:
                kwargs[field_name] = value

        return cls(**kwargs)

@dataclass
class ModelConfig(BaseConfig):
    """Configuration for model architecture"""
    name: str = "resnet20"  # "resnet20", "plain20", etc.
    num_classes: int = 10
    net_type: str = "pre"
    # Add model-specific params later


@dataclass
class OptimizerConfig(BaseConfig):
    """Configuration for optimizer"""
    name: str = "sgd"  # "sgd", "adam", "adamw"
    lr: float = 0.1
    momentum: float = 0.9
    weight_decay: float = 1e-4

@dataclass
class TrainingConfig(BaseConfig):
    """Main training configuration"""
    epochs: int = 15
    batch_size: int = 32
    enable_gradient_tracking: bool = False

@dataclass
class SchedulerConfig(BaseConfig):
    """Main scheduler configuration"""
    name: str = "step"
    step_size: int = 10
    gamma: float = 0.1

@dataclass
class DataConfig(BaseConfig):
    """Dataset configuration"""
    name: str = "cifar10"
    root: str = "./data"
    num_workers: int = 2
    augmentation: bool = True


@dataclass
class ExperimentConfig(BaseConfig):
    """Top-level experiment configuration"""
    name: str = "baseline_experiment"
    seed: int = 42
    device: Optional[str] = None  # None = auto-detect

    # Sub-configurations
    model: ModelConfig = field(default_factory=ModelConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)

    def save_to_file(self, filepath: str):
        """Save config to YAML file"""
        with open(filepath, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False)

    @classmethod
    def load_from_file(cls, filepath: str):
        """Load config from YAML file"""
        with open(filepath, 'r') as f:
            data = yaml.safe_load(f)
        return cls.from_dict(data)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return asdict(self)

    def validate(self):
        """Validate configuration settings"""
        errors = []

        if self.training.epochs <= 0:
            errors.append("Epochs must be positive")
        if self.training.batch_size <= 0:
            errors.append("Batch size must be positive")
        if self.optimizer.lr <= 0:
            errors.append("Learning rate must be positive")
        if self.model.net_type not in ["pre", "post", "nobn"]:
            errors.append(f"Invalid net_type: {self.model.net_type}")

        if errors:
            raise ValueError("Configuration validation failed:\n" + "\n".join(f"  - {e}" for e in errors))

        return True

    @classmethod
    def load_with_overrides(cls, base_path: str, override_path: str = None):
        """
        Load a base config and optionally apply overrides from another file.

        Args:
            base_path: Path to base config YAML
            override_path: Optional path to override config YAML

        Returns:
            ExperimentConfig with merged settings

        Example:
            config = ExperimentConfig.load_with_overrides(
                'configs/base.yaml',
                'configs/pre_activation.yaml'
            )
        """
        # Load base config
        with open(base_path, 'r') as f:
            base_data = yaml.safe_load(f)

        # Load and merge override if provided
        if override_path:
            with open(override_path, 'r') as f:
                override_data = yaml.safe_load(f)

            # Merge: override_data takes precedence
            merged_data = deep_update(base_data, override_data)
        else:
            merged_data = base_data

        # Create config from merged data
        config = cls.from_dict(merged_data)

        # Validate
        config.validate()

        return config


def deep_update(base_dict: dict, update_dict: dict) -> dict:
    """
    Recursively update base_dict with values from update_dict.

    This is like dict.update() but works on nested dictionaries.

    Args:
        base_dict: The base configuration
        update_dict: Values to override

    Returns:
        Merged dictionary
    """
    result = base_dict.copy()

    for key, value in update_dict.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            # Both are dicts - merge recursively
            result[key] = deep_update(result[key], value)
        else:
            # Override the value
            result[key] = value

    return result