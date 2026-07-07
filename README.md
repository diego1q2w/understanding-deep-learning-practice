# Understanding Deep Learning - Practice Repository

Hands-on practice exercises for the book **"Understanding Deep Learning"** by Simon Prince.

## Project Structure

| Module | Chapter | Topics |
|--------|---------|--------|
| `ch5_loss_functions/` | Ch 5: Loss Functions | Cross-entropy, MSE, class imbalance, focal loss |
| `ch6_optimization/` | Ch 6: Fitting Models | SGD, AdamW, learning rate scheduling |
| `ch7_init_activation_norm/` | Ch 7: Gradients & Initialization | Xavier/He init, batch normalization |
| `ch10_cnn/` | Ch 10: Convolutional Networks | CNN from scratch (numpy + PyTorch) |
| `ch11_residual_networks/` | Ch 11: Residual Networks | ResNet-20, skip connections |

## Quick Start

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # macOS/Linux
# venv\Scripts\activate    # Windows

# Install dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Verify setup
python -c "from udl import get_device; print(get_device())"
```

## Running Experiments

Each chapter module contains experiments you can run directly:

```bash
# Example: Run ResNet training
cd ch11_residual_networks
python trainer.py

# Example: Run loss function comparison
cd ch5_loss_functions/losses
python experiment.py
```

## Shared Utilities

The `udl/` package provides shared utilities across all experiments:

```python
from udl import get_device, set_seed
from udl.tracking import ExperimentLogger

# Device detection (CUDA/MPS/CPU)
device = get_device()

# Reproducibility
set_seed(42)

# Experiment logging
logger = ExperimentLogger("my_experiment")
logger.log_config({"lr": 0.01, "epochs": 100})
logger.log_metrics({"loss": 0.5, "accuracy": 0.92}, step=1)
```

## Testing

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_models/test_resnet.py

# Run with verbose output
pytest -v
```

## Book Reference

Prince, Simon J.D. "Understanding Deep Learning." MIT Press, 2023.

Available at: https://udlbook.github.io/udlbook/
