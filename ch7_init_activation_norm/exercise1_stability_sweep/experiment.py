import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import pickle
from pathlib import Path
from model import DeepMLP, compute_activation_stats, compute_gradient_stats


def get_mnist_sample(batch_size=128):
    """Load a single batch of MNIST for testing."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    dataset = datasets.MNIST(
        root='data',
        train=True,
        download=True,
        transform=transform
    )

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    x, y = next(iter(loader))
    return x, y


def run_stability_analysis(activation, init_method, device='cpu'):
    """
    Run stability analysis for a given activation + init combination.

    Returns:
        dict with activation and gradient statistics at init and after few steps
    """
    print(f"\n{'=' * 70}")
    print(f"Testing: {activation.upper()} + {init_method.upper()} initialization")
    print(f"{'=' * 70}")

    # Create model
    model = DeepMLP(
        input_dim=784,
        hidden_dim=256,
        output_dim=10,
        num_layers=20,
        activation=activation,
        init_method=init_method
    ).to(device)

    # Get sample data
    x, y = get_mnist_sample(batch_size=128)

    # === AT INITIALIZATION ===
    print("\n📊 At Initialization:")
    act_stats_init = compute_activation_stats(model, x, device)
    grad_stats_init = compute_gradient_stats(model, x, y, device)

    # Print summary
    print(f"\nActivation Variance (first 5 layers):")
    for stat in act_stats_init[:5]:
        print(f"  Layer {stat['layer']:2d}: Var={stat['var']:.6f}, Mean={stat['mean']:.6f}")

    print(f"\nActivation Variance (last 5 layers):")
    for stat in act_stats_init[-5:]:
        print(f"  Layer {stat['layer']:2d}: Var={stat['var']:.6f}, Mean={stat['mean']:.6f}")

    print(f"\nGradient Norms (first 5 layers):")
    for stat in grad_stats_init[:5]:
        print(f"  Layer {stat['layer']:2d}: Norm={stat['grad_norm']:.6f}")

    print(f"\nGradient Norms (last 5 layers):")
    for stat in grad_stats_init[-5:]:
        print(f"  Layer {stat['layer']:2d}: Norm={stat['grad_norm']:.6f}")

    # === AFTER A FEW TRAINING STEPS ===
    print("\n📊 After 10 Training Steps:")

    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()

    # Train for 10 steps
    model.train()
    for step in range(10):
        x_batch, y_batch = get_mnist_sample(batch_size=128)
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)

        optimizer.zero_grad()
        output = model(x_batch)
        loss = criterion(output, y_batch)
        loss.backward()
        optimizer.step()

    # Measure stats again
    act_stats_trained = compute_activation_stats(model, x, device)
    grad_stats_trained = compute_gradient_stats(model, x, y, device)

    print(f"\nActivation Variance (first 5 layers):")
    for stat in act_stats_trained[:5]:
        print(f"  Layer {stat['layer']:2d}: Var={stat['var']:.6f}, Mean={stat['mean']:.6f}")

    print(f"\nActivation Variance (last 5 layers):")
    for stat in act_stats_trained[-5:]:
        print(f"  Layer {stat['layer']:2d}: Var={stat['var']:.6f}, Mean={stat['mean']:.6f}")

    print(f"\nGradient Norms (first 5 layers):")
    for stat in grad_stats_trained[:5]:
        print(f"  Layer {stat['layer']:2d}: Norm={stat['grad_norm']:.6f}")

    print(f"\nGradient Norms (last 5 layers):")
    for stat in grad_stats_trained[-5:]:
        print(f"  Layer {stat['layer']:2d}: Norm={stat['grad_norm']:.6f}")

    # === DIAGNOSIS ===
    print("\n🔍 Diagnosis:")

    # Check activation variance stability
    var_first = act_stats_trained[0]['var']
    var_last = act_stats_trained[-1]['var']
    var_ratio = var_last / (var_first + 1e-8)

    if var_ratio < 0.01:
        print(f"  ⚠️  VANISHING ACTIVATIONS: Variance dropped {1 / var_ratio:.1f}× from layer 1 to 20")
    elif var_ratio > 100:
        print(f"  ⚠️  EXPLODING ACTIVATIONS: Variance grew {var_ratio:.1f}× from layer 1 to 20")
    else:
        print(f"  ✅ STABLE ACTIVATIONS: Variance ratio = {var_ratio:.2f}")

    # Check gradient norm stability
    grad_first = grad_stats_trained[0]['grad_norm']
    grad_last = grad_stats_trained[-1]['grad_norm']
    grad_ratio = grad_first / (grad_last + 1e-8)

    if grad_ratio > 100:
        print(f"  ⚠️  VANISHING GRADIENTS: Early layer gradients {grad_ratio:.1f}× smaller than late layers")
    elif grad_ratio < 0.01:
        print(f"  ⚠️  EXPLODING GRADIENTS: Early layer gradients {1 / grad_ratio:.1f}× larger than late layers")
    else:
        print(f"  ✅ STABLE GRADIENTS: Gradient ratio = {grad_ratio:.2f}")

    return {
        'activation': activation,
        'init_method': init_method,
        'act_stats_init': act_stats_init,
        'grad_stats_init': grad_stats_init,
        'act_stats_trained': act_stats_trained,
        'grad_stats_trained': grad_stats_trained,
        'var_ratio': var_ratio,
        'grad_ratio': grad_ratio
    }


if __name__ == '__main__':
    # Setup
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {device}")

    results_dir = Path('results')
    results_dir.mkdir(exist_ok=True)

    # Test combinations
    activations = ['relu', 'tanh', 'gelu']
    init_methods = ['xavier', 'he']

    all_results = []

    for activation in activations:
        for init_method in init_methods:
            result = run_stability_analysis(activation, init_method, device)
            all_results.append(result)

    # Save results
    output_file = results_dir / 'stability_analysis.pkl'
    with open(output_file, 'wb') as f:
        pickle.dump(all_results, f)

    print(f"\n{'=' * 70}")
    print(f"✅ Results saved to {output_file}")
    print(f"{'=' * 70}")

    # Print summary table
    print("\n📊 SUMMARY TABLE:")
    print(f"{'Activation':<10} {'Init':<8} {'Var Ratio':<12} {'Grad Ratio':<12} {'Status'}")
    print("-" * 70)

    for result in all_results:
        status = "✅ Good" if (0.1 < result['var_ratio'] < 10 and 0.1 < result['grad_ratio'] < 10) else "❌ Unstable"
        print(f"{result['activation']:<10} {result['init_method']:<8} "
              f"{result['var_ratio']:<12.4f} {result['grad_ratio']:<12.4f} {status}")