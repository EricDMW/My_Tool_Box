# Kuramoto Environment Setup Guide

This guide explains how to install and use the Kuramoto Oscillator Synchronization Environment as a standard Gymnasium environment.

## Quick Installation

### Option 1: Development Installation (Recommended)
```bash
# Navigate to the kos_env directory
cd env_lib/kos_env

# Install in development mode
pip install -e .
```

### Option 2: Using the Installation Script
```bash
# Navigate to the kos_env directory
cd env_lib/kos_env

# Run the installation script
./install.sh
```

### Option 3: Manual Installation
```bash
# Navigate to the kos_env directory
cd env_lib/kos_env

# Install dependencies first
pip install -r requirements.txt

# Install the package
pip install -e .
```

## Verification

After installation, you can verify that everything works correctly:

```bash
# Run the test script
python test_setup.py
```

This will test:
- Environment registration
- Environment creation
- Basic interaction

## Usage

Once installed, you can use the environments in your Python code:

```python
import gymnasium as gym

# Create a basic Kuramoto environment
env = gym.make('KuramotoOscillator-v0')

# Create a PyTorch-based environment
env = gym.make('KuramotoOscillatorTorch-v0')

# Create a multi-agent environment
env = gym.make('KuramotoOscillatorTorch-v1')  # 4 agents, 8 oscillators

# Create a GPU-optimized environment
env = gym.make('KuramotoOscillatorTorch-v2')  # GPU, 10 oscillators
```

## Available Environments

### NumPy-based Environments
- `KuramotoOscillator-v0`: Standard environment with 10 oscillators
- `KuramotoOscillator-v1`: Simplified environment with 5 oscillators
- `KuramotoOscillator-Constant-v0`: Environment with constant coupling matrix
- `KuramotoOscillator-FreqSync-Constant-v0`: Frequency synchronization with constant coupling

### PyTorch-based Environments
- `KuramotoOscillatorTorch-v0`: Standard PyTorch environment
- `KuramotoOscillatorTorch-v1`: Multi-agent environment (4 agents, 8 oscillators)
- `KuramotoOscillatorTorch-v2`: GPU-optimized environment
- `KuramotoOscillatorTorch-Constant-v0`: PyTorch version with constant coupling
- `KuramotoOscillatorTorch-FreqSync-Constant-v0`: PyTorch frequency synchronization

## Package Structure

```
kos_env/
├── setup.py              # Package setup configuration
├── MANIFEST.in           # Files to include in distribution
├── requirements.txt      # Python dependencies
├── install.sh           # Installation script
├── test_setup.py        # Test script
├── __init__.py          # Environment registration
├── kuramoto_env.py      # NumPy-based environment
├── kuramoto_env_torch.py # PyTorch-based environment
├── example_usage.py     # Usage examples
├── README.md            # Main documentation
└── README_KURAMOTO_IMPLEMENTATION.md # Implementation details
```

## Development

### Installing in Development Mode
```bash
pip install -e .
```

This installs the package in "editable" mode, meaning changes to the source code will be immediately available without reinstalling.

### Running Tests
```bash
python test_setup.py
```

### Uninstalling
```bash
pip uninstall kos-env
```

## Troubleshooting

### Common Issues

1. **Import Error**: Make sure you've installed the package with `pip install -e .`

2. **Environment Not Found**: Verify that the environments are registered by running `python test_setup.py`

3. **Missing Dependencies**: Install all requirements with `pip install -r requirements.txt`

4. **Permission Error**: Make sure the installation script is executable: `chmod +x install.sh`

### Getting Help

If you encounter issues:
1. Check that all dependencies are installed
2. Run the test script to verify installation
3. Check the main README.md for detailed usage examples
4. Ensure you're using Python 3.8 or higher

## Contributing

To contribute to this package:
1. Make your changes to the source code
2. Test with `python test_setup.py`
3. Update documentation if needed
4. Submit a pull request

## License

This package is distributed under the MIT License. See the LICENSE file for details. 