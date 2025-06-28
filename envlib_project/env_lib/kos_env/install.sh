#!/bin/bash

# Install the kos-env package in development mode
echo "Installing kos-env package in development mode..."
pip install -e .

echo "Installation complete!"
echo ""
echo "You can now use the Kuramoto environments:"
echo "  - KuramotoOscillator-v0"
echo "  - KuramotoOscillator-v1"
echo "  - KuramotoOscillator-Constant-v0"
echo "  - KuramotoOscillator-FreqSync-Constant-v0"
echo "  - KuramotoOscillatorTorch-v0"
echo "  - KuramotoOscillatorTorch-v1"
echo "  - KuramotoOscillatorTorch-v2"
echo "  - KuramotoOscillatorTorch-Constant-v0"
echo "  - KuramotoOscillatorTorch-FreqSync-Constant-v0"
echo ""
echo "Example usage:"
echo "  import gymnasium as gym"
echo "  env = gym.make('KuramotoOscillator-v0')" 