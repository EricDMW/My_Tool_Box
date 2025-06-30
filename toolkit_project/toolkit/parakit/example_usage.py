#!/usr/bin/env python3
"""
Comprehensive example usage of the ParameterTuner toolkit.

This example demonstrates all features including:
- Basic parameter tuning
- Custom validation
- Save path management
- Inactivity timeout configuration
- Auto-save functionality
- Integration patterns
"""

import argparse
import logging
from pathlib import Path

from toolkit.parakit import ParameterTuner

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def validate_learning_rate(value):
    """Custom validation for learning rate."""
    try:
        lr = float(value)
        if 0 < lr <= 1:
            return True, lr, ""
        else:
            return False, None, "Learning rate must be between 0 and 1"
    except ValueError:
        return False, None, "Learning rate must be a valid number"


def validate_batch_size(value):
    """Custom validation for batch size."""
    try:
        bs = int(value)
        if bs > 0 and bs % 2 == 0:
            return True, bs, ""
        else:
            return False, None, "Batch size must be a positive even number"
    except ValueError:
        return False, None, "Batch size must be a valid integer"


def validate_epochs(value):
    """Custom validation for epochs."""
    try:
        epochs = int(value)
        if 1 <= epochs <= 1000:
            return True, epochs, ""
        else:
            return False, None, "Epochs must be between 1 and 1000"
    except ValueError:
        return False, None, "Epochs must be a valid integer"


def example_1_basic_usage():
    """Example 1: Basic parameter tuning with default settings."""
    print("\n=== Example 1: Basic Usage ===")
    
    # Create argument parser
    parser = argparse.ArgumentParser(description="Basic ML Training")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--optimizer", choices=["adam", "sgd", "rmsprop"], default="adam", help="Optimizer")
    
    # Launch GUI with default settings
    # - Auto-save every 30 seconds
    # - Save to current working directory
    # - 10 seconds inactivity timeout
    updated_parser = ParameterTuner.tune_parameters(parser)
    
    # Parse arguments with updated defaults
    args = updated_parser.parse_args()
    print(f"Final parameters: lr={args.learning_rate}, batch_size={args.batch_size}, epochs={args.epochs}, optimizer={args.optimizer}")


def example_2_custom_validation():
    """Example 2: Custom validation with specific constraints."""
    print("\n=== Example 2: Custom Validation ===")
    
    # Create argument parser
    parser = argparse.ArgumentParser(description="Validated ML Training")
    parser.add_argument("--learning_rate", type=float, default=0.01, help="Learning rate (0-1)")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size (even number)")
    parser.add_argument("--epochs", type=int, default=200, help="Epochs (1-1000)")
    parser.add_argument("--dropout", type=float, default=0.5, help="Dropout rate")
    
    # Define validation callbacks
    validation_callbacks = {
        "learning_rate": validate_learning_rate,
        "batch_size": validate_batch_size,
        "epochs": validate_epochs
    }
    
    # Launch GUI with custom validation
    updated_parser = ParameterTuner.tune_parameters(
        parser=parser,
        validation_callbacks=validation_callbacks,
        save_delay=45000,  # 45 seconds auto-save
        inactivity_timeout=15  # 15 seconds inactivity timeout
    )
    
    args = updated_parser.parse_args()
    print(f"Validated parameters: lr={args.learning_rate}, batch_size={args.batch_size}, epochs={args.epochs}, dropout={args.dropout}")


def example_3_save_path_management():
    """Example 3: Custom save paths and directory management."""
    print("\n=== Example 3: Save Path Management ===")
    
    # Create argument parser
    parser = argparse.ArgumentParser(description="Save Path Example")
    parser.add_argument("--model_type", choices=["cnn", "rnn", "transformer"], default="cnn", help="Model architecture")
    parser.add_argument("--hidden_size", type=int, default=512, help="Hidden layer size")
    parser.add_argument("--num_layers", type=int, default=3, help="Number of layers")
    
    # Create custom save directory
    custom_save_path = Path("./experiments/parameters")
    custom_save_path.mkdir(parents=True, exist_ok=True)
    
    # Launch GUI with custom save path
    updated_parser = ParameterTuner.tune_parameters(
        parser=parser,
        save_path=str(custom_save_path),
        auto_save_enabled=True,
        inactivity_timeout=20  # 20 seconds inactivity timeout
    )
    
    args = updated_parser.parse_args()
    print(f"Model config: type={args.model_type}, hidden_size={args.hidden_size}, layers={args.num_layers}")
    print(f"Parameters saved to: {custom_save_path}")


def example_4_inactivity_timeout():
    """Example 4: Different inactivity timeout configurations."""
    print("\n=== Example 4: Inactivity Timeout Configurations ===")
    
    # Create argument parser
    parser = argparse.ArgumentParser(description="Inactivity Example")
    parser.add_argument("--param1", type=float, default=1.0, help="First parameter")
    parser.add_argument("--param2", type=int, default=10, help="Second parameter")
    parser.add_argument("--param3", type=str, default="test", help="Third parameter")
    
    # Example with very short inactivity timeout (for testing)
    print("Launching GUI with 5-second inactivity timeout...")
    updated_parser = ParameterTuner.tune_parameters(
        parser=parser,
        inactivity_timeout=5,  # Very short for demonstration
        auto_save_enabled=True,
        save_delay=10000  # 10 seconds auto-save
    )
    
    args = updated_parser.parse_args()
    print(f"Parameters after inactivity handling: {args.param1}, {args.param2}, {args.param3}")


def example_5_auto_save_disabled():
    """Example 5: Manual save only (auto-save disabled)."""
    print("\n=== Example 5: Manual Save Only ===")
    
    # Create argument parser
    parser = argparse.ArgumentParser(description="Manual Save Example")
    parser.add_argument("--precision", type=float, default=0.001, help="Precision threshold")
    parser.add_argument("--max_iter", type=int, default=1000, help="Maximum iterations")
    parser.add_argument("--tolerance", type=float, default=1e-6, help="Convergence tolerance")
    
    # Launch GUI with auto-save disabled
    updated_parser = ParameterTuner.tune_parameters(
        parser=parser,
        auto_save_enabled=False,  # Manual save only
        inactivity_timeout=30,  # 30 seconds inactivity timeout
        save_path="./manual_saves"
    )
    
    args = updated_parser.parse_args()
    print(f"Manual save parameters: precision={args.precision}, max_iter={args.max_iter}, tolerance={args.tolerance}")


def example_6_integration_pattern():
    """Example 6: Integration with training scripts."""
    print("\n=== Example 6: Integration Pattern ===")
    
    def train_model(lr, batch_size, epochs, model_type):
        """Simulate model training."""
        print(f"Training {model_type} model with lr={lr}, batch_size={batch_size}, epochs={epochs}")
        print("Training completed successfully!")
    
    # Create argument parser for training
    parser = argparse.ArgumentParser(description="Training Integration")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--model_type", choices=["cnn", "rnn", "transformer"], default="cnn", help="Model type")
    
    # Tune parameters before training
    print("Launching parameter tuner...")
    updated_parser = ParameterTuner.tune_parameters(
        parser=parser,
        save_path="./training_configs",
        inactivity_timeout=10
    )
    
    # Get final parameters
    args = updated_parser.parse_args()
    
    # Start training with tuned parameters
    train_model(args.learning_rate, args.batch_size, args.epochs, args.model_type)


def example_7_current_directory_default():
    """Example 7: Using current directory as default save path."""
    print("\n=== Example 7: Current Directory Default ===")
    
    # Create argument parser
    parser = argparse.ArgumentParser(description="Current Directory Example")
    parser.add_argument("--alpha", type=float, default=0.1, help="Alpha parameter")
    parser.add_argument("--beta", type=float, default=0.9, help="Beta parameter")
    parser.add_argument("--gamma", type=float, default=0.99, help="Gamma parameter")
    
    # Use default save path (current working directory)
    print(f"Current working directory: {Path.cwd()}")
    updated_parser = ParameterTuner.tune_parameters(
        parser=parser,
        # No save_path specified - uses current directory
        inactivity_timeout=10
    )
    
    args = updated_parser.parse_args()
    print(f"Parameters saved to current directory: alpha={args.alpha}, beta={args.beta}, gamma={args.gamma}")


def main():
    """Run all examples."""
    print("ParameterTuner Toolkit - Comprehensive Examples")
    print("=" * 50)
    
    try:
        # Run examples
        example_1_basic_usage()
        example_2_custom_validation()
        example_3_save_path_management()
        example_4_inactivity_timeout()
        example_5_auto_save_disabled()
        example_6_integration_pattern()
        example_7_current_directory_default()
        
        print("\n" + "=" * 50)
        print("All examples completed successfully!")
        
    except KeyboardInterrupt:
        print("\nExamples interrupted by user.")
    except Exception as e:
        print(f"\nError running examples: {e}")
        logger.exception("Example execution failed")


if __name__ == "__main__":
    main() 