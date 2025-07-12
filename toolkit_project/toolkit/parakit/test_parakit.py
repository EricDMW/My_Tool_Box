#!/usr/bin/env python3
"""
Test script for parakit module functionality.
"""

import argparse
import tempfile
import json
from pathlib import Path
from toolkit.parakit import ParameterTuner


def test_parameter_tuner_creation():
    """Test ParameterTuner creation and basic functionality."""
    print("Testing ParameterTuner creation...")
    
    # Create a simple parser
    parser = argparse.ArgumentParser(description="Test Parser")
    parser.add_argument("--test_param", type=float, default=1.0, help="Test parameter")
    
    # Create ParameterTuner instance
    tuner = ParameterTuner(parser)
    
    # Check basic attributes
    assert tuner.parser == parser
    assert tuner.save_delay == 30000
    assert tuner.auto_save_enabled == True
    assert len(tuner.args_defaults) == 1
    assert "test_param" in tuner.args_defaults
    
    print("✓ ParameterTuner creation test passed")


def test_parameter_validation():
    """Test parameter validation functionality."""
    print("Testing parameter validation...")
    
    parser = argparse.ArgumentParser(description="Validation Test")
    parser.add_argument("--int_param", type=int, default=10, help="Integer parameter")
    parser.add_argument("--float_param", type=float, default=1.5, help="Float parameter")
    parser.add_argument("--choice_param", choices=["a", "b", "c"], default="a", help="Choice parameter")
    
    tuner = ParameterTuner(parser)
    
    # Test valid parameters
    is_valid, value, error = tuner._validate_parameter("int_param", "42")
    assert is_valid == True
    assert value == 42
    assert error == ""
    
    is_valid, value, error = tuner._validate_parameter("float_param", "3.14")
    assert is_valid == True
    assert value == 3.14
    assert error == ""
    
    is_valid, value, error = tuner._validate_parameter("choice_param", "b")
    assert is_valid == True
    assert value == "b"
    assert error == ""
    
    # Test invalid parameters
    is_valid, value, error = tuner._validate_parameter("int_param", "not_a_number")
    assert is_valid == False
    assert value is None
    assert "invalid literal" in error
    
    is_valid, value, error = tuner._validate_parameter("choice_param", "d")
    assert is_valid == False
    assert value is None
    assert "must be one of" in error
    
    print("✓ Parameter validation test passed")


def test_custom_validation():
    """Test custom validation callbacks."""
    print("Testing custom validation...")
    
    def validate_positive(value):
        try:
            val = float(value)
            if val > 0:
                return True, val, ""
            else:
                return False, None, "Value must be positive"
        except ValueError:
            return False, None, "Value must be a number"
    
    parser = argparse.ArgumentParser(description="Custom Validation Test")
    parser.add_argument("--positive_param", type=float, default=1.0, help="Positive parameter")
    
    validation_callbacks = {"positive_param": validate_positive}
    tuner = ParameterTuner(parser, validation_callbacks=validation_callbacks)
    
    # Test valid custom validation
    is_valid, value, error = tuner._validate_parameter("positive_param", "5.0")
    assert is_valid == True
    assert value == 5.0
    assert error == ""
    
    # Test invalid custom validation
    is_valid, value, error = tuner._validate_parameter("positive_param", "-1.0")
    assert is_valid == False
    assert value is None
    assert "must be positive" in error
    
    is_valid, value, error = tuner._validate_parameter("positive_param", "not_a_number")
    assert is_valid == False
    assert value is None
    assert "must be a number" in error
    
    print("✓ Custom validation test passed")


def test_class_method():
    """Test the convenience class method."""
    print("Testing class method...")
    
    parser = argparse.ArgumentParser(description="Class Method Test")
    parser.add_argument("--test_param", type=str, default="default", help="Test parameter")
    
    # Test the class method by creating a tuner instance directly
    tuner = ParameterTuner(
        parser, 
        save_delay=60000,
        auto_save_enabled=False
    )
    
    assert isinstance(tuner, ParameterTuner)
    assert tuner.parser == parser
    assert tuner.save_delay == 60000
    assert tuner.auto_save_enabled == False
    
    print("✓ Class method test passed")


def test_backward_compatibility():
    """Test backward compatibility with old class name."""
    print("Testing backward compatibility...")
    
    from toolkit.parakit import ParameterAdjuster
    
    parser = argparse.ArgumentParser(description="Backward Compatibility Test")
    parser.add_argument("--test_param", type=int, default=42, help="Test parameter")
    
    # Should work the same as ParameterTuner
    tuner = ParameterAdjuster(parser)
    assert isinstance(tuner, ParameterTuner)
    assert tuner.parser == parser
    
    print("✓ Backward compatibility test passed")


def main():
    """Run all tests."""
    print("Parakit Module Tests")
    print("=" * 40)
    
    try:
        test_parameter_tuner_creation()
        test_parameter_validation()
        test_custom_validation()
        test_class_method()
        test_backward_compatibility()
        
        print("\n" + "=" * 40)
        print("All tests passed successfully!")
        
    except Exception as e:
        print(f"\nTest failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 