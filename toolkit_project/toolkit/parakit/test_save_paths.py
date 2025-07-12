#!/usr/bin/env python3
"""
Test script for save path functionality in parakit module.
"""

import argparse
import tempfile
import json
from pathlib import Path
from toolkit.parakit import ParameterTuner


def test_default_save_path():
    """Test default save path functionality."""
    print("Testing default save path...")
    
    # Test class method
    default_path = ParameterTuner.get_default_save_path()
    expected_path = Path.home() / ".toolkit" / "parameters"
    
    assert default_path == expected_path
    print(f"✓ Default save path: {default_path}")
    
    # Test that default path is created when needed
    tuner = ParameterTuner(argparse.ArgumentParser())
    assert tuner.save_path == expected_path
    assert tuner.save_path.exists()
    print("✓ Default path creation works")


def test_custom_save_path():
    """Test custom save path functionality."""
    print("Testing custom save path...")
    
    # Create temporary directory
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Test with custom path
        parser = argparse.ArgumentParser()
        parser.add_argument("--test_param", type=float, default=1.0)
        
        tuner = ParameterTuner(parser, save_path=str(temp_path))
        assert tuner.save_path == temp_path
        assert tuner.save_path.exists()
        
        print(f"✓ Custom save path: {temp_path}")


def test_save_path_creation():
    """Test that save paths are created automatically."""
    print("Testing save path creation...")
    
    # Test with non-existent path
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        nested_path = temp_path / "nested" / "deep" / "path"
        
        parser = argparse.ArgumentParser()
        parser.add_argument("--test_param", type=float, default=1.0)
        
        tuner = ParameterTuner(parser, save_path=str(nested_path))
        assert tuner.save_path == nested_path
        assert tuner.save_path.exists()
        
        print(f"✓ Nested path creation: {nested_path}")


def test_save_path_validation():
    """Test save path validation and error handling."""
    print("Testing save path validation...")
    
    # Test with invalid path (should still work due to automatic creation)
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_param", type=float, default=1.0)
    
    # This should work even with a potentially problematic path
    tuner = ParameterTuner(parser, save_path="./test_save_path")
    assert tuner.save_path.exists()
    
    print("✓ Save path validation works")


def test_save_path_display():
    """Test that save path is properly displayed in GUI."""
    print("Testing save path display...")
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_param", type=float, default=1.0)
    
    # Test with custom path
    custom_path = Path("./test_display_path")
    tuner = ParameterTuner(parser, save_path=str(custom_path))
    
    # Check that save_path_var would be set correctly
    # (This is tested in the GUI, but we can verify the path is correct)
    assert tuner.save_path == custom_path
    
    print("✓ Save path display setup works")


def test_reset_functionality():
    """Test reset to default functionality."""
    print("Testing reset functionality...")
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_param", type=float, default=1.0)
    
    # Start with custom path
    custom_path = Path("./test_reset_path")
    tuner = ParameterTuner(parser, save_path=str(custom_path))
    assert tuner.save_path == custom_path
    
    # Test reset method
    tuner._reset_save_path()
    default_path = ParameterTuner.get_default_save_path()
    assert tuner.save_path == default_path
    
    print("✓ Reset to default functionality works")


def main():
    """Run all save path tests."""
    print("Save Path Functionality Tests")
    print("=" * 40)
    
    try:
        test_default_save_path()
        test_custom_save_path()
        test_save_path_creation()
        test_save_path_validation()
        test_save_path_display()
        test_reset_functionality()
        
        print("\n" + "=" * 40)
        print("All save path tests passed successfully!")
        
    except Exception as e:
        print(f"\nTest failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 