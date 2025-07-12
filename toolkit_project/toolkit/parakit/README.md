# Parameter Toolkit (parakit)

A comprehensive parameter tuning toolkit for argparse-based applications with an intuitive GUI interface.

## 🎯 Features

- **🎨 Interactive GUI**: User-friendly interface with Times New Roman fonts
- **⚡ Auto-save**: Automatic parameter saving with configurable intervals
- **⏰ Inactivity Auto-close**: Auto-save and close after 10 seconds of inactivity
- **💾 Flexible Save Paths**: Configurable save locations with current directory as default
- **✅ Parameter Validation**: Built-in and custom validation support
- **🔄 Real-time Updates**: Live status updates and error handling
- **🎮 Multiple Control Options**: Save button, OK button (save & close), and auto-save toggle

## 🚀 Quick Start

### Basic Usage

```python
import argparse
from toolkit.parakit import ParameterTuner

# Create argument parser
parser = argparse.ArgumentParser()
parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')

# Launch GUI
updated_parser = ParameterTuner.tune_parameters(parser)
```

### Advanced Configuration

```python
# Custom configuration
tuner = ParameterTuner(
    parser=parser,
    save_delay=60000,  # 60 seconds auto-save
    save_path="./my_parameters",  # Custom save directory
    auto_save_enabled=True,
    inactivity_timeout=15,  # 15 seconds inactivity timeout
    validation_callbacks={
        'learning_rate': lambda x: 0 < x < 1,
        'batch_size': lambda x: x > 0 and x % 2 == 0
    }
)

updated_parser = tuner.tune()
```

## ⚙️ Configuration Options

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `parser` | `argparse.ArgumentParser` | Required | The argument parser to tune |
| `save_delay` | `int` | 30000 | Auto-save delay in milliseconds |
| `save_path` | `str` | Current directory | Directory to save parameters |
| `auto_save_enabled` | `bool` | True | Whether to enable auto-save |
| `inactivity_timeout` | `int` | 10 | Seconds of inactivity before auto-close |
| `validation_callbacks` | `Dict[str, Callable]` | None | Custom validation functions |

## 🎮 GUI Controls

### Buttons
- **Save Parameters**: Save current parameters without closing
- **OK**: Save parameters and close the window
- **Browse**: Select custom save directory
- **Reset to Default**: Reset save path to current directory

### Features
- **Auto-save Toggle**: Enable/disable automatic saving
- **Activity Tracking**: Any user interaction resets the inactivity timer
- **Status Display**: Real-time status updates
- **Parameter Validation**: Immediate feedback on invalid inputs

## ⏰ Inactivity Management

The GUI automatically handles user inactivity:

1. **Activity Detection**: Tracks keyboard, mouse, and focus events
2. **Timer Reset**: Any interaction resets the 10-second countdown
3. **Auto-save**: Saves current parameters when timeout occurs
4. **Auto-close**: Closes the window after successful save

### Activity Events
- Keyboard input in any field
- Mouse clicks on buttons or fields
- Mouse movement over the window
- Field focus changes

## 💾 Save Path Management

### Default Behavior
- **Default Path**: Current working directory (`Path.cwd()`)
- **Auto-creation**: Save directory is created if it doesn't exist
- **File Naming**: `parameters_YYYYMMDD_HHMMSS.json` format

### Custom Paths
```python
# Set custom save path
tuner = ParameterTuner(
    parser=parser,
    save_path="/path/to/my/parameters"
)

# Use relative path
tuner = ParameterTuner(
    parser=parser,
    save_path="./experiments/params"
)
```

## ✅ Parameter Validation

### Built-in Validation
- **Type Checking**: Automatic conversion and validation
- **Required Fields**: Ensures all parameters have values
- **Error Display**: Clear error messages for invalid inputs

### Custom Validation
```python
def validate_learning_rate(value):
    return 0 < value < 1

def validate_batch_size(value):
    return value > 0 and value % 2 == 0

validation_callbacks = {
    'learning_rate': validate_learning_rate,
    'batch_size': validate_batch_size
}

tuner = ParameterTuner(
    parser=parser,
    validation_callbacks=validation_callbacks
)
```

## 📁 File Format

Parameters are saved in JSON format:

```json
{
    "learning_rate": 0.001,
    "batch_size": 32,
    "epochs": 100,
    "optimizer": "adam",
    "dropout": 0.5
}
```

## 🔧 Advanced Usage

### Custom Save Delays
```python
# Very frequent auto-save (every 5 seconds)
tuner = ParameterTuner(parser, save_delay=5000)

# Infrequent auto-save (every 2 minutes)
tuner = ParameterTuner(parser, save_delay=120000)
```

### Disable Auto-save
```python
# Manual save only
tuner = ParameterTuner(parser, auto_save_enabled=False)
```

### Custom Inactivity Timeout
```python
# Longer inactivity period (30 seconds)
tuner = ParameterTuner(parser, inactivity_timeout=30)

# Shorter inactivity period (5 seconds)
tuner = ParameterTuner(parser, inactivity_timeout=5)
```

### Integration with Training Scripts
```python
import argparse
from toolkit.parakit import ParameterTuner

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--batch_size', type=int, default=32)
    
    # Tune parameters before training
    parser = ParameterTuner.tune_parameters(parser)
    
    # Get final parameters
    args = parser.parse_args()
    
    # Start training with tuned parameters
    train_model(args.lr, args.batch_size)

if __name__ == "__main__":
    main()
```

## 🎨 GUI Appearance

The interface uses Times New Roman fonts for a professional appearance:

- **Parameter Labels**: Times New Roman 11pt bold
- **Help Text**: Times New Roman 9pt regular
- **Buttons**: Times New Roman 9-10pt bold
- **Status Text**: Times New Roman 8pt regular
- **Type Indicators**: Times New Roman 8pt blue text

## 🔍 Troubleshooting

### Common Issues

1. **Save Permission Errors**:
   ```python
   # Ensure save directory is writable
   tuner = ParameterTuner(parser, save_path="./writable/path")
   ```

2. **Validation Errors**:
   ```python
   # Check parameter types and constraints
   parser.add_argument('--lr', type=float, default=0.001)
   ```

3. **GUI Not Responding**:
   - Check for validation errors in parameter fields
   - Ensure all required parameters have valid values

### Debug Mode
```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Enable debug logging for troubleshooting
tuner = ParameterTuner(parser)
```

## 📚 Examples

See `example_usage.py` for comprehensive usage examples including:
- Basic parameter tuning
- Custom validation
- Save path management
- Inactivity timeout configuration
- Integration patterns

## 🤝 Contributing

1. Follow PEP 8 style guidelines
2. Add tests for new features
3. Update documentation for changes
4. Ensure backward compatibility

## 📄 License

This toolkit is part of the DSDP project and follows the same license terms. 