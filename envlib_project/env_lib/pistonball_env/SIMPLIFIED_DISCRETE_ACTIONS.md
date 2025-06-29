# Simplified Discrete Action Space

## Overview

The discrete action space in the pistonball environment has been simplified to make it more intuitive and easier to use. Instead of using complex base-3 decoding, the discrete action space now directly uses a sequence of discrete actions for each piston.

## Changes Made

### Before (Complex Base-3 Decoding)
```python
# Action space
Discrete(3^n_pistons)

# Action format: Single integer
action = 13  # For 3 pistons

# Complex decoding in step() method
action_array = np.zeros(self.n_pistons)
for i in range(self.n_pistons):
    piston_action = (action // (3 ** i)) % 3
    action_array[i] = piston_action - 1  # Convert 0,1,2 to -1,0,1
```

### After (Simplified Sequence)
```python
# Action space
Box(0, 2, shape=(n_pistons,), dtype=np.int32)

# Action format: Direct sequence
action = np.array([0, 1, 2])  # For 3 pistons: down, stay, up

# Simple conversion in step() method
action_array = np.zeros(self.n_pistons)
for i in range(self.n_pistons):
    action_array[i] = action[i] - 1  # Convert 0,1,2 to -1,0,1
```

## Action Values

| Value | Meaning | Continuous Equivalent |
|-------|---------|----------------------|
| 0 | Move piston down | -1 |
| 1 | Keep piston stationary | 0 |
| 2 | Move piston up | 1 |

## Examples

### Basic Usage
```python
from pistonball_env import PistonballEnv
import numpy as np

# Create discrete environment
env = PistonballEnv(n_pistons=5, continuous=False)
obs, info = env.reset()

# All pistons down
action = np.array([0, 0, 0, 0, 0])
obs, reward, terminated, truncated, info = env.step(action)

# All pistons stay
action = np.array([1, 1, 1, 1, 1])
obs, reward, terminated, truncated, info = env.step(action)

# All pistons up
action = np.array([2, 2, 2, 2, 2])
obs, reward, terminated, truncated, info = env.step(action)

# Mixed actions
action = np.array([0, 1, 2, 1, 0])  # down, stay, up, stay, down
obs, reward, terminated, truncated, info = env.step(action)
```

### Random Actions
```python
# Sample random discrete actions
action = env.action_space.sample()
print(action)  # e.g., [2, 0, 1, 2, 0]
obs, reward, terminated, truncated, info = env.step(action)
```

## Advantages

### 1. **Intuitive**
- Direct mapping: action[i] controls piston i
- No complex mathematical decoding required
- Easy to understand and debug

### 2. **Consistent**
- Same shape as continuous actions: `(n_pistons,)`
- Similar interface for both continuous and discrete modes
- Predictable behavior

### 3. **Efficient**
- No complex base-3 calculations
- Faster action processing
- Lower computational overhead

### 4. **Flexible**
- Easy to create specific action patterns
- Simple to extend or modify
- Compatible with standard RL libraries

## Comparison with Continuous Actions

| Aspect | Discrete | Continuous |
|--------|----------|------------|
| Action Space | `Box(0, 2, (n_pistons,))` | `Box(-1, 1, (n_pistons,))` |
| Values | `{0, 1, 2}` per piston | `[-1, 1]` per piston |
| Granularity | 3 levels per piston | Infinite precision |
| Interface | Same shape and indexing | Same shape and indexing |
| Conversion | Simple subtraction | Direct use |

## Migration Guide

### For Existing Code
If you have existing code using the old discrete action space:

**Old Code:**
```python
# Old discrete action (single integer)
action = 13
obs, reward, terminated, truncated, info = env.step(action)
```

**New Code:**
```python
# New discrete action (sequence)
action = np.array([1, 1, 1])  # Equivalent to old action 13
obs, reward, terminated, truncated, info = env.step(action)
```

### Action Mapping
For 3 pistons, here are some equivalent actions:

| Old Action | New Action | Description |
|------------|------------|-------------|
| 0 | `[0, 0, 0]` | All down |
| 13 | `[1, 1, 1]` | All stay |
| 26 | `[2, 2, 2]` | All up |
| 5 | `[2, 1, 0]` | up, stay, down |

## Testing

The simplified discrete action space has been thoroughly tested:

```bash
# Run the simplified discrete action test
python test_simplified_discrete.py

# Run all tests
python test_movement_penalty.py
python example.py
```

## Backward Compatibility

**Note**: This change is **not backward compatible** with the old discrete action format. If you have existing code using the old discrete actions, you'll need to update it to use the new sequence format.

However, the continuous action space remains unchanged and fully compatible.

## Benefits for Research

1. **Easier Experimentation**: Researchers can easily create specific action patterns
2. **Better Debugging**: Actions are human-readable and interpretable
3. **Faster Development**: No need to understand complex encoding/decoding
4. **Standard Interface**: Consistent with other multi-agent environments

## Conclusion

The simplified discrete action space makes the pistonball environment more accessible and easier to use while maintaining all the functionality. The direct sequence format is more intuitive and eliminates the complexity of the previous base-3 decoding system. 