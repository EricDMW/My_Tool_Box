# Creating Custom Maps for AJLATT Environment

This guide explains how to create custom maps for the Advanced Joint Localization and Target Tracking (AJLATT) Environment, including detailed methods for adding obstacles.

## Map Types

AJLATT supports two types of maps:
1. **Static Maps**: Fixed obstacle configurations defined in YAML and CFG files
2. **Dynamic Maps**: Obstacles that move or change over time using pre-defined obstacle libraries

## 🚀 Advanced Map Generation System

The AJLATT environment includes an advanced map generation system that allows you to create complex maps using YAML configuration files. This system automatically generates all necessary files (YAML, CFG, and visualizations) from a single configuration.

### Quick Start with Advanced Map Generator

```bash
# Create a sample configuration
python env/map_generate.py --create-sample my_custom_map

# Generate map from configuration
python env/map_generate.py my_custom_map.yaml --preview

# Generate map with custom output directory
python env/map_generate.py my_custom_map.yaml --output-dir ./my_maps
```

### YAML Configuration Format

The advanced map generator uses a comprehensive YAML format that defines all map parameters:

```yaml
# my_custom_map.yaml
map_info:
  name: "my_custom_map"
  width: 72.4          # Map width in meters
  height: 72.4         # Map height in meters
  resolution: 0.4      # Grid resolution in meters
  add_boundary: true   # Whether to add boundary walls

obstacles:
  # Rectangular obstacles
  - type: "rectangle"
    center: [20.0, 20.0]     # Center coordinates [x, y] in meters
    width: 10.0              # Width in meters
    height: 5.0              # Height in meters
    angle: 0.0               # Rotation angle in radians
  
  # Circular obstacles
  - type: "circle"
    center: [50.0, 50.0]     # Center coordinates [x, y] in meters
    radius: 5.0              # Radius in meters
  
  # Line obstacles (walls)
  - type: "line"
    start: [0.0, 30.0]       # Start point [x, y] in meters
    end: [50.0, 30.0]        # End point [x, y] in meters
    width: 0.5               # Line width in meters
  
  # Maze pattern
  - type: "maze"
    cell_size: 8.0           # Size of each maze cell in meters
    wall_thickness: 0.5      # Wall thickness in meters
    gaps:                    # Gaps in the maze walls
      - type: "horizontal"
        position: 20         # Grid position of the gap
      - type: "vertical"
        position: 25
  
  # Random obstacles
  - type: "random"
    num_obstacles: 10        # Number of random obstacles
    min_size: 1.0           # Minimum obstacle size in meters
    max_size: 3.0           # Maximum obstacle size in meters
```

### Supported Obstacle Types

#### 1. Rectangle Obstacles
```yaml
- type: "rectangle"
  center: [x, y]           # Center position in meters
  width: float             # Width in meters
  height: float            # Height in meters
  angle: float             # Rotation angle in radians (optional, default: 0)
```

#### 2. Circular Obstacles
```yaml
- type: "circle"
  center: [x, y]           # Center position in meters
  radius: float            # Radius in meters
```

#### 3. Line Obstacles (Walls)
```yaml
- type: "line"
  start: [x, y]            # Start point in meters
  end: [x, y]              # End point in meters
  width: float             # Line width in meters (optional, default: resolution)
```

#### 4. Maze Patterns
```yaml
- type: "maze"
  cell_size: float         # Size of each maze cell in meters
  wall_thickness: float    # Wall thickness in meters
  gaps:                    # List of gaps in walls
    - type: "horizontal"   # or "vertical"
      position: int        # Grid position of the gap
```

#### 5. Random Obstacles
```yaml
- type: "random"
  num_obstacles: int       # Number of random obstacles
  min_size: float         # Minimum obstacle size in meters
  max_size: float         # Maximum obstacle size in meters
```

### Programmatic Map Generation

You can also use the map generator programmatically:

```python
from env.map_generate import AdvancedMapGenerator

# Create map generator
generator = AdvancedMapGenerator('my_config.yaml')

# Generate the map
map_data = generator.generate_map()

# Preview the map
generator.preview_map()

# Save all files
generator.save_map_files('./output_directory')
```

### Testing the Map Generation System

Run the comprehensive test suite:

```bash
python test_map_generation.py
```

This will test:
- Basic map generation
- Advanced obstacle types
- Sample configurations
- Environment integration

## Static Maps

### 1. YAML Configuration

Create a YAML file with the following structure:

```yaml
# my_static_map.yaml
datatype: t                    # Data type (usually 't' for text)
mapdim: [181, 181]            # Map dimensions [rows, columns]
mapmax: [72.4, 72.4]          # Maximum coordinates [x_max, y_max]
mapmin: [0.0, 0.0]            # Minimum coordinates [x_min, y_min]
mappath: my_static_map.cfg     # Path to the CFG file (without extension)
mapres: [0.4, 0.4]            # Map resolution [x_res, y_res]
origin: [36.2, 36.2]          # Origin coordinates [x, y]
origincells: [90, 90]         # Origin in cell coordinates [row, col]
storage: colmajor             # Storage format
```

### 2. CFG File Creation with Obstacles

The CFG file contains the actual occupancy grid data. Here are several methods to create maps with obstacles:

#### Method 1: Using the Built-in Map Generator

```python
from env.maps.map_utils import generate_map
import numpy as np

# Generate a basic map with predefined obstacles
generate_map('my_static_map', mapdim=(181, 181), mapres=0.4)
```

#### Method 2: Creating Custom Obstacles Programmatically

```python
import numpy as np

def create_custom_map_with_obstacles(mapname, mapdim=(181, 181), mapres=0.4):
    """
    Create a custom map with specific obstacles.
    
    Args:
        mapname: Name of the map file (without extension)
        mapdim: Map dimensions in meters [width, height]
        mapres: Map resolution in meters
    """
    # Calculate grid dimensions
    grid_width = int(mapdim[0] / mapres)
    grid_height = int(mapdim[1] / mapres)
    
    # Initialize empty map
    new_map = np.zeros((grid_width, grid_height), dtype=np.int8)
    
    # Add boundary walls
    new_map[0, :] = 1.0      # Bottom wall
    new_map[:, 0] = 1.0      # Left wall
    new_map[-1, :] = 1.0     # Top wall
    new_map[:, -1] = 1.0     # Right wall
    
    # Add rectangular obstacles
    # Obstacle 1: Rectangle at position (20, 20) with size (10, 5)
    x1, y1 = int(20/mapres), int(20/mapres)
    w1, h1 = int(10/mapres), int(5/mapres)
    new_map[x1:x1+w1, y1:y1+h1] = 1.0
    
    # Obstacle 2: Rectangle at position (50, 40) with size (8, 12)
    x2, y2 = int(50/mapres), int(40/mapres)
    w2, h2 = int(8/mapres), int(12/mapres)
    new_map[x2:x2+w2, y2:y2+h2] = 1.0
    
    # Add L-shaped obstacle
    # Base of L
    x3, y3 = int(10/mapres), int(50/mapres)
    w3, h3 = int(15/mapres), int(5/mapres)
    new_map[x3:x3+w3, y3:y3+h3] = 1.0
    
    # Vertical part of L
    x4, y4 = int(20/mapres), int(50/mapres)
    w4, h4 = int(5/mapres), int(15/mapres)
    new_map[x4:x4+w4, y4:y4+h4] = 1.0
    
    # Add corridor-like structure
    # Left corridor wall
    x5, y5 = int(30/mapres), int(10/mapres)
    w5, h5 = int(3/mapres), int(40/mapres)
    new_map[x5:x5+w5, y5:y5+h5] = 1.0
    
    # Right corridor wall
    x6, y6 = int(45/mapres), int(10/mapres)
    w6, h6 = int(3/mapres), int(40/mapres)
    new_map[x6:x6+w6, y6:y6+h6] = 1.0
    
    # Save the map
    np.savetxt(f"{mapname}.cfg", new_map, fmt='%d')
    print(f"Map saved as {mapname}.cfg")
    print(f"Map dimensions: {new_map.shape}")
    print(f"Obstacle cells: {np.sum(new_map)}")

# Create the map
create_custom_map_with_obstacles('my_static_map')
```

#### Method 3: Interactive Obstacle Drawing

```python
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def create_interactive_map(mapname, mapdim=(181, 181), mapres=0.4):
    """
    Create a map interactively by clicking to place obstacles.
    """
    grid_width = int(mapdim[0] / mapres)
    grid_height = int(mapdim[1] / mapres)
    
    # Initialize empty map
    map_data = np.zeros((grid_width, grid_height), dtype=np.int8)
    
    # Add boundary walls
    map_data[0, :] = 1.0
    map_data[:, 0] = 1.0
    map_data[-1, :] = 1.0
    map_data[:, -1] = 1.0
    
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(map_data, cmap='gray_r', origin='lower')
    ax.set_title('Click to add obstacles. Press Enter when done.')
    
    obstacles = []
    
    def onclick(event):
        if event.inaxes != ax:
            return
        
        # Convert click coordinates to grid coordinates
        x = int(event.xdata)
        y = int(event.ydata)
        
        if 0 <= x < grid_width and 0 <= y < grid_height:
            # Add obstacle
            map_data[x, y] = 1
            obstacles.append([x, y])
            
            # Update display
            ax.clear()
            ax.imshow(map_data, cmap='gray_r', origin='lower')
            ax.set_title(f'Obstacles added: {len(obstacles)}. Press Enter when done.')
            plt.draw()
    
    def on_key(event):
        if event.key == 'enter':
            # Save the map
            np.savetxt(f"{mapname}.cfg", map_data, fmt='%d')
            print(f"Map saved as {mapname}.cfg with {len(obstacles)} obstacles")
            plt.close()
    
    fig.canvas.mpl_connect('button_press_event', onclick)
    fig.canvas.mpl_connect('key_press_event', on_key)
    plt.show()

# Create interactive map
create_interactive_map('my_interactive_map')
```

#### Method 4: Advanced Obstacle Patterns

```python
def create_advanced_obstacles(mapname, mapdim=(181, 181), mapres=0.4):
    """
    Create complex obstacle patterns for testing different scenarios.
    """
    grid_width = int(mapdim[0] / mapres)
    grid_height = int(mapdim[1] / mapres)
    
    new_map = np.zeros((grid_width, grid_height), dtype=np.int8)
    
    # Add boundary walls
    new_map[0, :] = 1.0
    new_map[:, 0] = 1.0
    new_map[-1, :] = 1.0
    new_map[:, -1] = 1.0
    
    # Create maze-like structure
    # Horizontal walls
    for i in range(5, grid_width-5, 15):
        new_map[i:i+2, 5:grid_height-5] = 1.0
    
    # Vertical walls
    for j in range(5, grid_height-5, 15):
        new_map[5:grid_width-5, j:j+2] = 1.0
    
    # Add some gaps for navigation
    new_map[10:12, 10:12] = 0.0
    new_map[25:27, 25:27] = 0.0
    new_map[40:42, 40:42] = 0.0
    
    # Add circular obstacle (approximated with squares)
    center_x, center_y = int(60/mapres), int(60/mapres)
    radius = int(8/mapres)
    for i in range(grid_width):
        for j in range(grid_height):
            if (i - center_x)**2 + (j - center_y)**2 <= radius**2:
                new_map[i, j] = 1.0
    
    # Add diagonal obstacles
    for k in range(20, 40, 2):
        new_map[k, k] = 1.0
        new_map[k, k+1] = 1.0
    
    np.savetxt(f"{mapname}.cfg", new_map, fmt='%d')
    print(f"Advanced map saved as {mapname}.cfg")

create_advanced_obstacles('maze_map')
```

### 3. Using the Custom Map

```python
import env_lib
import torch

# Create environment with custom map
env = env_lib.ajlatt_env(
    map_name='my_static_map',  # Your map name (without .yaml extension)
    num_Robot=4,
    num_targets=1
)

# Test the environment
obs = env.reset()
for step in range(1000):
    action = torch.rand(4, 2)  # Random actions for 4 robots
    obs, reward, done, info = env.step(action)
    env.render()
    
    if done:
        obs = env.reset()
```

## Dynamic Maps

### 1. Creating Obstacle Libraries

Dynamic maps use pre-defined obstacle objects stored as NumPy arrays. Create obstacle objects using the provided drawing tool:

```bash
# Run the obstacle drawing tool
python env/maps/draw_obstacles.py --log_dir ./my_obstacles

# Interactive drawing interface:
# - Click to define rectangle corners
# - Press 'f' to fill the rectangle
# - Press 'n' to save and move to next obstacle
# - Press 'c' to clear current obstacle
```

### 2. Programmatic Obstacle Library Creation

```python
import numpy as np
import os

def create_obstacle_library(lib_dir='./my_obstacles'):
    """
    Create a library of obstacle objects programmatically.
    """
    os.makedirs(lib_dir, exist_ok=True)
    
    # Obstacle 1: Small square
    obstacle_1 = np.zeros((20, 20), dtype=np.int8)
    obstacle_1[5:15, 5:15] = 1
    np.save(os.path.join(lib_dir, 'obstacle_0.npy'), obstacle_1)
    
    # Obstacle 2: Rectangle
    obstacle_2 = np.zeros((30, 20), dtype=np.int8)
    obstacle_2[5:25, 5:15] = 1
    np.save(os.path.join(lib_dir, 'obstacle_1.npy'), obstacle_2)
    
    # Obstacle 3: L-shape
    obstacle_3 = np.zeros((25, 25), dtype=np.int8)
    obstacle_3[5:20, 5:10] = 1  # Horizontal part
    obstacle_3[15:20, 5:20] = 1  # Vertical part
    np.save(os.path.join(lib_dir, 'obstacle_2.npy'), obstacle_3)
    
    # Obstacle 4: Cross shape
    obstacle_4 = np.zeros((25, 25), dtype=np.int8)
    obstacle_4[10:15, 5:20] = 1  # Horizontal
    obstacle_4[5:20, 10:15] = 1  # Vertical
    np.save(os.path.join(lib_dir, 'obstacle_3.npy'), obstacle_4)
    
    print(f"Created obstacle library in {lib_dir}")

create_obstacle_library()
```

### 3. Dynamic Map YAML Configuration

```yaml
# my_dynamic_map.yaml
datatype: t
mapdim: [181, 181]
mapmax: [72.4, 72.4]
mapmin: [0.0, 0.0]
mappath: my_dynamic_map.cfg
mapres: [0.4, 0.4]
origin: [36.2, 36.2]
origincells: [90, 90]
storage: colmajor
submaporigin: [45, 45, 45, 135, 135, 45, 135, 135]  # Submap origins for obstacle placement
lib_path: my_obstacles                               # Path to obstacle library
```

### 4. Creating a Custom Dynamic Map Class

```python
# my_custom_dynamic_map.py
import numpy as np
from env.maps.dynamic_map import DynamicMap
from skimage.transform import rotate

class MyCustomDynamicMap(DynamicMap):
    def __init__(self, map_dir_path, map_name, map_path, margin2wall=0.5):
        super().__init__(map_dir_path, map_name, map_path, margin2wall)
        
    def generate_map(self, chosen_idx=None, rot_angs=None, time_step=0, **kwargs):
        """
        Generate dynamic map with custom obstacle placement logic.
        
        Args:
            chosen_idx: Indices of obstacles to use (default: random selection)
            rot_angs: Rotation angles for obstacles (default: random)
            time_step: Current time step for time-varying obstacles
        """
        self.map = np.zeros(self.mapdim)
        
        if chosen_idx is None:
            # Custom obstacle selection logic
            chosen_idx = np.random.choice(len(self.obstacles), 4, replace=False)
        
        if rot_angs is None:
            # Custom rotation logic
            rot_angs = [np.random.choice(np.arange(-10, 10, 1) / 10. * 180) for _ in range(4)]
        
        # Place obstacles with custom logic
        for (i, c_id) in enumerate(chosen_idx):
            # Add time-varying behavior
            time_offset = time_step * 0.1
            base_rotation = rot_angs[i]
            dynamic_rotation = base_rotation + 10 * np.sin(time_offset + i)
            
            # Rotate and place obstacle
            rotated_obs = rotate(self.obstacles[c_id], dynamic_rotation, 
                               resize=True, center=(24, 24))
            
            # Calculate global coordinates
            rotated_obs_idx_local = np.array(np.nonzero(rotated_obs))
            rotated_obs_idx_global_0 = rotated_obs_idx_local[0] \
                - int(rotated_obs.shape[0]/2) + self.submap_coordinates[i][0]
            rotated_obs_idx_global_1 = rotated_obs_idx_local[1] \
                - int(rotated_obs.shape[1]/2) + self.submap_coordinates[i][1]
            
            # Place in map
            self.map[rotated_obs_idx_global_0, rotated_obs_idx_global_1] = 1.0
        
        # Add walls
        self.map[0, :] = 1.0
        self.map[-1, :] = 1.0
        self.map[:, 0] = 1.0
        self.map[:, -1] = 1.0
        
        # Update linear representation
        self.map_linear = np.squeeze(self.map.astype(np.int8).reshape(-1, 1))
        
        # Store for debugging
        self.chosen_idx = chosen_idx
        self.rot_angs = rot_angs
```

### 5. Using the Dynamic Map

```python
import env_lib

# Create environment with dynamic map
env = env_lib.ajlatt_env(
    map_name='my_dynamic_map',
    num_Robot=4,
    num_targets=1
)

# The environment will automatically use the dynamic map
obs = env.reset()
```

## Map Coordinate System

### Important Coordinate Conventions

The map uses a specific coordinate system:

```python
# Coordinate system notes:
# - (xmin, ymin) is at the bottom left corner
# - Rows are counted from bottom to top
# - Columns are counted from left to right
# - Matplotlib displays with (0,0) at top left, so maps are flipped for display

# Coordinate conversion functions:
def se2_to_cell(pos, mapmin, mapres):
    """Convert SE2 position to cell coordinates"""
    cell_idx = (pos - mapmin) / mapres - 0.5
    return round(cell_idx[0]), round(cell_idx[1])

def cell_to_se2(cell_idx, mapmin, mapres):
    """Convert cell coordinates to SE2 position"""
    return (np.array(cell_idx) + 0.5) * mapres + mapmin
```

## Available Map Utilities

### 1. GridMap Class

The `GridMap` class provides comprehensive map functionality:

```python
from env.maps.map_utils import GridMap

# Create map object
map_obj = GridMap('path/to/map', margin2wall=0.5)

# Check collision
collision = map_obj.is_collision([x, y], margin=0.5)

# Get closest obstacle
closest = map_obj.get_closest_obstacle([x, y, theta], r_max=3.0)

# Check if position is in bounds
in_bounds = map_obj.in_bound([x, y])

# Ray casting for line of sight
blocked = map_obj.is_blocked([x1, y1], [x2, y2])
```

### 2. Map Generation Tools

```python
from env.maps.map_utils import generate_map, generate_trajectory

# Generate a new map
generate_map('new_map', mapdim=(181, 181), mapres=0.4)

# Generate trajectory (interactive)
generate_trajectory(map_obj)
```

## Advanced Map Features

### 1. Custom Obstacle Shapes

Create complex obstacle shapes using the drawing tool:

```python
# Example: Creating L-shaped obstacle
# 1. Run draw_obstacles.py
# 2. Click to define rectangle corners
# 3. Press 'f' to fill
# 4. Repeat for complex shapes
# 5. Press 'n' to save
```

### 2. Time-Varying Obstacles

```python
class TimeVaryingMap(DynamicMap):
    def generate_map(self, chosen_idx=None, rot_angs=None, time_step=0, **kwargs):
        # Obstacles that move in patterns
        for i, c_id in enumerate(chosen_idx):
            # Circular motion
            angle = time_step * 0.1 + i * np.pi/2
            x_offset = 10 * np.cos(angle)
            y_offset = 10 * np.sin(angle)
            
            # Apply offset to obstacle position
            # ... implementation details
```

### 3. Environment-Specific Maps

```python
class OfficeEnvironment(DynamicMap):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.desk_positions = [
            [10, 10], [40, 10], [10, 40], [40, 40]
        ]
    
    def generate_map(self, **kwargs):
        # Add static office furniture
        for desk_pos in self.desk_positions:
            # Place desk obstacles
            pass
        
        # Add moving obstacles (people, robots)
        if time_step % 200 < 100:
            # Add temporary obstacles
            pass
```

## Map Validation and Testing

### 1. Visual Inspection

```python
import matplotlib.pyplot as plt
import env_lib

# Create environment with your map
env = env_lib.ajlatt_env(map_name='my_map')

# Reset and render
obs = env.reset()
env.render()

# Keep the plot open
plt.show()
```

### 2. Collision Testing

```python
def test_map_collisions():
    """Test if robots can navigate the map without collisions."""
    env = env_lib.ajlatt_env(
        map_name='my_map', 
        num_Robot=4, 
        render=False
    )
    
    obs = env.reset()
    collision_count = 0
    
    for step in range(1000):
        action = torch.rand(4, 2)  # Random actions
        obs, reward, done, info = env.step(action)
        
        if done:
            collision_count += 1
            obs = env.reset()
    
    print(f"Collisions: {collision_count}")
    env.close()
```

### 3. Map Properties Validation

```python
def validate_map_properties():
    """Validate map properties and constraints."""
    map_obj = GridMap('path/to/map')
    
    # Check map bounds
    print(f"Map bounds: {map_obj.mapmin} to {map_obj.mapmax}")
    print(f"Map resolution: {map_obj.mapres}")
    print(f"Map dimensions: {map_obj.mapdim}")
    
    # Test coordinate conversions
    test_pos = [25, 25]
    cell = map_obj.se2_to_cell(test_pos)
    pos_back = map_obj.cell_to_se2(cell)
    print(f"Coordinate conversion test: {test_pos} -> {cell} -> {pos_back}")
```

## Best Practices

### 1. Map Design Guidelines

- **Size**: Keep maps reasonably sized (50x50 to 100x100 meters)
- **Resolution**: Use 0.4m resolution for good balance of detail and performance
- **Obstacle Density**: Don't make maps too cluttered - robots need navigation space
- **Wall Margins**: Use appropriate `margin2wall` values (0.5-1.0m)

### 2. Performance Considerations

- **Grid Size**: Larger grids require more memory and computation
- **Obstacle Complexity**: Complex obstacles increase collision detection time
- **Dynamic Updates**: Frequent map changes can impact performance

### 3. Testing Strategy

- **Unit Tests**: Test individual map functions
- **Integration Tests**: Test map with environment
- **Visual Validation**: Always visually inspect generated maps
- **Collision Testing**: Verify collision detection works correctly

## Troubleshooting

### Common Issues

1. **Map not found**: Ensure YAML and CFG files are in the correct directory
2. **Invalid coordinates**: Check that coordinates are within map bounds
3. **Performance issues**: Reduce map complexity or increase resolution
4. **Collision detection errors**: Verify obstacle data format

### Debug Tips

```python
# Enable debug mode
import logging
logging.basicConfig(level=logging.DEBUG)

# Check map properties
print(f"Map bounds: {map_obj.mapmin} to {map_obj.mapmax}")
print(f"Obstacle count: {len(map_obj.obstacles) if hasattr(map_obj, 'obstacles') else 'N/A'}")

# Test specific positions
test_pos = [25, 25]
closest_obstacle = map_obj.get_closest_obstacle(test_pos)
print(f"Closest obstacle to {test_pos}: {closest_obstacle}")

# Visualize map
import matplotlib.pyplot as plt
plt.imshow(map_obj.map, cmap='gray_r')
plt.show()
```

## Example: Complete Custom Map Creation

### Step 1: Create Static Map with Obstacles

```python
# Create YAML configuration
yaml_content = """
datatype: t
mapdim: [181, 181]
mapmax: [72.4, 72.4]
mapmin: [0.0, 0.0]
mappath: custom_static.cfg
mapres: [0.4, 0.4]
origin: [36.2, 36.2]
origincells: [90, 90]
storage: colmajor
"""

with open('custom_static.yaml', 'w') as f:
    f.write(yaml_content)

# Create CFG file with obstacles
def create_custom_static_map():
    grid_width = int(181)
    grid_height = int(181)
    
    new_map = np.zeros((grid_width, grid_height), dtype=np.int8)
    
    # Add boundary walls
    new_map[0, :] = 1.0
    new_map[:, 0] = 1.0
    new_map[-1, :] = 1.0
    new_map[:, -1] = 1.0
    
    # Add rectangular obstacles
    new_map[20:40, 20:30] = 1.0  # Large rectangle
    new_map[60:80, 50:70] = 1.0  # Another rectangle
    
    # Add L-shaped obstacle
    new_map[30:50, 60:65] = 1.0  # Horizontal part
    new_map[45:50, 60:80] = 1.0  # Vertical part
    
    # Add corridor
    new_map[40:45, 10:60] = 1.0  # Left wall
    new_map[40:45, 80:130] = 1.0  # Right wall
    
    np.savetxt('custom_static.cfg', new_map, fmt='%d')
    print("Custom static map created")

create_custom_static_map()
```

### Step 2: Create Dynamic Map

```python
# Create obstacle library
create_obstacle_library('./custom_obstacles')

# Create dynamic map YAML
dynamic_yaml = """
datatype: t
mapdim: [181, 181]
mapmax: [72.4, 72.4]
mapmin: [0.0, 0.0]
mappath: custom_dynamic.cfg
mapres: [0.4, 0.4]
origin: [36.2, 36.2]
origincells: [90, 90]
storage: colmajor
submaporigin: [45, 45, 45, 135, 135, 45, 135, 135]
lib_path: custom_obstacles
"""

with open('custom_dynamic.yaml', 'w') as f:
    f.write(dynamic_yaml)
```

### Step 3: Test the Maps

```python
import env_lib
import torch

# Test static map
env_static = env_lib.ajlatt_env(map_name='custom_static')
obs = env_static.reset()
env_static.render()

# Test dynamic map
env_dynamic = env_lib.ajlatt_env(map_name='custom_dynamic')
obs = env_dynamic.reset()
env_dynamic.render()

# Run a simple test
for step in range(100):
    action = torch.rand(4, 2)
    obs, reward, done, info = env_static.step(action)
    env_static.render()
    
    if done:
        obs = env_static.reset()
```

This comprehensive guide covers all aspects of creating custom maps for the AJLATT environment, from basic static maps to advanced dynamic obstacle systems, with detailed methods for adding various types of obstacles and a complete advanced map generation system. 