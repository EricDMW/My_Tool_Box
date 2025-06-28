# TkAgg Backend Setup Guide

This guide helps you set up the TkAgg backend for interactive matplotlib plots in the environment library.

## What is TkAgg?

TkAgg is a matplotlib backend that uses Tkinter (Python's standard GUI toolkit) to create interactive plot windows. It's one of the most reliable interactive backends for matplotlib.

## Installation

### Ubuntu/Debian Systems

```bash
# Install Tkinter and related packages
sudo apt-get update
sudo apt-get install python3-tk tk-dev

# If using conda environment
conda install tk
```

### CentOS/RHEL/Fedora Systems

```bash
# Install Tkinter
sudo yum install tkinter
# or for newer versions
sudo dnf install python3-tkinter
```

### macOS

```bash
# Using Homebrew
brew install python-tk

# Using conda
conda install tk
```

### Windows

Tkinter usually comes pre-installed with Python on Windows. If not:

```bash
# Using conda
conda install tk
```

## Testing the Setup

Run the test script to verify TkAgg is working:

```bash
cd envlib_project/env_lib
python test_tkagg.py
```

You should see:
1. A message indicating TkAgg backend is being used
2. An interactive plot window with sine and cosine curves
3. The environment rendering test should also work

## Troubleshooting

### Common Issues

1. **"No module named '_tkinter'"**
   - Solution: Install python3-tk package for your system

2. **"No display name and no $DISPLAY environment variable"**
   - This happens when running on a headless server
   - Solution: Use X11 forwarding or set up a virtual display

3. **"TkAgg not available"**
   - Solution: Install tkinter development packages

### X11 Forwarding (for SSH connections)

If you're running on a remote server via SSH:

```bash
# Connect with X11 forwarding
ssh -X username@server

# Or for trusted X11 forwarding
ssh -Y username@server

# Set display variable if needed
export DISPLAY=:0
```

### Virtual Display (for headless servers)

If you don't have a physical display:

```bash
# Install xvfb
sudo apt-get install xvfb

# Start virtual display
Xvfb :99 -screen 0 1024x768x24 &
export DISPLAY=:99

# Now run your script
python test_tkagg.py
```

### Alternative Backends

If TkAgg doesn't work, the system will automatically try these alternatives:

1. **Qt5Agg** - Requires PyQt5 or PySide2
2. **GTK3Agg** - Requires PyGObject
3. **Agg** - Non-interactive (fallback)

## Environment Configuration

The environment library automatically detects and uses the best available backend:

- **Interactive mode**: Uses TkAgg, Qt5Agg, or GTK3Agg
- **SSH/Headless mode**: Falls back to Agg (non-interactive)
- **Debug mode**: Forces Agg backend

## Usage

Once TkAgg is working, you can use the environment with interactive rendering:

```python
from envlib_project.env_lib.ajlatt_env import ajlatt_env

# Create environment with rendering
env = ajlatt_env(render=True, figID=0)

# Reset and run
obs = env.reset()
for i in range(100):
    action = env.action_space.sample()
    obs, reward, done, truncated, info = env.step(action)
    env.render()  # This will show interactive plots
    if done:
        break
```

## Performance Tips

1. **Reduce update frequency**: Use `skip` parameter in Display2D
2. **Close unused figures**: Call `plt.close()` when done
3. **Use non-interactive mode**: Set `render=False` for training

## Support

If you continue to have issues:

1. Check your system's Python installation
2. Verify tkinter is available: `python -c "import tkinter; print('tkinter available')"`
3. Try the test script and check error messages
4. Consider using a different backend or non-interactive mode 