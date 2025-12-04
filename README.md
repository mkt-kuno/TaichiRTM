# TaichiRTM - Reverse Time Migration

<div align="left">
  <img src="https://www.python.org/static/community_logos/python-logo-generic.svg"
       alt="Python Logo"
       width="200"
       style="margin-right: 20px;" />
</div>

A GPU-accelerated Reverse Time Migration (RTM) program for seismic exploration using Taichi.

## Features

### Reverse Time Migration for Seismic Exploration
Reverse Time Migration (RTM) is a seismic wave inversion method that calculates reflection cross sections by combining (cross-correlation, convolution, etc.) seismic wave forward and backward propagation data using observed waveform and source information.

This program implements RTM using seismic wave forward/backward propagation modeling of P-SV waves and SH waves.

### Python + GPU Acceleration with Taichi
- **Multi-backend support**: CPU, CUDA GPU, Vulkan, OpenGL, Metal
- **No CUDA dependency**: Works on any system with Taichi support
- **Simple installation**: Just use pip or uv, no conda required

## Installation

### Using uv (Recommended)
```bash
# Install uv if not already installed
pip install uv

# Create virtual environment and install
uv venv
source .venv/bin/activate  # Linux/Mac
# or .venv\Scripts\activate  # Windows

uv pip install -r requirements.txt
```

### Using pip
```bash
pip install numpy taichi matplotlib
```

### From pyproject.toml
```bash
pip install .
# or with visualization support
pip install ".[visualization]"
```

## Quick Start

```python
import numpy as np
from src import ReverseTimeMigration, init_taichi

# Initialize Taichi with desired backend
# Options: 'cpu', 'gpu', 'cuda', 'vulkan', 'opengl', 'metal'
init_taichi(backend='cpu')

# Load your seismic data
npz = np.load('your_data.npz')

# Create RTM instance
rtm = ReverseTimeMigration(
    observed_u=npz['x'],      # Observed velocity data (x-axis)
    observed_v=npz['y'],      # Observed velocity data (y-axis)
    observed_w=npz['z'],      # Observed velocity data (z-axis)
    source_u=source_wavelet_x,
    source_v=source_wavelet_y,
    source_w=source_wavelet_z,
    receiver_loc=distances,    # Receiver positions
    source_loc=source_x,       # Source position
    fs=100000,                 # Sampling frequency (Hz)
    v_fix=120,                 # Fixed velocity (m/s)
)

# Run RTM
rtm.run()

# Save results
rtm.save_result(directory='results/', savename='output')
```

## Examples

Sample waveform files in np.array format are provided. See `examples/example.py` for the analysis procedure.

**Example rho model and calculated cross-section:**

<img src="https://github.com/HaraandYutaro/Reverse-Time-Migration/blob/main/examples/ex_model/Ex_rhomodel.png" width="300" alt="Example rho model" /> 
<img src="https://github.com/HaraandYutaro/Reverse-Time-Migration/blob/main/examples/results/RTMimages/y_120.png" width="400" alt="Calculated cross-section" />

### Running the Example

```bash
cd examples
python example.py
```

### Backend Selection

You can select the computation backend when initializing Taichi:

```python
from src import init_taichi

# CPU backend (default, works everywhere)
init_taichi(backend='cpu')

# GPU backends (requires compatible hardware)
init_taichi(backend='gpu')      # Auto-select best GPU backend
init_taichi(backend='cuda')     # NVIDIA CUDA
init_taichi(backend='vulkan')   # Vulkan (cross-platform GPU)
init_taichi(backend='opengl')   # OpenGL
init_taichi(backend='metal')    # Apple Metal (macOS)
```

## Usage Guide

### 1. Input Data
Prepare record waveforms in numpy array format (shape: `[num_receivers, num_samples]`)

### 2. Parameter Settings
Set parameters such as:
- Analysis area size
- Sampling rate
- Wave velocity
- Source/receiver locations

See `examples/example.py` for details.

### 3. RTM Execution
Call the RTM core functions. The wave is propagated forward and backward, and imaging results are generated according to the imaging conditions (e.g., cross-correlation).

### 4. Visualization
The library provides imaging utilities without matplotlib dependency in core modules. For visualization, you can:
- Use the provided callback functions with matplotlib
- Implement your own visualization using any library
- Load the saved npz results and visualize separately

```python
from src import create_visualization_data

# Get visualization-ready data
vis_data = create_visualization_data(rtm_instance)

# Use with matplotlib or any other library
import matplotlib.pyplot as plt
plt.imshow(vis_data['u'], extent=vis_data['extent'], cmap='gray')
plt.show()
```

## API Reference

### Main Classes

- `ReverseTimeMigration`: Main RTM class
- `ForwardModeling`: Forward wave propagation
- `BackwardModeling`: Backward wave propagation

### Utility Functions

- `init_taichi(backend)`: Initialize Taichi with specified backend
- `load_rtm_results(directory)`: Load RTM results from directory
- `stack_rtm_images(results)`: Stack multiple RTM images
- `create_visualization_data(rtm)`: Prepare data for visualization

## Coordinate System

```
       Source   |---sensors--|
        o-------#---#---#---#--> x (wave propagation direction)
       /|                       
      / |
     /  |
    /   |
   y    | 
        z (vertical/depth direction)

Components:
  u: x-axis velocity
  v: y-axis velocity  
  w: z-axis velocity
```

## Contributions

We welcome issues and pull requests. Please feel free to contact us with bug reports, feature requests, etc.

## Future Prospects

We are working on:
- Support for different file formats (e.g., SEG-Y files)
- Additional imaging conditions
- Performance optimizations

## License

This project is licensed under the GNU Lesser General Public License v2.1 or later (LGPL-2.1-or-later).
