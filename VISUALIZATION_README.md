# Visualization Module Documentation

## Overview

The visualization module provides a comprehensive, object-oriented system for visualizing LBM simulations and environments. It follows clean code principles with clear separation of concerns:

- **Field Visualizers**: Abstract interface for different field types (velocity, vorticity, pressure)
- **Plotters**: Handles real-time plotting and figure management
- **Video Recording**: Specialized class for FFMpeg-based video export
- **Environment Visualization**: Displays simulation setup and initial conditions

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    LBMSolver                                │
│  (delegates visualization to external modules)              │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
          ┌────────────────────────┐
          │ SimulationVisualizer   │
          │ (high-level interface) │
          └────────────┬───────────┘
                       │
        ┌──────────────┴──────────────┐
        ▼                             ▼
┌─────────────────────┐      ┌──────────────────┐
│ FlowFieldPlotter    │      │ FlowVideoRecorder│
│ (live plotting)     │      │ (FFMpeg export)  │
└─────────┬───────────┘      └──────────────────┘
          │
          ▼
   ┌──────────────────────────┐
   │ FieldVisualizer (ABC)    │
   │ - Velocity Magnitude     │
   │ - Vorticity              │
   │ - Pressure               │
   │ - Custom (user-defined)  │
   └──────────────────────────┘
```

## Class Reference

### FieldVisualizer (Abstract Base Class)

Base class for all field visualization strategies.

**Methods:**
- `compute_field(ux, uy, rho=None)`: Compute the field to visualize
- `get_plot_params()`: Return dict with 'cmap', 'vmin', 'vmax', 'label'
- `get_title(step=None)`: Return plot title string

**Concrete Implementations:**

#### VelocityMagnitudeVisualizer
Visualizes |u| = sqrt(ux² + uy²)

```python
from visualization import VelocityMagnitudeVisualizer

visualizer = VelocityMagnitudeVisualizer(u_inlet=0.12, cmap='viridis')
```

#### VorticityVisualizer
Visualizes ω = ∂uy/∂x - ∂ux/∂y (curl of velocity)

```python
from visualization import VorticityVisualizer

visualizer = VorticityVisualizer(cmap='RdBu_r')
```

#### PressureVisualizer
Visualizes pressure field (proportional to density)

```python
from visualization import PressureVisualizer

visualizer = PressureVisualizer(cmap='coolwarm')
```

### EnvironmentVisualizer

Visualize simulation setup (geometry, initial conditions).

```python
from visualization import EnvironmentVisualizer
from envirmonment import KarmannVortex

env = KarmannVortex(v0=1.0)
env_vis = EnvironmentVisualizer(env, nx=300, ny=120)
fig, ax = env_vis.plot_environment(show_initial_conditions=True)
```

### FlowFieldPlotter

Real-time flow field visualization with interactive mode.

```python
from visualization import FlowFieldPlotter, VelocityMagnitudeVisualizer

plotter = FlowFieldPlotter(nx=150, ny=60, obstacle=obstacle_mask)
plotter.enable_interactive_mode()

# In simulation loop:
visualizer = VelocityMagnitudeVisualizer(u_inlet=0.12)
field = visualizer.compute_field(ux, uy)
plotter.plot_field(field, visualizer, step=100)
plotter.pause(interval=0.001)

plotter.disable_interactive_mode()
plotter.save('final_frame.png')
```

### FlowVideoRecorder

Record simulations to video file using FFMpeg.

```python
from visualization import FlowVideoRecorder, VorticityVisualizer

recorder = FlowVideoRecorder(nx=150, ny=60, obstacle=obstacle_mask, fps=30)
visualizer = VorticityVisualizer()
recorder.start_recording('output.mp4', visualizer)

# In simulation loop:
field = visualizer.compute_field(ux, uy)
recorder.add_frame(field, visualizer, step=step)

recorder.finish_recording()
```

### SimulationVisualizer

High-level interface combining plotting and video recording.

```python
from visualization import SimulationVisualizer, VelocityMagnitudeVisualizer

visualizer = VelocityMagnitudeVisualizer(u_inlet=0.12)
sim_vis = SimulationVisualizer(
    nx=150, ny=60, obstacle=obstacle_mask,
    fps=30,
    record_video=True,
    video_filename='simulation.mp4'
)
sim_vis.setup(visualizer)

# In simulation loop:
field = visualizer.compute_field(ux, uy)
sim_vis.update(field, step)

sim_vis.finalize()
```

## Usage Examples

### Example 1: Simple Live Visualization

```python
from solvers import LBMSolver
from envirmonment import KarmannVortex
from visualization import VelocityMagnitudeVisualizer

env = KarmannVortex(v0=1.0)
solver = LBMSolver(env, nx=300, ny=120, u_inlet=0.12, n_steps=5000)

visualizer = VelocityMagnitudeVisualizer(u_inlet=0.12)
result = solver.solve(visualizer=visualizer, record_video=False)
```

### Example 2: Record to Video

```python
result = solver.solve(
    visualizer=visualizer,
    record_video=True,
    video_filename='karman_street.mp4'
)
```

### Example 3: Multiple Field Visualizations

```python
from visualization import (
    VelocityMagnitudeVisualizer,
    VorticityVisualizer,
    PressureVisualizer,
    FlowFieldPlotter
)

# Run simulation without visualization
result = solver.solve(visualizer=None)

# Create plotter
plotter = FlowFieldPlotter(
    result['metadata']['nx'],
    result['metadata']['ny'],
    obstacle=result['obstacle']
)

# Visualize different fields
vel_viz = VelocityMagnitudeVisualizer(u_inlet=0.12)
vel_field = vel_viz.compute_field(result['ux'], result['uy'])
plotter.plot_field(vel_field, vel_viz)
plotter.save('velocity.png')

vor_viz = VorticityVisualizer()
vor_field = vor_viz.compute_field(result['ux'], result['uy'])
plotter.plot_field(vor_field, vor_viz)
plotter.save('vorticity.png')
```

### Example 4: Custom Field Visualizer

```python
from visualization import FieldVisualizer

class CustomVisualizer(FieldVisualizer):
    def compute_field(self, ux, uy, rho=None):
        # Your custom field computation
        return your_field
    
    def get_plot_params(self):
        return {
            'cmap': 'viridis',
            'vmin': 0,
            'vmax': 1,
            'label': 'Custom field'
        }
    
    def get_title(self, step=None):
        return f"Custom field — step {step}" if step else "Custom field"

visualizer = CustomVisualizer()
result = solver.solve(visualizer=visualizer)
```

## Design Principles

### 1. **Separation of Concerns**
- **Solvers** handle physics computation
- **Visualizers** handle field extraction and rendering parameters
- **Plotters** handle matplotlib figure management
- **Recorders** handle video export

### 2. **Strategy Pattern**
FieldVisualizer subclasses implement different visualization strategies without changing solver code.

### 3. **Composition over Inheritance**
SimulationVisualizer composes FlowFieldPlotter and FlowVideoRecorder rather than inheriting.

### 4. **Open/Closed Principle**
Easy to add new visualizers (open for extension) without modifying existing code (closed for modification).

## Requirements

### Core Requirements
- numpy
- matplotlib

### Optional Requirements (for video export)
- matplotlib (FFMpegWriter backend)
- FFMpeg (system dependency)

Install FFMpeg:
- **Ubuntu/Debian**: `sudo apt-get install ffmpeg`
- **macOS**: `brew install ffmpeg`
- **Windows**: Download from https://ffmpeg.org/download.html or use `choco install ffmpeg`

## Tips and Troubleshooting

### Video Recording Not Working

If video recording fails, check:
1. FFMpeg is installed: `ffmpeg -version`
2. matplotlib has FFMpeg backend: Check your matplotlib installation
3. Disk space available for video output
4. Output directory exists and is writable

The module will gracefully fall back to live plotting if FFMpeg is unavailable.

### Performance with Large Grids

For large grids (>500×500):
- Reduce visualization frequency: Update every 100-200 steps instead of every 10
- Disable video recording (reduces overhead)
- Use lower DPI for figures: `dpi=75` instead of 100

### Custom Colormaps

All visualizers accept matplotlib colormap names:

```python
visualizer = VelocityMagnitudeVisualizer(cmap='plasma')
visualizer = VorticityVisualizer(cmap='seismic')
visualizer = PressureVisualizer(cmap='Spectral')
```

See https://matplotlib.org/stable/users/explain/colors/colormaps.html for available colormaps.

## Complete Example

See `visualization_examples.py` for a runnable demonstration of all features.

```bash
python visualization_examples.py
```

This will:
1. Visualize environment setup
2. Run quick simulation with live plotting
3. Run simulation with video recording
4. Generate snapshots of different field types
5. Demonstrate custom field visualizer

