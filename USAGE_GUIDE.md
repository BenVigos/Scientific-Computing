# LBM Solver with Visualization - Usage Guide

## Quick Start

### 1. Basic Simulation with Live Visualization

```python
from solvers import LBMSolver
from envirmonment import KarmannVortex
from visualization import VelocityMagnitudeVisualizer

# Create environment
env = KarmannVortex(v0=1.0)

# Create solver
solver = LBMSolver(
    environment=env,
    nx=300,
    ny=120,
    u_inlet=0.12,
    reynolds_number=150,
    n_steps=5000
)

# Create visualizer for velocity magnitude
visualizer = VelocityMagnitudeVisualizer(u_inlet=0.12, cmap='viridis')

# Run simulation with live plotting
result = solver.solve(
    verbose=True,
    visualizer=visualizer,
    record_video=False
)

print(f"Simulation complete!")
print(f"Final ux range: [{result['ux'].min():.4f}, {result['ux'].max():.4f}]")
```

### 2. Simulation with Video Recording

```python
# Same setup as above, but with video recording
result = solver.solve(
    verbose=True,
    visualizer=visualizer,
    record_video=True,
    video_filename='karman_vortex.mp4'
)
```

### 3. Different Field Visualizations

```python
from visualization import (
    VelocityMagnitudeVisualizer,
    VorticityVisualizer,
    PressureVisualizer
)

# Visualize vorticity
vor_visualizer = VorticityVisualizer(cmap='RdBu_r')
result = solver.solve(visualizer=vor_visualizer, record_video=False)

# Visualize pressure
pres_visualizer = PressureVisualizer(cmap='coolwarm')
result = solver.solve(visualizer=pres_visualizer, record_video=False)
```

## Visualization Options

### Available Field Visualizers

1. **VelocityMagnitudeVisualizer**
   - Shows |u| = sqrt(ux² + uy²)
   - Good for overall flow structure
   - Parameters: `u_inlet`, `cmap`

2. **VorticityVisualizer**
   - Shows ω = ∂uy/∂x - ∂ux/∂y
   - Highlights vortex structures
   - Parameters: `cmap`

3. **PressureVisualizer**
   - Shows density field (proportional to pressure)
   - Good for compressibility analysis
   - Parameters: `cmap`

### Custom Field Visualizer

Create your own visualizer for specialized analysis:

```python
from visualization import FieldVisualizer

class KineticEnergyVisualizer(FieldVisualizer):
    """Visualize kinetic energy field."""
    
    def compute_field(self, ux, uy, rho=None):
        """Kinetic energy = 0.5 * (ux^2 + uy^2)"""
        return 0.5 * (ux**2 + uy**2)
    
    def get_plot_params(self):
        return {
            'cmap': 'plasma',
            'vmin': 0,
            'vmax': 0.01,
            'label': 'Kinetic energy'
        }
    
    def get_title(self, step=None):
        if step is not None:
            return f"Kinetic energy — step {step}"
        return "Kinetic energy"

# Use custom visualizer
ke_visualizer = KineticEnergyVisualizer()
result = solver.solve(visualizer=ke_visualizer)
```

## Advanced Usage

### Separate Simulation and Visualization

Run simulation without visualization, then visualize the final result:

```python
from visualization import FlowFieldPlotter

# Run simulation (no visualization overhead)
result = solver.solve(visualizer=None)

# Create plotter
plotter = FlowFieldPlotter(
    result['metadata']['nx'],
    result['metadata']['ny'],
    obstacle=result['obstacle']
)

# Visualize different fields
visualizers = [
    VelocityMagnitudeVisualizer(u_inlet=0.12),
    VorticityVisualizer(),
    PressureVisualizer()
]

for viz in visualizers:
    field = viz.compute_field(result['ux'], result['uy'], result['rho'])
    plotter.plot_field(field, viz)
    plotter.save(f'final_{viz.__class__.__name__}.png')

plotter.close()
```

### Environment Visualization

Visualize the simulation setup before running:

```python
from visualization import EnvironmentVisualizer

env = KarmannVortex(v0=1.0)
env_vis = EnvironmentVisualizer(env, nx=300, ny=120)

fig, ax = env_vis.plot_environment(show_initial_conditions=True)
fig.savefig('environment_setup.png', dpi=150)
```

### Manual Plotting Control

For fine-grained control over visualization:

```python
from visualization import FlowFieldPlotter

plotter = FlowFieldPlotter(300, 120, obstacle=obstacle_mask)
plotter.enable_interactive_mode()

visualizer = VelocityMagnitudeVisualizer(u_inlet=0.12)

# In your custom loop:
for step in range(1, 1000):
    # ... compute ux, uy, rho ...
    
    field = visualizer.compute_field(ux, uy)
    plotter.plot_field(field, visualizer, step=step)
    plotter.pause(interval=0.01)
    
    if step % 100 == 0:
        plotter.save(f'frame_{step}.png')

plotter.disable_interactive_mode()
plotter.close()
```

## Performance Considerations

### Grid Size and Visualization

| Grid Size | Live Plot | Video Record | Notes |
|-----------|-----------|--------------|-------|
| 100×50    | Fast      | Fast         | Good for testing |
| 300×120   | Moderate  | Moderate     | Typical size |
| 600×240   | Slow      | Slow         | Consider disabling viz |
| 1000×400  | Very slow | Very slow    | Visualization only at end |

### Tips for Better Performance

1. **Disable visualization during simulation**, save result, visualize separately:
   ```python
   result = solver.solve(visualizer=None)  # Fast
   # Then visualize at your own pace
   ```

2. **Reduce video recording quality**:
   ```python
   recorder = FlowVideoRecorder(nx, ny, fps=15, dpi=75)  # Lower fps and dpi
   ```

3. **Update visualization less frequently**:
   In `solve()`, change `if step % 100 == 0:` to `if step % 500 == 0:`

4. **Use non-interactive mode for batch processing**:
   ```python
   plotter = FlowFieldPlotter(...)
   # Don't call enable_interactive_mode() - faster
   plotter.plot_field(field, visualizer)
   plotter.save('output.png')
   ```

## Video Output

### Requirements

FFMpeg must be installed:

**Linux (Ubuntu/Debian)**:
```bash
sudo apt-get install ffmpeg
```

**macOS**:
```bash
brew install ffmpeg
```

**Windows**:
- Download from https://ffmpeg.org/download.html
- Or use: `choco install ffmpeg`

### Video Parameters

```python
result = solver.solve(
    visualizer=visualizer,
    record_video=True,
    video_filename='output.mp4'  # File extension determines format
)
```

Supported formats: .mp4, .avi, .mov, .mkv, etc.

### Troubleshooting

If video recording fails:
1. Check FFMpeg is installed: `ffmpeg -version`
2. Ensure output directory exists and is writable
3. Try a different filename/format
4. Check disk space for video file

The module gracefully falls back to live plotting if FFMpeg is unavailable.

## Results and Output

### Return Values

The `solve()` method returns a dictionary with:

```python
{
    'ux': ndarray,              # x-component of velocity
    'uy': ndarray,              # y-component of velocity  
    'rho': ndarray,             # density field
    'obstacle': ndarray,        # boolean obstacle mask
    'metadata': {
        'nx': int,              # grid x-dimension
        'ny': int,              # grid y-dimension
        'u_inlet': float,       # inlet velocity
        'reynolds_number': float,
        'tau': float,           # relaxation time
        'n_steps': int          # total steps run
    }
}
```

### Accessing Results

```python
result = solver.solve(...)

ux = result['ux']                    # Velocity x-component
uy = result['uy']                    # Velocity y-component
speed = np.sqrt(ux**2 + uy**2)       # Speed magnitude
rho = result['rho']                  # Density/pressure
nx = result['metadata']['nx']        # Grid dimensions
re = result['metadata']['reynolds_number']
```

### Saving Results

```python
import numpy as np

# Save as NumPy binary
np.save('result.npy', result)

# Load results
loaded = np.load('result.npy', allow_pickle=True).item()

# Or save individual fields
np.save('ux.npy', result['ux'])
np.save('uy.npy', result['uy'])
np.save('rho.npy', result['rho'])
```

## Complete Example Script

See `visualization_examples.py` for a complete working example demonstrating:
- Environment visualization
- Live plotting
- Video recording
- Multiple field visualizations
- Custom field visualizer

Run it with:
```bash
python visualization_examples.py
```

## Troubleshooting

### "Visualizer not updating"
Make sure you're passing the `visualizer` argument to `solve()`:
```python
# Wrong:
result = solver.solve(verbose=True)

# Correct:
result = solver.solve(verbose=True, visualizer=visualizer)
```

### "Video file is empty or corrupted"
1. Check FFMpeg is installed correctly
2. Ensure disk space available
3. Try recording with fewer frames (reduce simulation steps)
4. Check output directory permissions

### "Live plotting is slow"
1. Disable interactive mode for offline processing
2. Reduce plotting frequency
3. Use smaller grid for visualization
4. Disable colorbar updates (modify FlowFieldPlotter if needed)

### "Memory usage is high"
1. Don't store all frames in memory during video recording
2. Process results in batches
3. Close figures after use: `plotter.close()`

## See Also

- `VISUALIZATION_README.md` - Detailed API documentation
- `visualization_examples.py` - Runnable examples
- Demos/lbm_karman.py - Reference implementation

