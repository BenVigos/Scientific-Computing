# Convergence Study

Grid convergence study for **LBM**, **FDM**, and **FEM** methods.

## Overview

This script runs controlled convergence studies to evaluate:
- **Accuracy**: Error metrics (L2, relative L2) vs. a reference benchmark
- **Performance**: Runtime and throughput (cells/sec) for each grid size
- **Stability**: Finite-value checks

## Architecture

### Benchmark Run
- **Single high-resolution LBM simulation** on a fixed coarse grid
- Cached on first run; reused for all method sweeps
- Acts as reference for error computation

### Convergence Sweeps
- **LBM sweep**: Multiple grid sizes at fixed Re and time steps
- **FDM sweep**: Multiple grid sizes at fixed Re and time steps
- **FEM sweep**: Multiple mesh resolutions (controlled by `global_maxh`, `cyl_maxh`)

Each method runs **independently**; you select which to run via config flags.

## Configuration

Edit `build_run_config()` in the script:

```python
{
    # General
    "u_inlet": 0.12,
    "convergence_reynolds": 200.0,
    "force_recompute_benchmark": False,

    # Benchmark LBM (coarse grid reference)
    "benchmark_nx": 200,
    "benchmark_ny": 80,
    "benchmark_steps": 5000,

    # LBM convergence sweep (enable/disable and configure)
    "run_lbm_convergence": True,
    "lbm_grid_sizes": [(100, 40), (150, 60), (200, 80), (250, 100)],
    "lbm_convergence_steps": 4000,

    # FDM convergence sweep (enable/disable and configure)
    "run_fdm_convergence": False,
    "fdm_grid_sizes": [(151, 60), (201, 80), (301, 120), (401, 160)],
    "fdm_base_dt": 5e-4,
    "fdm_convergence_steps": 3000,

    # FEM convergence sweep (enable/disable and configure)
    "run_fem_convergence": False,
    "fem_mesh_resolutions": [
        (global_maxh, cyl_maxh, export_nx, export_ny),
        ...
    ],
    "fem_base_dt": 1e-3,
    "fem_convergence_steps": 3000,

    # Quick mode for debugging
    "quick_mode": False,
}
```

### Enable/Disable Methods

```python
"run_lbm_convergence": True,    # Enable LBM sweep
"run_fdm_convergence": False,   # Disable FDM sweep
"run_fem_convergence": False,   # Disable FEM sweep
```

### Quick Mode

Set `"quick_mode": True` to run tiny sweeps for debugging (coarse grids, few steps).

## Output

Results are stored in separate method-specific CSV files with a common schema.

### File Structure

```
assignment_3/data/convergence_study/
├── benchmark/
│   ├── lbm_benchmark.npz
│   └── lbm_benchmark_config.json
├── lbm_convergence/
│   └── results.csv
├── fdm_convergence/
│   └── results.csv
└── fem_convergence/
    └── results.csv
```

### CSV Schema

All convergence results follow this schema:

| Column | Type | Description |
|--------|------|-------------|
| `grid_index` or `mesh_index` | int | Sequential index in sweep |
| `nx`, `ny` | int | Grid resolution (LBM/FDM) |
| `global_maxh`, `cyl_maxh` | float | Mesh resolution (FEM) |
| `ncells` | int | Total number of grid cells |
| `runtime_sec` | float | Wall-clock runtime in seconds |
| `cells_per_sec` | float | Throughput (cells / second) |
| `stable` | bool | Whether all fields are finite-valued |
| `l2_ux`, `l2_uy`, `l2_rho` | float | Absolute L2 error vs. benchmark |
| `rel_l2_ux`, `rel_l2_uy`, `rel_l2_rho` | float | Relative L2 error vs. benchmark |

## Usage

### Run Full Convergence Study (LBM only)

```powershell
python assignment_3/scripts/convergence_study.py
```

### Run LBM + FDM Convergence

Edit `build_run_config()`:

```python
"run_lbm_convergence": True,
"run_fdm_convergence": True,
"run_fem_convergence": False,
```

Then run:

```powershell
python assignment_3/scripts/convergence_study.py
```

### Quick Debug Run (All Methods)

Edit `build_run_config()`:

```python
"quick_mode": True,
"run_lbm_convergence": True,
"run_fdm_convergence": True,
"run_fem_convergence": True,
```

Then run:

```powershell
python assignment_3/scripts/convergence_study.py
```

## Data Analysis

After running, analyze results:

```python
import pandas as pd

# Load LBM convergence data
lbm_df = pd.read_csv("assignment_3/data/convergence_study/lbm_convergence/results.csv")

# Convergence order (how error decreases with grid refinement)
lbm_df[["ncells", "rel_l2_ux", "runtime_sec"]]

# FDM results
fdm_df = pd.read_csv("assignment_3/data/convergence_study/fdm_convergence/results.csv")
```

## Notes

- **Benchmark caching**: The benchmark is computed once and cached. Delete `benchmark/lbm_benchmark.npz` to force recompute.
- **Error computation**: All errors are computed on resized grids to match reference resolution, using bilinear interpolation.
- **Independence**: Methods are independent; run only the ones you need.
- **FEM mesh refinement**: Controlled via `global_maxh` (global element size) and `cyl_maxh` (cylinder mesh size).

