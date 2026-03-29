# Convergence Study Plotting

Visualization script for convergence study results.

## Usage

```powershell
python assignment_3/scripts/plot_convergence_study.py
```

## Output Plots

- `convergence_error.png` - Relative L2 error vs. grid size (3 panels: ux, uy, rho)
- `runtime_scaling.png` - Runtime vs. ncells with O(n) and O(n log n) reference lines
- `throughput.png` - Cells per second vs. grid size
- `accuracy_vs_cost.png` - Error vs. runtime tradeoff (3 panels)
- `lbm_details.png` - LBM convergence details (error, runtime, throughput, stability)
- `fdm_details.png` - FDM convergence details
- `fem_details.png` - FEM convergence details

## Features

- Automatically loads available CSV files from convergence study
- Skips missing methods with warnings
- Publication-quality plots (300 dpi PNG)
- Log-log scales for convergence rate visualization
- Summary statistics printed to terminal

