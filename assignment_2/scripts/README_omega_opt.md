# DLA Omega Optimization Study

This directory contains scripts for analyzing the effect of the SOR over-relaxation parameter (omega) on the computational efficiency of Diffusion-Limited Aggregation (DLA) simulations.

## Files

### `dla_omega_opt.py`
Main script that performs a parameter sweep over:
- **Omega (ω)**: SOR over-relaxation parameter (1.0 to 1.9 in steps of 0.1)
- **Eta (η)**: Sticking probability exponent (0.0, 1.0, 2.0)
- **Seeds**: Multiple random seeds (10 per combination) for statistical significance

The script tracks the number of SOR iterations required for convergence at each growth step and generates comprehensive analysis plots.

**Output:**
- `dla_omega_convergence.csv`: CSV file with detailed statistics for each simulation
- `figures/omega_optimization/`: Directory containing analysis plots:
  - `omega_vs_mean_iterations.png`: Mean SOR iterations per growth step vs omega
  - `omega_vs_total_iterations.png`: Total computational cost vs omega
  - `omega_ita_heatmap.png`: Heatmap showing mean iterations for all omega-eta combinations
  - `omega_boxplots.png`: Distribution of iterations across seeds
  - `omega_vs_stability.png`: Convergence stability analysis
- `optimal_omega_summary.txt`: Text file with recommendations for optimal omega values

### `dla_omega_opt_quick_test.py`
Quick test version with fewer combinations (2 omega × 2 eta × 2 seeds) to verify functionality.

## Running the Analysis

### Full Analysis
```bash
python assignment_2/scripts/dla_omega_opt.py
```
**Note:** This will run 300 simulations and may take 30-60 minutes depending on your system.

### Quick Test
```bash
python assignment_2/scripts/dla_omega_opt_quick_test.py
```
Runs in a few minutes to verify the code is working correctly.

## Configuration

You can modify the parameters at the bottom of `dla_omega_opt.py`:

```python
# Configuration
grid_size = (100, 100)          # Lattice size
steps = 1000                    # Maximum growth steps
stop_threshold = 0.1            # Stop when this fraction of grid is occupied

# Parameter ranges
omega_values = np.arange(1.0, 2.0, 0.1)  # Omega sweep range
ita_values = [0.0, 1.0, 2.0]             # Eta values to test
seeds = list(range(10))                  # Number of seeds per combination
```

## Results Interpretation

The analysis helps answer:
1. **Which omega value minimizes SOR iterations?** - Important for computational efficiency
2. **How does omega affect convergence for different eta values?** - Some eta values may benefit from different omega
3. **Is convergence stable across different random seeds?** - Ensures results are robust

## CSV Output Format

The `dla_omega_convergence.csv` file contains:
- `omega`: Over-relaxation parameter used
- `ita`: Sticking probability exponent used
- `seed`: Random seed
- `n_particles`: Number of particles added before stopping
- `mean_sor_iter`: Average SOR iterations per growth step
- `median_sor_iter`: Median SOR iterations per growth step
- `std_sor_iter`: Standard deviation of SOR iterations
- `max_sor_iter`: Maximum SOR iterations in any single step
- `min_sor_iter`: Minimum SOR iterations in any single step
- `total_sor_iter`: Total SOR iterations for entire simulation

## Dependencies

- numpy
- pandas
- matplotlib
- seaborn
- tqdm
- numba (used by sor_numba)

