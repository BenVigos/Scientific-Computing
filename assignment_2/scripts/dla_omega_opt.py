"""
DLA Optimization Study: Omega Parameter Sweep

This script runs diffusion-limited aggregation with different:
- Seeds (for statistical significance)
- Eta values (sticking probability exponent)
- Omega values (SOR over-relaxation parameter)

Records the number of SOR iterations required for convergence at each step
and generates plots to analyze the effect of omega on computational efficiency.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import seaborn as sns

# Add parent directory to path to import from src
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

# Import DLA components
from src.diffusion_limited_aggregation import (
    outer_neighbors,
    compute_stick_prob,
    select_stick_cell
)
from src.iter_schemes import sor_numba


def dla_with_sor_tracking(grid_size, steps=1000, stop_threshold=0.5,
                           debug=False, ita=1.0, omega=1.9,
                           sor_max_iter=500, sor_tol=1e-5):
    """
    Run DLA simulation and track SOR iterations at each growth step.

    Returns:
        grid: final cluster configuration
        sor_iterations: list of iteration counts for each growth step
        total_particles: number of particles actually added
    """
    # Initialize grid
    grid = np.zeros(grid_size, dtype=int)
    diffusion_grid = np.zeros(grid_size, dtype=float)
    diffusion_grid[0, :] = 1  # top boundary concentration = 1

    # Place seed particle at bottom center
    seed_x = grid_size[1] // 2
    seed_y = grid_size[0] - 1
    grid[seed_y, seed_x] = 1

    sor_iterations = []

    for i in range(steps):
        if debug and i % 100 == 0:
            print(f"Step {i+1}/{steps}")

        # Prepare for SOR solve
        insulator_mask = np.zeros(grid.shape, dtype=np.bool_)

        # Solve Laplace equation with SOR
        diffusion_grid, iter_count, converged = sor_numba(
            len(grid),
            omega=omega,
            c=diffusion_grid,
            sink=grid,
            insulator=insulator_mask,
            max_iter=sor_max_iter,
            tol=sor_tol
        )

        sor_iterations.append(iter_count)

        if not converged and debug:
            print(f"  Warning: SOR did not converge at step {i+1}")

        # Compute sticking probabilities
        neighbours = outer_neighbors(grid)
        neighbour_concentrations = neighbours * diffusion_grid
        probabilities = compute_stick_prob(neighbour_concentrations, neighbours, ita=ita)

        # Select and add particle
        selection = select_stick_cell(probabilities)
        if selection is None:
            if debug:
                print(f"No valid cells at step {i+1}. Stopping.")
            break

        grid[selection] = 1

        # Check stopping condition
        occupied_percentage = np.mean(grid)
        if occupied_percentage >= stop_threshold:
            if debug:
                print(f"Reached stop threshold at step {i+1}")
            break

    return grid, sor_iterations, i+1


def run_omega_sweep(omega_values, ita_values, seeds,
                    grid_size=(100, 100), steps=1000,
                    stop_threshold=0.1, out_csv="dla_omega_convergence.csv"):
    """
    Run DLA for multiple omega, ita, and seed combinations.
    Track SOR convergence statistics.
    """
    results = []

    total_runs = len(omega_values) * len(ita_values) * len(seeds)
    pbar = tqdm(total=total_runs, desc="Running DLA omega sweep")

    for omega in omega_values:
        for ita in ita_values:
            for seed in seeds:
                np.random.seed(seed)

                grid, sor_iters, n_particles = dla_with_sor_tracking(
                    grid_size=grid_size,
                    steps=steps,
                    stop_threshold=stop_threshold,
                    debug=False,
                    ita=ita,
                    omega=omega,
                    sor_max_iter=100000,
                    sor_tol=1e-5
                )

                # Compute statistics
                sor_iters_array = np.array(sor_iters)
                mean_iter = float(np.mean(sor_iters_array))
                median_iter = float(np.median(sor_iters_array))
                std_iter = float(np.std(sor_iters_array))
                max_iter = int(np.max(sor_iters_array))
                min_iter = int(np.min(sor_iters_array))
                total_iter = int(np.sum(sor_iters_array))

                row = {
                    "omega": omega,
                    "ita": ita,
                    "seed": seed,
                    "n_particles": n_particles,
                    "mean_sor_iter": mean_iter,
                    "median_sor_iter": median_iter,
                    "std_sor_iter": std_iter,
                    "max_sor_iter": max_iter,
                    "min_sor_iter": min_iter,
                    "total_sor_iter": total_iter
                }
                results.append(row)
                pbar.update(1)

    pbar.close()

    df = pd.DataFrame(results)
    df.to_csv(out_csv, index=False)
    print(f"\nSaved results to: {os.path.abspath(out_csv)}")
    return df


def plot_omega_results(df, out_dir):
    """
    Generate plots analyzing the effect of omega on SOR convergence.
    """
    os.makedirs(out_dir, exist_ok=True)

    # Set style
    sns.set_style("whitegrid")
    plt.rcParams.update({'font.size': 10})

    # 1. Mean SOR iterations vs omega (for each ita)
    fig, ax = plt.subplots(figsize=(10, 6))

    for ita in sorted(df['ita'].unique()):
        subset = df[df['ita'] == ita]
        grouped = subset.groupby('omega')['mean_sor_iter'].agg(['mean', 'std'])

        ax.errorbar(grouped.index, grouped['mean'], yerr=grouped['std'],
                   marker='o', capsize=5, label=f'η = {ita:.1f}')

    ax.set_xlabel('Omega (ω)', fontsize=12)
    ax.set_ylabel('Mean SOR Iterations per Growth Step', fontsize=12)
    ax.set_title('SOR Convergence Speed vs Over-relaxation Parameter', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'omega_vs_mean_iterations.png'), dpi=150)
    plt.close()

    # 2. Total computational cost (total iterations) vs omega
    fig, ax = plt.subplots(figsize=(10, 6))

    for ita in sorted(df['ita'].unique()):
        subset = df[df['ita'] == ita]
        grouped = subset.groupby('omega')['total_sor_iter'].agg(['mean', 'std'])

        ax.errorbar(grouped.index, grouped['mean'], yerr=grouped['std'],
                   marker='s', capsize=5, label=f'η = {ita:.1f}')

    ax.set_xlabel('Omega (ω)', fontsize=12)
    ax.set_ylabel('Total SOR Iterations per Simulation', fontsize=12)
    ax.set_title('Total Computational Cost vs Over-relaxation Parameter', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'omega_vs_total_iterations.png'), dpi=150)
    plt.close()

    # 3. Heatmap: omega vs ita showing mean iterations
    fig, ax = plt.subplots(figsize=(10, 8))

    pivot_mean = df.groupby(['ita', 'omega'])['mean_sor_iter'].mean().unstack()
    sns.heatmap(pivot_mean, annot=True, fmt='.1f', cmap='YlOrRd',
                cbar_kws={'label': 'Mean SOR Iterations'}, ax=ax)

    ax.set_xlabel('Omega (ω)', fontsize=12)
    ax.set_ylabel('Eta (η)', fontsize=12)
    ax.set_title('Mean SOR Iterations: Omega vs Eta', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'omega_ita_heatmap.png'), dpi=150)
    plt.close()

    # 4. Box plots showing distribution across seeds for selected omega values
    # Select a few omega values for comparison
    omega_subset = sorted(df['omega'].unique())
    if len(omega_subset) > 6:
        # Pick evenly spaced values
        indices = np.linspace(0, len(omega_subset)-1, 6, dtype=int)
        omega_subset = [omega_subset[i] for i in indices]

    fig, axes = plt.subplots(1, len(df['ita'].unique()),
                             figsize=(5*len(df['ita'].unique()), 5))

    if len(df['ita'].unique()) == 1:
        axes = [axes]

    for idx, ita in enumerate(sorted(df['ita'].unique())):
        subset = df[(df['ita'] == ita) & (df['omega'].isin(omega_subset))]

        sns.boxplot(data=subset, x='omega', y='mean_sor_iter', ax=axes[idx])
        axes[idx].set_title(f'η = {ita:.1f}', fontsize=12)
        axes[idx].set_xlabel('Omega (ω)', fontsize=11)
        axes[idx].set_ylabel('Mean SOR Iterations', fontsize=11)
        axes[idx].tick_params(axis='x', rotation=45)

    plt.suptitle('SOR Iteration Distribution Across Seeds', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'omega_boxplots.png'), dpi=150)
    plt.close()

    # 5. Standard deviation plot (stability of convergence)
    fig, ax = plt.subplots(figsize=(10, 6))

    for ita in sorted(df['ita'].unique()):
        subset = df[df['ita'] == ita]
        grouped = subset.groupby('omega')['std_sor_iter'].mean()

        ax.plot(grouped.index, grouped.values, marker='d', label=f'η = {ita:.1f}')

    ax.set_xlabel('Omega (ω)', fontsize=12)
    ax.set_ylabel('Average Std Dev of SOR Iterations', fontsize=12)
    ax.set_title('Stability of SOR Convergence vs Omega', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'omega_vs_stability.png'), dpi=150)
    plt.close()

    print(f"\nPlots saved to: {os.path.abspath(out_dir)}")


def find_optimal_omega(df, out_file="optimal_omega_summary.txt"):
    """
    Analyze results to find optimal omega for each eta value.
    """
    summary = []

    for ita in sorted(df['ita'].unique()):
        subset = df[df['ita'] == ita]

        # Find omega with minimum mean total iterations
        best_by_total = subset.groupby('omega')['total_sor_iter'].mean().idxmin()
        min_total = subset.groupby('omega')['total_sor_iter'].mean().min()

        # Find omega with minimum mean iterations per step
        best_by_mean = subset.groupby('omega')['mean_sor_iter'].mean().idxmin()
        min_mean = subset.groupby('omega')['mean_sor_iter'].mean().min()

        summary.append({
            'ita': ita,
            'best_omega_total': best_by_total,
            'min_total_iter': min_total,
            'best_omega_mean': best_by_mean,
            'min_mean_iter': min_mean
        })

    summary_df = pd.DataFrame(summary)

    # Save to file
    with open(out_file, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("OPTIMAL OMEGA VALUES FOR DLA SIMULATION\n")
        f.write("=" * 60 + "\n\n")

        for _, row in summary_df.iterrows():
            f.write(f"η = {row['ita']:.2f}\n")
            f.write(f"  Best ω (by total iterations): {row['best_omega_total']:.2f}\n")
            f.write(f"    → Total iterations: {row['min_total_iter']:.1f}\n")
            f.write(f"  Best ω (by mean per step): {row['best_omega_mean']:.2f}\n")
            f.write(f"    → Mean iterations/step: {row['min_mean_iter']:.2f}\n")
            f.write("\n")

    print(f"\nOptimal omega summary saved to: {os.path.abspath(out_file)}")
    print("\nOptimal Omega Summary:")
    print(summary_df.to_string(index=False))

    return summary_df


if __name__ == "__main__":
    # Configuration
    grid_size = (100, 100)
    steps = 1000
    stop_threshold = 0.1

    # Parameter ranges
    omega_values = np.arange(1.7, 1.95, 0.05)  # Test omega from 1.0 to 1.9
    ita_values = [0.0, 0.5, 1.0, 1.5, 2.0]  # Test a few eta values
    seeds = list(range(5))  # 10 different seeds for statistics

    # Output directory
    out_dir = os.path.join(os.path.dirname(__file__), "..", "data", "dla")
    os.makedirs(out_dir, exist_ok=True)

    out_csv = os.path.join(out_dir, "dla_omega_convergence.csv")

    print("=" * 60)
    print("DLA OMEGA OPTIMIZATION STUDY")
    print("=" * 60)
    print(f"Grid size: {grid_size}")
    print(f"Max steps: {steps}")
    print(f"Stop threshold: {stop_threshold}")
    print(f"Omega values: {len(omega_values)} values from {omega_values[0]:.1f} to {omega_values[-1]:.1f}")
    print(f"Eta values: {ita_values}")
    print(f"Seeds per combination: {len(seeds)}")
    print(f"Total simulations: {len(omega_values) * len(ita_values) * len(seeds)}")
    print("=" * 60 + "\n")

    # Run experiments
    df = run_omega_sweep(
        omega_values=omega_values,
        ita_values=ita_values,
        seeds=seeds,
        grid_size=grid_size,
        steps=steps,
        stop_threshold=stop_threshold,
        out_csv=out_csv
    )

    # Generate plots
    fig_dir = os.path.join(out_dir, "figures", "omega_optimization")
    plot_omega_results(df, fig_dir)

    # Find optimal omega
    summary_file = os.path.join(out_dir, "optimal_omega_summary.txt")
    optimal_summary = find_optimal_omega(df, summary_file)

    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)

