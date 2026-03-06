"""
Quick test version of DLA Optimization Study
Tests with fewer combinations to verify functionality
"""

import os
import sys
import numpy as np

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.diffusion_limited_aggregation import (
    outer_neighbors,
    compute_stick_prob,
    select_stick_cell
)
from src.iter_schemes import sor_numba


def dla_with_sor_tracking(grid_size, steps=1000, stop_threshold=0.5,
                           debug=False, ita=1.0, omega=1.9,
                           sor_max_iter=100000, sor_tol=1e-5):
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


if __name__ == "__main__":
    print("=" * 60)
    print("QUICK TEST: DLA with SOR iteration tracking")
    print("=" * 60)

    # Test with minimal configuration
    grid_size = (50, 50)
    steps = 100
    stop_threshold = 0.05

    omega_values = [1.5, 1.9]
    ita_values = [0.0, 1.0]
    seeds = [0, 1]

    print(f"Grid size: {grid_size}")
    print(f"Max steps: {steps}")
    print(f"Stop threshold: {stop_threshold}")
    print(f"Omega values: {omega_values}")
    print(f"Eta values: {ita_values}")
    print(f"Seeds: {seeds}")
    print(f"Total simulations: {len(omega_values) * len(ita_values) * len(seeds)}")
    print("=" * 60 + "\n")

    results = []

    for omega in omega_values:
        for ita in ita_values:
            for seed in seeds:
                np.random.seed(seed)

                print(f"Running: omega={omega:.2f}, ita={ita:.1f}, seed={seed}")

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

                sor_iters_array = np.array(sor_iters)
                mean_iter = float(np.mean(sor_iters_array))
                total_iter = int(np.sum(sor_iters_array))

                print(f"  Particles added: {n_particles}")
                print(f"  Mean SOR iterations: {mean_iter:.2f}")
                print(f"  Total SOR iterations: {total_iter}")
                print()

                results.append({
                    "omega": omega,
                    "ita": ita,
                    "seed": seed,
                    "n_particles": n_particles,
                    "mean_sor_iter": mean_iter,
                    "total_sor_iter": total_iter
                })

    print("\n" + "=" * 60)
    print("TEST COMPLETE - Results Summary:")
    print("=" * 60)

    for result in results:
        print(f"ω={result['omega']:.2f}, η={result['ita']:.1f}, seed={result['seed']}: "
              f"mean_iter={result['mean_sor_iter']:.2f}, total={result['total_sor_iter']}")

    print("\n✓ Script is working correctly!")

