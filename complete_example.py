#!/usr/bin/env python
"""
Comprehensive Example: LBM Solver with Full Visualization Pipeline

This script demonstrates a complete workflow:
1. Visualize environment setup
2. Run simulation with live visualization
3. Analyze results and generate multiple field visualizations
4. Save outputs
"""

import sys
sys.path.insert(0, 'src')

import os
import numpy as np
from solvers import LBMSolver
from envirmonment import KarmannVortex
from visualization import (
    VelocityMagnitudeVisualizer,
    VorticityVisualizer,
    PressureVisualizer,
    EnvironmentVisualizer,
    FlowFieldPlotter
)


def main():
    """Run complete LBM simulation with visualization pipeline."""

    print("\n" + "="*70)
    print("LBM Solver - Complete Visualization Pipeline")
    print("="*70)

    # Create output directory
    os.makedirs('assignment_3/outputs', exist_ok=True)

    # =========================================================================
    # Step 1: Visualize Environment Setup
    # =========================================================================
    print("\n[1/4] Visualizing environment setup...")

    env = KarmannVortex(v0=0.12)
    print(f"      Environment: KarmannVortex")
    print(f"      Domain: {env.x_range} × {env.y_range}")
    print(f"      Obstacle: center={env.circle_center}, radius={env.circle_radius}")

    env_vis = EnvironmentVisualizer(env, nx=300, ny=120)
    fig, ax = env_vis.plot_environment(show_initial_conditions=True)
    fig.savefig('assignment_3/outputs/01_environment_setup.png', dpi=150, bbox_inches='tight')
    print(f"      Saved: outputs/01_environment_setup.png")

    # =========================================================================
    # Step 2: Run Simulation with Live Visualization
    # =========================================================================
    print("\n[2/4] Running LBM simulation with live velocity visualization...")

    scaling =4.0
    # Create solver with realistic parameters
    solver = LBMSolver(
        environment=env,
        nx=int(220*scaling),
        ny=int(41*scaling),
        u_inlet=0.12,
        reynolds_number=450,
        n_steps=10000,  # Full run
        vis_interval=200,
        velocity_ramp_tau=400,
    )

    print(f"      Solver configuration:")
    print(f"        Grid size: {solver.nx} × {solver.ny}")
    print(f"        Inlet velocity: {solver.u_inlet}")
    print(f"        Reynolds number: {solver.reynolds_number}")
    print(f"        Relaxation time (tau): {solver.tau:.6f}")
    print(f"        Total steps: {solver.n_steps}")

    # Run simulation with velocity magnitude visualization
    visualizer = VelocityMagnitudeVisualizer(u_inlet=0.12, cmap='viridis')

    print(f"      Running simulation...")
    result = solver.solve(
        verbose=True,
        visualizer=visualizer,
        record_video=False  # Can enable if FFMpeg available
    )

    print(f"      Simulation complete!")
    print(f"      Final state - ux range: [{result['ux'].min():.6f}, {result['ux'].max():.6f}]")
    print(f"      Final state - uy range: [{result['uy'].min():.6f}, {result['uy'].max():.6f}]")
    print(f"      Final state - rho range: [{result['rho'].min():.6f}, {result['rho'].max():.6f}]")

    # =========================================================================
    # Step 3: Generate Multiple Field Visualizations
    # =========================================================================
    print("\n[3/4] Generating field visualizations...")

    ux = result['ux']
    uy = result['uy']
    rho = result['rho']
    obstacle = result['obstacle']
    metadata = result['metadata']

    # Create plotter for final state visualizations
    plotter = FlowFieldPlotter(
        metadata['nx'],
        metadata['ny'],
        obstacle=obstacle,
        figsize=(14, 6),
        dpi=100
    )

    # Visualization 1: Velocity Magnitude
    print("      Generating velocity magnitude field...")
    vel_viz = VelocityMagnitudeVisualizer(u_inlet=metadata['u_inlet'])
    vel_field = vel_viz.compute_field(ux, uy)
    plotter.plot_field(vel_field, vel_viz, step=metadata['n_steps'])
    plotter.save('assignment_3/outputs/02_final_velocity.png')

    # Visualization 2: Vorticity
    print("      Generating vorticity field...")
    vor_viz = VorticityVisualizer(cmap='RdBu_r')
    vor_field = vor_viz.compute_field(ux, uy)
    plotter.plot_field(vor_field, vor_viz, step=metadata['n_steps'])
    plotter.save('assignment_3/outputs/03_final_vorticity.png')

    # Visualization 3: Pressure (Density)
    print("      Generating pressure field...")
    pres_viz = PressureVisualizer(cmap='coolwarm')
    pres_field = pres_viz.compute_field(ux, uy, rho)
    plotter.plot_field(pres_field, pres_viz, step=metadata['n_steps'])
    plotter.save('assignment_3/outputs/04_final_pressure.png')

    plotter.close()

    # =========================================================================
    # Step 4: Analysis and Statistics
    # =========================================================================
    print("\n[4/4] Analyzing results...")

    # Compute field statistics
    speed = np.sqrt(ux**2 + uy**2)
    speed_fluid = speed[~obstacle]

    vorticity = (np.roll(uy, -1, axis=0) - np.roll(uy, 1, axis=0)
                 - np.roll(ux, -1, axis=1) + np.roll(ux, 1, axis=1))
    vor_fluid = vorticity[~obstacle]

    rho_fluid = rho[~obstacle]

    print(f"\n      Velocity statistics:")
    print(f"        Mean speed: {speed_fluid.mean():.6f}")
    print(f"        Max speed: {speed_fluid.max():.6f}")
    print(f"        Min speed: {speed_fluid.min():.6f}")
    print(f"        Std dev: {speed_fluid.std():.6f}")

    print(f"\n      Vorticity statistics:")
    print(f"        Mean: {vor_fluid.mean():.6f}")
    print(f"        Max: {vor_fluid.max():.6f}")
    print(f"        Min: {vor_fluid.min():.6f}")
    print(f"        Std dev: {vor_fluid.std():.6f}")

    print(f"\n      Density statistics:")
    print(f"        Mean: {rho_fluid.mean():.6f}")
    print(f"        Max: {rho_fluid.max():.6f}")
    print(f"        Min: {rho_fluid.min():.6f}")
    print(f"        Std dev: {rho_fluid.std():.6f}")

    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "="*70)
    print("Complete Visualization Pipeline Summary")
    print("="*70)

    print("\nGenerated files:")
    print("  1. outputs/01_environment_setup.png  - Initial setup visualization")
    print("  2. outputs/02_final_velocity.png     - Final velocity magnitude")
    print("  3. outputs/03_final_vorticity.png    - Final vorticity field")
    print("  4. outputs/04_final_pressure.png     - Final pressure/density")

    print("\nSimulation metadata:")
    print(f"  Grid size: {metadata['nx']} × {metadata['ny']}")
    print(f"  Reynolds number: {metadata['reynolds_number']}")
    print(f"  Total timesteps: {metadata['n_steps']}")
    print(f"  Relaxation time: {metadata['tau']:.6f}")

    print("\n" + "="*70)
    print("Pipeline complete! Check outputs/ directory for generated figures.")
    print("="*70 + "\n")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        exit(1)

