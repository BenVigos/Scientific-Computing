#!/usr/bin/env python
"""
Example script demonstrating LBM solver with comprehensive visualization.

Shows how to:
  1. Visualize the environment setup
  2. Run simulation with live plotting
  3. Record simulation to video
  4. Save individual frames
"""

import sys
sys.path.insert(0, 'src')

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


def example_environment_visualization():
    """Example 1: Visualize the environment setup."""
    print("\n" + "="*70)
    print("Example 1: Environment Visualization")
    print("="*70)

    # Create environment
    environment = KarmannVortex(v0=1.0)

    # Visualize environment
    env_vis = EnvironmentVisualizer(environment, nx=150, ny=60)
    fig, ax = env_vis.plot_environment(show_initial_conditions=True)

    print("\nEnvironment visualized. Figure displayed.")
    fig.savefig('outputs/environment_setup.png', dpi=150, bbox_inches='tight')
    print("Saved to: outputs/environment_setup.png")


def example_quick_run():
    """Example 2: Quick simulation run with live velocity visualization."""
    print("\n" + "="*70)
    print("Example 2: Quick Simulation with Live Visualization")
    print("="*70)

    # Create environment and solver
    environment = KarmannVortex(v0=1.0)
    solver = LBMSolver(
        environment=environment,
        nx=150,
        ny=60,
        u_inlet=0.12,
        reynolds_number=15,
        n_steps=1000  # Short run
    )

    # Create visualizer for velocity magnitude
    visualizer = VelocityMagnitudeVisualizer(u_inlet=0.12, cmap='viridis')

    # Run with live plotting (no video recording)
    print("\nRunning simulation with live velocity visualization...")
    result = solver.solve(verbose=True, visualizer=visualizer, record_video=False)

    print(f"\nFinal velocity range: [{result['ux'].min():.6f}, {result['ux'].max():.6f}]")
    print(f"Final density range: [{result['rho'].min():.6f}, {result['rho'].max():.6f}]")


def example_with_video():
    """Example 3: Simulation with video recording."""
    print("\n" + "="*70)
    print("Example 3: Simulation with Video Recording")
    print("="*70)

    environment = KarmannVortex(v0=1.0)
    solver = LBMSolver(
        environment=environment,
        nx=150,
        ny=60,
        u_inlet=0.12,
        reynolds_number=15,
        n_steps=2000
    )

    # Visualize vorticity this time
    visualizer = VorticityVisualizer(cmap='RdBu_r')

    print("\nRunning simulation with vorticity visualization and video recording...")
    print("This may take a while depending on FFMpeg availability...")

    try:
        result = solver.solve(
            verbose=True,
            visualizer=visualizer,
            record_video=True,
            video_filename='outputs/lbm_vorticity.mp4'
        )
        print("\nVideo saved to: outputs/lbm_vorticity.mp4")
    except Exception as e:
        print(f"\nVideo recording failed: {e}")
        print("Continuing without video...")
        result = solver.solve(verbose=True, visualizer=visualizer)


def example_multiple_field_visualizations():
    """Example 4: Save snapshots of different field types."""
    print("\n" + "="*70)
    print("Example 4: Multiple Field Visualizations")
    print("="*70)

    environment = KarmannVortex(v0=1.0)
    solver = LBMSolver(
        environment=environment,
        nx=150,
        ny=60,
        u_inlet=0.12,
        reynolds_number=15,
        n_steps=5000
    )

    # Run simulation without visualization first
    print("\nRunning simulation (no visualization)...")
    result = solver.solve(verbose=True, visualizer=None)

    ux = result['ux']
    uy = result['uy']
    rho = result['rho']
    obstacle = result['obstacle']

    # Now visualize final state with different fields
    print("\nGenerating field visualizations...")

    plotter = FlowFieldPlotter(
        result['metadata']['nx'],
        result['metadata']['ny'],
        obstacle=obstacle,
        figsize=(14, 6)
    )

    # Velocity magnitude
    print("  - Velocity magnitude...")
    vel_viz = VelocityMagnitudeVisualizer(u_inlet=0.12)
    vel_field = vel_viz.compute_field(ux, uy)
    plotter.plot_field(vel_field, vel_viz)
    plotter.save('outputs/final_velocity.png')

    # Vorticity
    print("  - Vorticity...")
    vor_viz = VorticityVisualizer()
    vor_field = vor_viz.compute_field(ux, uy)
    plotter.plot_field(vor_field, vor_viz)
    plotter.save('outputs/final_vorticity.png')

    # Pressure (density)
    print("  - Pressure (density)...")
    pres_viz = PressureVisualizer()
    pres_field = pres_viz.compute_field(ux, uy, rho)
    plotter.plot_field(pres_field, pres_viz)
    plotter.save('outputs/final_pressure.png')

    plotter.close()
    print("\nVisualization files saved to outputs/")


def example_custom_field_visualizer():
    """Example 5: Create and use a custom field visualizer."""
    print("\n" + "="*70)
    print("Example 5: Custom Field Visualizer")
    print("="*70)

    from visualization import FieldVisualizer

    class StreamfunctionVisualizer(FieldVisualizer):
        """Custom visualizer for streamfunction field."""

        def compute_field(self, ux, uy, rho=None):
            """Compute streamfunction using cumulative integration."""
            # Simple approximation: cumulative y-momentum
            return np.cumsum(ux, axis=1)

        def get_plot_params(self):
            """Plot parameters for streamfunction."""
            return {
                'cmap': 'twilight_shifted',
                'vmin': None,
                'vmax': None,
                'label': 'Streamfunction'
            }

        def get_title(self, step=None):
            """Title for streamfunction plot."""
            if step is not None:
                return f"Streamfunction field — step {step}"
            return "Streamfunction field"

    environment = KarmannVortex(v0=1.0)
    solver = LBMSolver(
        environment=environment,
        nx=150,
        ny=60,
        u_inlet=0.12,
        reynolds_number=15,
        n_steps=1500
    )

    visualizer = StreamfunctionVisualizer()

    print("\nRunning simulation with custom streamfunction visualizer...")
    result = solver.solve(verbose=True, visualizer=visualizer)

    print("\nCustom visualizer example complete!")


def main():
    """Run all examples."""
    print("\n" + "="*70)
    print("LBM Solver Visualization Examples")
    print("="*70)

    # Create outputs directory if it doesn't exist
    import os
    os.makedirs('outputs', exist_ok=True)

    # Run examples
    try:
        example_environment_visualization()
    except Exception as e:
        print(f"Environment visualization failed: {e}")

    try:
        example_quick_run()
    except Exception as e:
        print(f"Quick run failed: {e}")

    try:
        example_with_video()
    except Exception as e:
        print(f"Video example failed: {e}")

    try:
        example_multiple_field_visualizations()
    except Exception as e:
        print(f"Multiple field visualization failed: {e}")

    try:
        example_custom_field_visualizer()
    except Exception as e:
        print(f"Custom visualizer example failed: {e}")

    print("\n" + "="*70)
    print("All examples completed!")
    print("="*70)


if __name__ == "__main__":
    main()

