#!/usr/bin/env python
"""
Comprehensive Example: LBM Solver with headless video recording.

This script demonstrates:
1. Environment setup image export
2. Headless simulation recording of velocity video
3. Headless simulation recording of vorticity video
4. Final-state image export and basic statistics
"""

import sys
sys.path.insert(0, 'src')

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Disable GUI/live windows while still allowing figure/video output

from solvers import LBMSolver
from envirmonment import KarmannVortex
from visualization import (
    VelocityMagnitudeVisualizer,
    VorticityVisualizer,
    EnvironmentVisualizer,
    FlowFieldPlotter,
)


def _build_solver(env, scaling=5):
    """Create a solver instance with shared parameters for reproducible runs."""
    return LBMSolver(
        environment=env,
        nx=int(220 * scaling),
        ny=int(41 * scaling),
        u_inlet=0.12,
        reynolds_number=1300,
        n_steps=12000,
        vis_interval=100,
        velocity_ramp_tau=500,
        inlet_bc='regularized',
        outlet_bc='open',
        alpha=1,
        collision_model='bgk',
        bc_ramp_tau=10,
        outlet_sponge_width=100 * scaling / 4,
        outlet_sponge_sigma_max=0.3,
    )


def _run_and_record(env, visualizer, video_path, scaling=5, verbose=True, video_fps=12):
    """Run one headless simulation pass and record a video for the provided field visualizer."""
    solver = _build_solver(env, scaling=scaling)
    result = solver.solve(
        verbose=verbose,
        visualizer=visualizer,
        record_video=True,
        video_filename=video_path,
        video_fps=video_fps,
    )
    return result


def main():
    """Run complete LBM simulation with headless video recording and final-state images."""

    print("\n" + "=" * 70)
    print("LBM Solver - Headless Recording Pipeline")
    print("=" * 70)

    # Create output directory
    os.makedirs('assignment_3/outputs', exist_ok=True)

    # =====================================================================
    # Step 1: Environment setup image
    # =====================================================================
    print("\n[1/4] Saving environment setup image...")

    env = KarmannVortex(v0=0.12)
    env_vis = EnvironmentVisualizer(env, nx=300, ny=120)
    fig, _ = env_vis.plot_environment(show_initial_conditions=True)
    fig.savefig('assignment_3/outputs/01_environment_setup.png', dpi=150, bbox_inches='tight')
    print("      Saved: assignment_3/outputs/01_environment_setup.png")

    # =====================================================================
    # Step 2: Velocity video (headless)
    # =====================================================================
    print("\n[2/4] Recording velocity video (no live visualization)...")
    vel_viz = VelocityMagnitudeVisualizer(u_inlet=0.12, cmap='viridis')
    vel_video = 'assignment_3/outputs/02_velocity.mp4'
    result = _run_and_record(env, vel_viz, vel_video, scaling=5, verbose=True, video_fps=12)
    print(f"      Saved: {vel_video}")

    # =====================================================================
    # Step 3: Vorticity video (headless)
    # =====================================================================
    print("\n[3/4] Recording vorticity video (no live visualization)...")
    vor_viz = VorticityVisualizer(cmap='RdBu_r')
    vor_video = 'assignment_3/outputs/03_vorticity.mp4'
    _ = _run_and_record(env, vor_viz, vor_video, scaling=5, verbose=False, video_fps=12)
    print(f"      Saved: {vor_video}")

    # =====================================================================
    # Step 4: Final-state images + stats
    # =====================================================================
    print("\n[4/4] Saving final-state images and statistics...")

    ux = result['ux']
    uy = result['uy']
    rho = result['rho']
    obstacle = result['obstacle']
    metadata = result['metadata']

    plotter = FlowFieldPlotter(
        metadata['nx'],
        metadata['ny'],
        obstacle=obstacle,
        figsize=(22, 4),
        dpi=120,
    )

    vel_field = vel_viz.compute_field(ux, uy)
    plotter.plot_field(vel_field, vel_viz, step=metadata['n_steps'])
    plotter.save('assignment_3/outputs/04_final_velocity.png')

    vor_field = vor_viz.compute_field(ux, uy)
    plotter.plot_field(vor_field, vor_viz, step=metadata['n_steps'])
    plotter.save('assignment_3/outputs/05_final_vorticity.png')

    plotter.close()

    speed = np.sqrt(ux**2 + uy**2)
    speed_fluid = speed[~obstacle]
    rho_fluid = rho[~obstacle]

    print(f"      Mean speed: {speed_fluid.mean():.6f}")
    print(f"      Max speed: {speed_fluid.max():.6f}")
    print(f"      Mean density: {rho_fluid.mean():.6f}")

    print("\nGenerated files:")
    print("  - assignment_3/outputs/01_environment_setup.png")
    print("  - assignment_3/outputs/02_velocity.mp4")
    print("  - assignment_3/outputs/03_vorticity.mp4")
    print("  - assignment_3/outputs/04_final_velocity.png")
    print("  - assignment_3/outputs/05_final_vorticity.png")

    print("\n" + "=" * 70)
    print("Headless recording pipeline complete.")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        exit(1)

