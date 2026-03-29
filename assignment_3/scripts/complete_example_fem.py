#!/usr/bin/env python
"""
Comprehensive Example: FEM Solver with Full Visualization Pipeline

This script mirrors the FDM/LBM example structure:
1. Visualize environment setup
2. Run FEM simulation with optional live visualization
3. Generate final field plots
4. Print summary statistics
"""

import sys
sys.path.insert(0, "../../src")

import os
import numpy as np

from envirmonment import KarmannVortex
from solvers import FEMSolver
from visualization import (
    VelocityMagnitudeVisualizer,
    VorticityVisualizer,
    PressureVisualizer,
    EnvironmentVisualizer,
    FlowFieldPlotter,
)


def main():
    print("\n" + "=" * 70)
    print("FEM Solver - Complete Visualization Pipeline")
    print("=" * 70)

    output_dir = "assignment_3/outputs_fem"
    os.makedirs(output_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # User settings
    # ------------------------------------------------------------------
    inlet_velocity = 0.12
    reynolds_number = 500.0
    fallback_viscosity = 0.001

    dt = 1e-3
    n_steps = 12000
    global_maxh = 0.02
    cyl_maxh = 0.0025
    order = 2

    export_nx = 300
    export_ny = 120

    record_video = False
    # visualize_every = 500
    video_filename = os.path.join(output_dir, "fem_simulation.mp4")

    # ------------------------------------------------------------------
    # Step 1: environment visualization
    # ------------------------------------------------------------------
    print("\n[1/4] Visualizing environment setup...")

    env = KarmannVortex(
        viscosity=fallback_viscosity,
        v0=inlet_velocity,
    )

    print(f"      Environment: KarmannVortex")
    print(f"      Domain: {env.x_range} × {env.y_range}")
    print(f"      Obstacle: center={env.circle_center}, radius={env.circle_radius}")
    print(f"      Inlet velocity: {env.v0}")
    print(f"      Target Reynolds number: {reynolds_number}")

    env_vis = EnvironmentVisualizer(env, nx=export_nx, ny=export_ny)
    fig, ax = env_vis.plot_environment(show_initial_conditions=True)
    env_path = os.path.join(output_dir, "01_environment_setup.png")
    fig.savefig(env_path, dpi=150, bbox_inches="tight")
    print(f"      Saved: {env_path}")

    # ------------------------------------------------------------------
    # Step 2: run FEM simulation
    # ------------------------------------------------------------------
    print("\n[2/4] Running FEM simulation with live visualization...")

    solver = FEMSolver(
        environment=env,
        dt=dt,
        n_steps=n_steps,
        global_maxh=global_maxh,
        cyl_maxh=cyl_maxh,
        order=2,
        reynolds_number=reynolds_number,          # or 250.0 if you want to compare later
        graddiv_gamma=1e-3,
        inlet_profile="parabolic",
        inlet_perturbation=1e-3,
        ramp_time=0.0,                  # use Stokes start instead
        stokes_start=True,
        curved_order=3,
        probe_point=(0.6, 0.21),
        num_threads=8,                  # adjust to your machine
        inverse_name="cg",
        preconditioner_name="amg",
        verbose=True,)

    print("      Solver configuration:")
    print(f"        dt: {dt}")
    print(f"        n_steps: {n_steps}")
    print(f"        global_maxh: {global_maxh}")
    print(f"        cyl_maxh: {cyl_maxh}")
    print(f"        order: {order}")

    live_visualizer = VelocityMagnitudeVisualizer(u_inlet=inlet_velocity, cmap="viridis")

    result = solver.solve(
        verbose=True,
        # visualizer=live_visualizer,
        # record_video=record_video,
        # video_filename=video_filename,
        # visualize_every=visualize_every,
        return_history=True,
        export_nx=export_nx,
        export_ny=export_ny,
    )
    

    ux = result["ux"]
    uy = result["uy"]
    p = result["p"]
    obstacle = result["obstacle"]
    fluid = result["fluid"]
    metadata = result["metadata"]

    print(f"      Simulation complete!")
    print(f"      Final ux range: [{np.nanmin(ux):.6f}, {np.nanmax(ux):.6f}]")
    print(f"      Final uy range: [{np.nanmin(uy):.6f}, {np.nanmax(uy):.6f}]")
    print(f"      Final p range:  [{np.nanmin(p):.6f}, {np.nanmax(p):.6f}]")

    # ------------------------------------------------------------------
    # Step 3: final field visualization
    # ------------------------------------------------------------------
    print("\n[3/4] Generating field visualizations...")

    plotter = FlowFieldPlotter(
        metadata["nx"],
        metadata["ny"],
        obstacle=obstacle,
        figsize=(22, 4),
        dpi=100,
    )

    vel_viz = VelocityMagnitudeVisualizer(u_inlet=metadata["u_inlet"])
    vel_field = vel_viz.compute_field(ux, uy)
    plotter.plot_field(vel_field, vel_viz, step=metadata["n_steps"])
    vel_path = os.path.join(output_dir, "02_final_velocity.png")
    plotter.save(vel_path)

    vor_viz = VorticityVisualizer(cmap="RdBu_r")
    vor_field = vor_viz.compute_field(np.nan_to_num(ux), np.nan_to_num(uy))
    plotter.plot_field(vor_field, vor_viz, step=metadata["n_steps"])
    vor_path = os.path.join(output_dir, "03_final_vorticity.png")
    plotter.save(vor_path)

    pres_viz = PressureVisualizer(cmap="coolwarm")
    pres_field = pres_viz.compute_field(np.nan_to_num(ux), np.nan_to_num(uy), np.nan_to_num(p))
    plotter.plot_field(pres_field, pres_viz, step=metadata["n_steps"])
    pres_path = os.path.join(output_dir, "04_final_pressure.png")
    plotter.save(pres_path)

    plotter.close()

    # ------------------------------------------------------------------
    # Step 4: analysis
    # ------------------------------------------------------------------
    print("\n[4/4] Analyzing results...")

    speed = np.sqrt(ux**2 + uy**2)
    speed_fluid = speed[fluid]

    vorticity = (
        (np.roll(np.nan_to_num(uy), -1, axis=0) - np.roll(np.nan_to_num(uy), 1, axis=0)) / (2 * (env.x_range[1] - env.x_range[0]) / (export_nx - 1))
        - (np.roll(np.nan_to_num(ux), -1, axis=1) - np.roll(np.nan_to_num(ux), 1, axis=1)) / (2 * (env.y_range[1] - env.y_range[0]) / (export_ny - 1))
    )
    vor_fluid = vorticity[fluid]
    p_fluid = p[fluid]

    print("\n      Velocity statistics:")
    print(f"        Mean speed: {np.nanmean(speed_fluid):.6f}")
    print(f"        Max speed:  {np.nanmax(speed_fluid):.6f}")
    print(f"        Min speed:  {np.nanmin(speed_fluid):.6f}")
    print(f"        Std dev:    {np.nanstd(speed_fluid):.6f}")

    print("\n      Vorticity statistics:")
    print(f"        Mean:    {np.nanmean(vor_fluid):.6f}")
    print(f"        Max:     {np.nanmax(vor_fluid):.6f}")
    print(f"        Min:     {np.nanmin(vor_fluid):.6f}")
    print(f"        Std dev: {np.nanstd(vor_fluid):.6f}")

    print("\n      Pressure statistics:")
    print(f"        Mean:    {np.nanmean(p_fluid):.6f}")
    print(f"        Max:     {np.nanmax(p_fluid):.6f}")
    print(f"        Min:     {np.nanmin(p_fluid):.6f}")
    print(f"        Std dev: {np.nanstd(p_fluid):.6f}")

    if "history" in result:
        hist = result["history"]
        print("\n      Time integration diagnostics:")
        print(f"        Final simulated time: {metadata['time_final']:.6f}")
        print(f"        Final |u|max:         {hist['u_max_sampled'][-1]:.6f}")
        print(f"        Final div L2:         {hist['div_l2'][-1]:.6e}")

    np.savez(
        os.path.join(output_dir, "fem_history.npz"),
        time=np.asarray(result["history"]["time"]),
        u_max=np.asarray(result["history"]["u_max_sampled"]),
        div_l2=np.asarray(result["history"]["div_l2"]),
    )

    print("\n" + "=" * 70)
    print("Complete FEM Visualization Pipeline Summary")
    print("=" * 70)
    print("\nGenerated files:")
    print(f"  1. {env_path}  - Initial setup visualization")
    print(f"  2. {vel_path}  - Final velocity magnitude")
    print(f"  3. {vor_path}  - Final vorticity field")
    print(f"  4. {pres_path} - Final pressure field")
    print(f"  5. {os.path.join(output_dir, 'fem_history.npz')} - Time history arrays")

    print("\nSimulation metadata:")
    print(f"  Export grid: {metadata['nx']} × {metadata['ny']}")
    print(f"  Mesh elements: {metadata['mesh_elements']}")
    print(f"  Velocity DOFs: {metadata['velocity_dofs']}")
    print(f"  Pressure DOFs: {metadata['pressure_dofs']}")
    print(f"  Reynolds number: {metadata['reynolds_number']:.3f}")
    print(f"  Viscosity used: {metadata['nu']:.6e}")
    print(f"  Total timesteps: {metadata['n_steps']}")
    print(f"  Final simulated time: {metadata['time_final']:.6f}")
    print(f"  Element type: {metadata['element_type']}")
    print(f"  Time scheme: {metadata['time_scheme']}")
    print(f"  Pressure fix: {metadata['pressure_nullspace_fix']}")
    print(f"  Linear solver: {metadata['linear_solver']}")

    print("\n" + "=" * 70)
    print("Pipeline complete! Check outputs_fem/ directory for generated figures.")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        raise