#!/usr/bin/env python
"""
Comprehensive Example: FDM Solver with Full Visualization Pipeline

This script demonstrates a complete workflow:
1. Visualize environment setup
2. Run simulation with live visualization
3. Analyze results and generate multiple field visualizations
4. Save outputs

This version is written for the updated FDMSolver that supports:
    - reynolds_number -> viscosity conversion
    - sparse pressure solves via bicgstab / cg / amg
"""

import sys
sys.path.insert(0, 'src')

import os
import numpy as np

from solvers import FDMSolver
from envirmonment import KarmannVortex
from visualization import (
    VelocityMagnitudeVisualizer,
    VorticityVisualizer,
    PressureVisualizer,
    EnvironmentVisualizer,
    FlowFieldPlotter,
)


def main():
    """Run complete FDM simulation with visualization pipeline."""

    print("\n" + "=" * 70)
    print("FDM Solver - Complete Visualization Pipeline")
    print("=" * 70)

    # ---------------------------------------------------------------------
    # User settings: change these first
    # ---------------------------------------------------------------------
    output_dir = "assignment_3/outputs_fdm"
    os.makedirs(output_dir, exist_ok=True)

    # Physical/environment settings
    inlet_velocity = 0.12
    reynolds_number = 400.0     
    fallback_viscosity = 0.001   # used only if reynolds_number=None

    # Grid / solver settings
    nx = 301
    ny = 120
    dt = 1e-1
    n_steps = 4000

    # Pressure solver: "bicgstab", "cg", or "amg"
    poisson_method = "amg"

    pressure_tol = 1e-5
    pressure_maxiter = 300
    adaptive_dt = True

    # Visualization
    # live_visualizer = VelocityMagnitudeVisualizer(u_inlet=inlet_velocity, cmap="viridis")
    record_video = False
    # video_filename = os.path.join(output_dir, "fdm_simulation.mp4")
    # visualize_every = 50

    # ---------------------------------------------------------------------
    # Step 1: Visualize environment setup
    # ---------------------------------------------------------------------
    print("\n[1/4] Visualizing environment setup...")

    env = KarmannVortex(
        viscosity=fallback_viscosity,
        v0=inlet_velocity,
    )

    print(f"      Environment: KarmannVortex")
    print(f"      Domain: {env.x_range} × {env.y_range}")
    print(f"      Obstacle: center={env.circle_center}, radius={env.circle_radius}")
    print(f"      Fallback viscosity in environment: {env.viscosity}")
    print(f"      Inlet velocity: {env.v0}")
    print(f"      Target Reynolds number: {reynolds_number}")

    env_vis = EnvironmentVisualizer(env, nx=nx, ny=ny)
    fig, ax = env_vis.plot_environment(show_initial_conditions=True)
    env_path = os.path.join(output_dir, "01_environment_setup.png")
    fig.savefig(env_path, dpi=150, bbox_inches="tight")
    print(f"      Saved: {env_path}")

    # ---------------------------------------------------------------------
    # Step 2: Run simulation with live visualization
    # ---------------------------------------------------------------------
    print("\n[2/4] Running FDM simulation with live velocity visualization...")

    solver = FDMSolver(
        environment=env,
        nx=nx,
        ny=ny,
        dt=dt,
        n_steps=n_steps,
        reynolds_number=reynolds_number,   # <- new interface
        poisson_method=poisson_method,
        pressure_tol=pressure_tol,
        pressure_maxiter=pressure_maxiter,
        adaptive_dt=adaptive_dt,
        cfl_safety=0.20,
        diff_safety=0.20,
        use_preconditioner=True,
        convection_order="second", 
        outlet_bc="zero_gradient",
        outlet_convection_speed=None,  # used for convective outlet BC


    )

    print(f"      Solver configuration:")
    print(f"        Grid size: {nx} × {ny}")
    print(f"        Base dt: {dt}")
    print(f"        Number of steps: {n_steps}")
    print(f"        Pressure method: {poisson_method}")
    print(f"        Pressure tolerance: {pressure_tol}")
    print(f"        Pressure max iterations: {pressure_maxiter}")
    print(f"        Adaptive dt: {adaptive_dt}")

    print(f"      Running simulation...")
    result = solver.solve(
        verbose=True,
        # visualizer=live_visualizer,
        # record_video=record_video,
        # video_filename=video_filename,
        # visualize_every=visualize_every,
        return_history=True,
    )

    ux = result["ux"]
    uy = result["uy"]
    p = result["p"]
    obstacle = result["obstacle"]
    fluid = result["fluid"]
    metadata = result["metadata"]
    history = result.get("history", None)

    print(f"      Simulation complete!")
    print(f"      Final state - ux range: [{ux.min():.6f}, {ux.max():.6f}]")
    print(f"      Final state - uy range: [{uy.min():.6f}, {uy.max():.6f}]")
    print(f"      Final state - p range:  [{p.min():.6f}, {p.max():.6f}]")

    # ---------------------------------------------------------------------
    # Step 3: Generate multiple field visualizations
    # ---------------------------------------------------------------------
    print("\n[3/4] Generating field visualizations...")

    plotter = FlowFieldPlotter(
        metadata["nx"],
        metadata["ny"],
        obstacle=obstacle,
        figsize=(22, 4),
        dpi=100,
    )

    # 1) Velocity magnitude
    print("      Generating velocity magnitude field...")
    vel_viz = VelocityMagnitudeVisualizer(u_inlet=metadata["u_inlet"])
    vel_field = vel_viz.compute_field(ux, uy)
    plotter.plot_field(vel_field, vel_viz, step=metadata["n_steps"])
    vel_path = os.path.join(output_dir, "02_final_velocity.png")
    plotter.save(vel_path)

    # 2) Vorticity
    print("      Generating vorticity field...")
    vor_viz = VorticityVisualizer(cmap="RdBu_r")
    vor_field = vor_viz.compute_field(ux, uy)
    ax.set_aspect('equal')
    plotter.plot_field(vor_field, vor_viz, step=metadata["n_steps"])
    vor_path = os.path.join(output_dir, "03_final_vorticity.png")
    plotter.save(vor_path)

    # 3) Pressure
    print("      Generating pressure field...")
    pres_viz = PressureVisualizer(cmap="coolwarm")
    pres_field = pres_viz.compute_field(ux, uy, p)
    plotter.plot_field(pres_field, pres_viz, step=metadata["n_steps"])
    pres_path = os.path.join(output_dir, "04_final_pressure.png")
    plotter.save(pres_path)

    plotter.close()



    # ---------------------------------------------------------------------
    # Step 4: Analysis and statistics
    # ---------------------------------------------------------------------
    print("\n[4/4] Analyzing results...")

    speed = np.sqrt(ux**2 + uy**2)
    speed_fluid = speed[fluid]

    vorticity = (
        (np.roll(uy, -1, axis=0) - np.roll(uy, 1, axis=0)) / (2 * metadata["dx"])
        - (np.roll(ux, -1, axis=1) - np.roll(ux, 1, axis=1)) / (2 * metadata["dy"])
    )
    vor_fluid = vorticity[fluid]

    p_fluid = p[fluid]

    print(f"\n      Velocity statistics:")
    print(f"        Mean speed: {speed_fluid.mean():.6f}")
    print(f"        Max speed:  {speed_fluid.max():.6f}")
    print(f"        Min speed:  {speed_fluid.min():.6f}")
    print(f"        Std dev:    {speed_fluid.std():.6f}")

    print(f"\n      Vorticity statistics:")
    print(f"        Mean:    {vor_fluid.mean():.6f}")
    print(f"        Max:     {vor_fluid.max():.6f}")
    print(f"        Min:     {vor_fluid.min():.6f}")
    print(f"        Std dev: {vor_fluid.std():.6f}")

    print(f"\n      Pressure statistics:")
    print(f"        Mean:    {p_fluid.mean():.6f}")
    print(f"        Max:     {p_fluid.max():.6f}")
    print(f"        Min:     {p_fluid.min():.6f}")
    print(f"        Std dev: {p_fluid.std():.6f}")


    if "history" in result:
        hist = result["history"]
        print(f"\n      Time integration diagnostics:")
        print(f"        Final simulated time: {metadata['time_final']:.6f}")
        print(f"        Final dt:             {hist['dt'][-1]:.6e}")
        print(f"        Final div_inf:        {hist['div_inf'][-1]:.6e}")
        print(f"        Max speed over run:   {np.max(hist['speed_max']):.6f}")
        print(f"        Max pressure over run:{np.max(hist['p_max']):.6f}")

    # ---------------------------------------------------------------------
    # Optional: save history arrays
    # ---------------------------------------------------------------------
    if "history" in result:
        hist = result["history"]
        np.savez(
            os.path.join(output_dir, "fdm_history.npz"),
            time=np.asarray(hist["time"]),
            dt=np.asarray(hist["dt"]),
            u_max=np.asarray(hist["u_max"]),
            v_max=np.asarray(hist["v_max"]),
            speed_max=np.asarray(hist["speed_max"]),
            div_inf=np.asarray(hist["div_inf"]),
            p_max=np.asarray(hist["p_max"]),
        )
        print(f"\n      Saved: {os.path.join(output_dir, 'fdm_history.npz')}")

    # ---------------------------------------------------------------------
    # Summary
    # ---------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("Complete Visualization Pipeline Summary")
    print("=" * 70)

    print("\nGenerated files:")
    print(f"  1. {env_path}  - Initial setup visualization")
    print(f"  2. {vel_path}  - Final velocity magnitude")
    print(f"  3. {vor_path}  - Final vorticity field")
    print(f"  4. {pres_path} - Final pressure field")
    if "history" in result:
        print(f"  5. {os.path.join(output_dir, 'fdm_history.npz')} - Time history arrays")

    print("\nSimulation metadata:")
    print(f"  Grid size: {metadata['nx']} × {metadata['ny']}")
    print(f"  Reynolds number: {metadata['reynolds_number']:.3f}")
    print(f"  Viscosity used: {metadata['nu']:.6e}")
    print(f"  Total timesteps: {metadata['n_steps']}")
    print(f"  Final simulated time: {metadata['time_final']:.6f}")
    print(f"  Pressure method: {metadata['poisson_method']}")
    print(f"  Pressure tolerance: {metadata['pressure_tol']:.1e}")
    print(f"  Pressure maxiter: {metadata['pressure_maxiter']}")
    print(f"  Pressure unknowns: {metadata['pressure_unknowns']}")

    print("\n" + "=" * 70)
    print("Pipeline complete! Check outputs_fdm/ directory for generated figures.")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        exit(1)