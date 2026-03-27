#!/usr/bin/env python
"""
Quick test of the visualization module.
"""

import sys
sys.path.insert(0, 'src')

from visualization import (
    VelocityMagnitudeVisualizer,
    VorticityVisualizer,
    PressureVisualizer,
    EnvironmentVisualizer,
    FlowFieldPlotter,
    FlowVideoRecorder,
    SimulationVisualizer
)
from envirmonment import KarmannVortex
import numpy as np


def test_visualizers():
    """Test that all visualizer classes can be instantiated."""
    print("Testing visualizer instantiation...")

    vel_viz = VelocityMagnitudeVisualizer(u_inlet=0.12, cmap='viridis')
    print("[OK] VelocityMagnitudeVisualizer")

    vor_viz = VorticityVisualizer(cmap='RdBu_r')
    print("[OK] VorticityVisualizer")

    pres_viz = PressureVisualizer(cmap='coolwarm')
    print("[OK] PressureVisualizer")

    # Test field computation
    ux = np.random.randn(100, 50) * 0.1
    uy = np.random.randn(100, 50) * 0.05
    rho = np.ones((100, 50))

    vel_field = vel_viz.compute_field(ux, uy)
    print(f"[OK] Velocity field shape: {vel_field.shape}")

    vor_field = vor_viz.compute_field(ux, uy)
    print(f"[OK] Vorticity field shape: {vor_field.shape}")

    pres_field = pres_viz.compute_field(ux, uy, rho)
    print(f"[OK] Pressure field shape: {pres_field.shape}")

    # Test plot parameters
    vel_params = vel_viz.get_plot_params()
    print(f"[OK] Velocity plot params: cmap={vel_params['cmap']}, label={vel_params['label']}")

    # Test titles
    title = vel_viz.get_title(step=100)
    print(f"[OK] Title: {title}")


def test_environment_visualizer():
    """Test environment visualization."""
    print("\nTesting environment visualizer...")

    env = KarmannVortex(v0=1.0)
    env_vis = EnvironmentVisualizer(env, nx=150, ny=60)
    print("[OK] EnvironmentVisualizer instantiated")

    # Note: We won't actually plot to avoid display issues in test environment
    # fig, ax = env_vis.plot_environment(show_initial_conditions=True)
    # print("[OK] Environment plot created")


def test_flow_field_plotter():
    """Test flow field plotter."""
    print("\nTesting flow field plotter...")

    obstacle = np.zeros((100, 50), dtype=bool)
    obstacle[40:60, 20:30] = True

    plotter = FlowFieldPlotter(100, 50, obstacle=obstacle)
    print("[OK] FlowFieldPlotter instantiated")

    # Test setup
    plotter.setup_figure()
    print("[OK] Figure setup complete")

    # Create dummy data
    ux = np.random.randn(100, 50) * 0.1
    uy = np.random.randn(100, 50) * 0.05

    visualizer = VelocityMagnitudeVisualizer(u_inlet=0.12)
    field = visualizer.compute_field(ux, uy)

    # Note: We won't actually display, just test that plotting methods work
    # plotter.plot_field(field, visualizer, step=100)
    # print("[OK] Field plotted successfully")

    plotter.close()
    print("[OK] Plotter closed")


def test_video_recorder():
    """Test video recorder initialization."""
    print("\nTesting video recorder...")

    obstacle = np.zeros((100, 50), dtype=bool)
    recorder = FlowVideoRecorder(100, 50, obstacle=obstacle, fps=30)
    print("[OK] FlowVideoRecorder instantiated")

    recorder.close()
    print("[OK] Recorder closed")


def test_simulation_visualizer():
    """Test simulation visualizer."""
    print("\nTesting simulation visualizer...")

    obstacle = np.zeros((100, 50), dtype=bool)
    obstacle[40:60, 20:30] = True

    sim_vis = SimulationVisualizer(
        100, 50,
        obstacle=obstacle,
        fps=30,
        record_video=False,
        video_filename='test.mp4'
    )
    print("[OK] SimulationVisualizer instantiated without video recording")

    sim_vis_with_video = SimulationVisualizer(
        100, 50,
        obstacle=obstacle,
        fps=30,
        record_video=True,
        video_filename='test_video.mp4'
    )
    print("[OK] SimulationVisualizer instantiated with video recording")


def main():
    """Run all tests."""
    print("=" * 70)
    print("Visualization Module Tests")
    print("=" * 70)

    try:
        test_visualizers()
        test_environment_visualizer()
        test_flow_field_plotter()
        test_video_recorder()
        test_simulation_visualizer()

        print("\n" + "=" * 70)
        print("[OK] All tests passed!")
        print("=" * 70)

    except Exception as e:
        print(f"\n[FAILED] Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())

