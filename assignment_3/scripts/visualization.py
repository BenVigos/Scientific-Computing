"""
Visualization module for LBM solver and environments.

Provides reusable visualization components for displaying:
  - Flow fields (velocity magnitude, vorticity, pressure)
  - Obstacle geometry
  - Real-time animations during simulation
  - Video export functionality
"""

from abc import ABC, abstractmethod
import warnings

import matplotlib.pyplot as plt
import numpy as np


class FieldVisualizer(ABC):
    """Abstract base class for visualizing scalar fields."""

    @abstractmethod
    def compute_field(self, ux, uy, rho=None):
        """Compute the field to be visualized from velocity and density."""

    @abstractmethod
    def get_plot_params(self):
        """Return dict with 'cmap', 'vmin', 'vmax', and 'label' for imshow."""

    @abstractmethod
    def get_title(self, step=None):
        """Return title string for the plot."""

    def draw_overlay(self, ax, obstacle=None):
        """Optional overlay hook (e.g., streamlines)."""
        return


class VelocityMagnitudeVisualizer(FieldVisualizer):
    """Visualize velocity magnitude |u| = sqrt(ux^2 + uy^2)."""

    def __init__(self, u_inlet=0.12, cmap="viridis"):
        self.u_inlet = u_inlet
        self.cmap = cmap

    def compute_field(self, ux, uy, rho=None):
        return np.sqrt(ux ** 2 + uy ** 2)

    def get_plot_params(self):
        return {
            "cmap": self.cmap,
            "vmin": 0.0,
            "vmax": self.u_inlet * 2.0,
            "label": "Speed |u|",
        }

    def get_title(self, step=None):
        return f"Velocity magnitude - step {step}" if step is not None else "Velocity magnitude"


class VorticityVisualizer(FieldVisualizer):
    """Visualize vorticity (curl of velocity)."""

    def __init__(self, cmap="RdBu_r"):
        self.cmap = cmap

    def compute_field(self, ux, uy, rho=None):
        return (
            np.roll(uy, -1, axis=0)
            - np.roll(uy, 1, axis=0)
            - np.roll(ux, -1, axis=1)
            + np.roll(ux, 1, axis=1)
        )

    def get_plot_params(self):
        return {
            "cmap": self.cmap,
            "vmin": -0.04,
            "vmax": 0.04,
            "label": "Vorticity w",
        }

    def get_title(self, step=None):
        return f"Vorticity field - step {step}" if step is not None else "Vorticity field"


class PressureVisualizer(FieldVisualizer):
    """Visualize pressure field (proportional to density)."""

    def __init__(self, cmap="coolwarm"):
        self.cmap = cmap

    def compute_field(self, ux, uy, rho=None):
        if rho is None:
            raise ValueError("Pressure visualization requires density field")
        return rho

    def get_plot_params(self):
        return {
            "cmap": self.cmap,
            "vmin": None,
            "vmax": None,
            "label": "Pressure (density)",
        }

    def get_title(self, step=None):
        return f"Pressure field - step {step}" if step is not None else "Pressure field"


class StreamlineVisualizer(FieldVisualizer):
    """Visualize speed with streamline overlay to highlight vortices."""

    def __init__(self, cmap="viridis", density=1.1, color="white", linewidth=0.9):
        self.cmap = cmap
        self.density = density
        self.color = color
        self.linewidth = linewidth
        self._ux = None
        self._uy = None

    def compute_field(self, ux, uy, rho=None):
        self._ux = ux
        self._uy = uy
        return np.sqrt(ux ** 2 + uy ** 2)

    def get_plot_params(self):
        return {
            "cmap": self.cmap,
            "vmin": None,
            "vmax": None,
            "label": "Speed |u|",
        }

    def get_title(self, step=None):
        return f"Speed + streamlines - step {step}" if step is not None else "Speed + streamlines"

    def draw_overlay(self, ax, obstacle=None):
        if self._ux is None or self._uy is None:
            return

        u_plot = self._ux.T
        v_plot = self._uy.T

        if obstacle is not None:
            mask = obstacle.T
            u_plot = np.ma.array(u_plot, mask=mask)
            v_plot = np.ma.array(v_plot, mask=mask)

        x = np.arange(self._ux.shape[0])
        y = np.arange(self._ux.shape[1])
        ax.streamplot(
            x,
            y,
            u_plot,
            v_plot,
            density=self.density,
            color=self.color,
            linewidth=self.linewidth,
            arrowsize=0.8,
        )


class EnvironmentVisualizer:
    """Visualize environment geometry and initial condition vectors."""

    def __init__(self, environment, nx, ny):
        self.environment = environment
        self.nx = nx
        self.ny = ny

    def plot_environment(self, ax=None, show_initial_conditions=True):
        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 5), dpi=100)
        else:
            fig = ax.get_figure()

        legend_handles = []
        mask_data = None

        if hasattr(self.environment, "build_condition_masks"):
            try:
                mask_data = self.environment.build_condition_masks(nx=self.nx, ny=self.ny, t=0.0)
            except Exception as exc:
                warnings.warn(
                    f"build_condition_masks failed ({exc}); falling back to geometry-based plotting."
                )

        if isinstance(mask_data, dict):
            masks = mask_data.get("masks", {})
            obstacle = masks.get("obstacle")

            if isinstance(obstacle, np.ndarray) and obstacle.shape == (self.nx, self.ny):
                obstacle_overlay = np.where(obstacle, 1.0, np.nan)
                ax.imshow(
                    obstacle_overlay.T,
                    origin="lower",
                    cmap="Greys",
                    vmin=0,
                    vmax=1,
                    alpha=0.5,
                    extent=(0, self.nx, 0, self.ny),
                    aspect="auto",
                )
                legend_handles.append(
                    plt.Line2D([0], [0], color="black", lw=6, alpha=0.6, label="Obstacle")
                )

            if show_initial_conditions:
                u0 = mask_data.get("u0")
                v0 = mask_data.get("v0")
                if isinstance(u0, np.ndarray) and isinstance(v0, np.ndarray):
                    if u0.shape == (self.nx, self.ny) and v0.shape == (self.nx, self.ny):
                        step_x = max(1, self.nx // 30)
                        step_y = max(1, self.ny // 15)
                        Xg, Yg = np.meshgrid(np.arange(self.nx), np.arange(self.ny), indexing="ij")
                        ax.quiver(
                            Xg[::step_x, ::step_y],
                            Yg[::step_x, ::step_y],
                            u0[::step_x, ::step_y],
                            v0[::step_x, ::step_y],
                            alpha=0.6,
                            scale=5,
                        )
                        legend_handles.append(
                            plt.Line2D([0], [0], color="C0", lw=2, label="Initial velocity")
                        )
        else:
            try:
                cx = self.environment.circle_center[0]
                cy = self.environment.circle_center[1]
                r = self.environment.circle_radius
                x_range = self.environment.x_range
                y_range = self.environment.y_range

                cx_disp = (cx - x_range[0]) / (x_range[1] - x_range[0]) * self.nx
                cy_disp = (cy - y_range[0]) / (y_range[1] - y_range[0]) * self.ny
                r_disp = r / (x_range[1] - x_range[0]) * self.nx

                circle = plt.Circle((cx_disp, cy_disp), r_disp, color="black", fill=True, alpha=0.7)
                ax.add_patch(circle)
                legend_handles.append(
                    plt.Line2D([0], [0], color="black", lw=6, alpha=0.7, label="Obstacle")
                )
            except (AttributeError, TypeError):
                pass

            if show_initial_conditions:
                try:
                    x_range = self.environment.x_range
                    y_range = self.environment.y_range
                    x = np.linspace(*x_range, 30)
                    y = np.linspace(*y_range, 15)
                    X, Y = np.meshgrid(x, y, indexing="xy")
                    U, V = self.environment.initial_condition(X, Y)

                    ax.quiver(X, Y, U, V, alpha=0.6, scale=5)
                    legend_handles.append(
                        plt.Line2D([0], [0], color="C0", lw=2, label="Initial velocity")
                    )
                except (AttributeError, TypeError):
                    pass

        ax.set_xlabel("x (lattice units)")
        ax.set_ylabel("y (lattice units)")
        ax.set_xlim(0, self.nx)
        ax.set_ylim(0, self.ny)
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)
        ax.set_title("Simulation Environment")
        if legend_handles:
            ax.legend(handles=legend_handles, loc="upper right")

        fig.tight_layout()
        return fig, ax


class FlowFieldPlotter:
    """Real-time/static field plotting independent of the solver."""

    def __init__(self, nx, ny, obstacle=None, figsize=(22, 4), dpi=100):
        self.nx = nx
        self.ny = ny
        self.obstacle = obstacle if obstacle is not None else np.zeros((nx, ny), dtype=bool)

        self.fig = None
        self.ax = None
        self.im = None
        self.cbar = None
        self.figsize = figsize
        self.dpi = dpi
        self._is_interactive = False

    def setup_figure(self):
        if self.fig is None:
            self.fig, self.ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)

    def enable_interactive_mode(self):
        if not self._is_interactive:
            plt.ion()
            self._is_interactive = True
            self.setup_figure()

    def disable_interactive_mode(self):
        if self._is_interactive:
            plt.ioff()
            self._is_interactive = False

    def plot_field(self, field, visualizer, step=None):
        self.setup_figure()
        self.ax.clear()

        field_masked = field.copy()
        field_masked[self.obstacle] = np.nan

        params = visualizer.get_plot_params()
        vmin = params["vmin"]
        vmax = params["vmax"]

        if vmin is None or vmax is None:
            vmin = np.nanmin(field_masked)
            vmax = np.nanmax(field_masked)

        self.im = self.ax.imshow(
            field_masked.T,
            origin="lower",
            cmap=params["cmap"],
            vmin=vmin,
            vmax=vmax,
            aspect="auto",
            extent=(0, self.nx, 0, self.ny),
        )

        if self.cbar is None:
            self.cbar = plt.colorbar(self.im, ax=self.ax, label=params["label"])
        else:
            self.cbar.update_normal(self.im)

        self.ax.set_xlabel("x (lattice units)")
        self.ax.set_ylabel("y (lattice units)")
        self.ax.set_title(visualizer.get_title(step))

        visualizer.draw_overlay(self.ax, self.obstacle)

        self.fig.tight_layout()
        return self.im

    def pause(self, interval=0.001):
        if self._is_interactive:
            plt.pause(interval)

    def show(self):
        plt.show()

    def save(self, filename):
        if self.fig is not None:
            self.fig.savefig(filename, dpi=self.dpi, bbox_inches="tight")
            print(f"Figure saved to {filename}")

    def close(self):
        if self.fig is not None:
            plt.close(self.fig)
            self.fig = None
            self.ax = None
            self.im = None
            self.cbar = None


class FlowVideoRecorder:
    """Record flow field animations to video file."""

    def __init__(self, nx, ny, obstacle=None, fps=30, figsize=(12, 5), dpi=100):
        self.nx = nx
        self.ny = ny
        self.obstacle = obstacle if obstacle is not None else np.zeros((nx, ny), dtype=bool)
        self.fps = fps
        self.figsize = figsize
        self.dpi = dpi

        self.fig = None
        self.ax = None
        self.im = None
        self.cbar = None
        self.writer = None
        self.frame_count = 0

    def _setup_figure(self):
        if self.fig is None:
            self.fig, self.ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)

    def start_recording(self, filename, visualizer):
        self._setup_figure()

        try:
            from matplotlib.animation import FFMpegWriter
        except Exception:
            warnings.warn("FFMpegWriter is unavailable in this Matplotlib build. Video recording disabled.")
            return False

        try:
            self.writer = FFMpegWriter(fps=self.fps, bitrate=1800)
            self.writer.setup(self.fig, filename, dpi=self.dpi)
            self.frame_count = 0
            print(f"Started recording to {filename} at {self.fps} fps")
            return True
        except Exception as exc:
            warnings.warn(
                f"Could not setup video writer: {exc}\n"
                "FFmpeg may be missing from PATH. Video recording disabled."
            )
            self.writer = None
            return False

    def add_frame(self, field, visualizer, step=None):
        if self.writer is None:
            return

        self.ax.clear()

        field_masked = field.copy()
        field_masked[self.obstacle] = np.nan

        params = visualizer.get_plot_params()
        vmin = params["vmin"]
        vmax = params["vmax"]

        if vmin is None or vmax is None:
            vmin = np.nanmin(field_masked)
            vmax = np.nanmax(field_masked)

        self.im = self.ax.imshow(
            field_masked.T,
            origin="lower",
            cmap=params["cmap"],
            vmin=vmin,
            vmax=vmax,
            aspect="auto",
            extent=(0, self.nx, 0, self.ny),
        )

        if self.cbar is None:
            self.cbar = plt.colorbar(self.im, ax=self.ax, label=params["label"])

        self.ax.set_xlabel("x (lattice units)")
        self.ax.set_ylabel("y (lattice units)")
        self.ax.set_title(visualizer.get_title(step))

        visualizer.draw_overlay(self.ax, self.obstacle)

        self.fig.tight_layout()
        self.writer.grab_frame()
        self.frame_count += 1

    def finish_recording(self):
        if self.writer is not None:
            self.writer.finish()
            print(f"Saved {self.frame_count} frames to video")
            self.writer = None

    def close(self):
        if self.writer is not None:
            self.finish_recording()
        if self.fig is not None:
            plt.close(self.fig)
            self.fig = None
            self.ax = None


class SimulationVisualizer:
    """High-level visualizer for live plotting and optional recording."""

    def __init__(
        self,
        nx,
        ny,
        obstacle=None,
        fps=30,
        record_video=False,
        video_filename="simulation.mp4",
        figsize=(22, 4),
        dpi=100,
    ):
        self.plotter = FlowFieldPlotter(nx, ny, obstacle, figsize, dpi)
        self.recorder = None
        self.record_video = record_video
        self.video_filename = video_filename

        if record_video:
            self.recorder = FlowVideoRecorder(nx, ny, obstacle, fps, figsize, dpi)

    def setup(self, visualizer):
        self.plotter.enable_interactive_mode()
        self.visualizer = visualizer

        if self.recorder is not None:
            if not self.recorder.start_recording(self.video_filename, visualizer):
                self.recorder = None

    def update(self, field, step):
        self.plotter.plot_field(field, self.visualizer, step)
        self.plotter.pause(interval=0.001)

        if self.recorder is not None:
            self.recorder.add_frame(field, self.visualizer, step)

    def finalize(self):
        self.plotter.disable_interactive_mode()
        if self.recorder is not None:
            self.recorder.finish_recording()


if __name__ == "__main__":
    print("Visualization module loaded successfully")
    print("Available visualizers:")
    print("  - VelocityMagnitudeVisualizer")
    print("  - VorticityVisualizer")
    print("  - PressureVisualizer")
    print("  - StreamlineVisualizer")
    print("  - EnvironmentVisualizer")
    print("  - FlowFieldPlotter")
    print("  - FlowVideoRecorder")
    print("  - SimulationVisualizer")

