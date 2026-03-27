"""
Visualization module for LBM solver and environments.

Provides reusable visualization components for displaying:
  - Flow fields (velocity magnitude, vorticity, pressure)
  - Obstacle geometry
  - Real-time animations during simulation
  - Video export functionality
"""

from abc import ABC, abstractmethod
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
import warnings


class FieldVisualizer(ABC):
    """Abstract base class for visualizing scalar fields."""

    @abstractmethod
    def compute_field(self, ux, uy, rho=None):
        """Compute the field to be visualized from velocity and density."""
        pass

    @abstractmethod
    def get_plot_params(self):
        """Return dict with 'cmap', 'vmin', 'vmax' for imshow."""
        pass

    @abstractmethod
    def get_title(self, step=None):
        """Return title string for the plot."""
        pass


class VelocityMagnitudeVisualizer(FieldVisualizer):
    """Visualize velocity magnitude |u| = sqrt(ux^2 + uy^2)."""

    def __init__(self, u_inlet=0.12, cmap='viridis'):
        self.u_inlet = u_inlet
        self.cmap = cmap

    def compute_field(self, ux, uy, rho=None):
        """Compute velocity magnitude."""
        return np.sqrt(ux**2 + uy**2)

    def get_plot_params(self):
        """Return plotting parameters for velocity magnitude."""
        return {
            'cmap': self.cmap,
            'vmin': 0,
            'vmax': self.u_inlet * 2.0,
            'label': 'Speed |u|'
        }

    def get_title(self, step=None):
        """Return title for velocity magnitude plot."""
        if step is not None:
            return f"Velocity magnitude — step {step}"
        return "Velocity magnitude"


class VorticityVisualizer(FieldVisualizer):
    """Visualize vorticity (curl of velocity)."""

    def __init__(self, cmap='RdBu_r'):
        self.cmap = cmap

    def compute_field(self, ux, uy, rho=None):
        """Compute vorticity: ω = ∂uy/∂x - ∂ux/∂y."""
        vorticity = (np.roll(uy, -1, axis=0) - np.roll(uy, 1, axis=0)
                     - np.roll(ux, -1, axis=1) + np.roll(ux, 1, axis=1))
        return vorticity

    def get_plot_params(self):
        """Return plotting parameters for vorticity."""
        return {
            'cmap': self.cmap,
            'vmin': -0.04,
            'vmax': 0.04,
            'label': 'Vorticity ω'
        }

    def get_title(self, step=None):
        """Return title for vorticity plot."""
        if step is not None:
            return f"Vorticity field — step {step}"
        return "Vorticity field"


class PressureVisualizer(FieldVisualizer):
    """Visualize pressure field (proportional to density)."""

    def __init__(self, cmap='coolwarm'):
        self.cmap = cmap

    def compute_field(self, ux, uy, rho=None):
        """Return density (proportional to pressure in incompressible LBM)."""
        if rho is None:
            raise ValueError("Pressure visualization requires density field")
        return rho

    def get_plot_params(self):
        """Return plotting parameters for pressure."""
        return {
            'cmap': self.cmap,
            'vmin': None,  # Auto scaling
            'vmax': None,
            'label': 'Pressure (density)'
        }

    def get_title(self, step=None):
        """Return title for pressure plot."""
        if step is not None:
            return f"Pressure field — step {step}"
        return "Pressure field"


class EnvironmentVisualizer:
    """Visualize the simulation environment (geometry, initial conditions, etc.)."""

    def __init__(self, environment, nx, ny):
        """
        Initialize environment visualizer.

        Parameters
        ----------
        environment : Environment
            Physical environment object
        nx, ny : int
            Grid dimensions in lattice units
        """
        self.environment = environment
        self.nx = nx
        self.ny = ny

    def plot_environment(self, ax=None, show_initial_conditions=True):
        """
        Plot the environment setup.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            Axes to plot on. If None, create new figure.
        show_initial_conditions : bool
            Whether to show initial velocity field

        Returns
        -------
        fig, ax : matplotlib figure and axes
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 5), dpi=100)
        else:
            fig = ax.get_figure()

        legend_handles = []
        mask_data = None

        # Preferred path: use environment-provided masks and initial fields.
        if hasattr(self.environment, "build_condition_masks"):
            try:
                mask_data = self.environment.build_condition_masks(
                    nx=self.nx, ny=self.ny, t=0.0
                )
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
                    extent=[0, self.nx, 0, self.ny],
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
                        Xg, Yg = np.meshgrid(
                            np.arange(self.nx), np.arange(self.ny), indexing="ij"
                        )
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
            # Fallback path for environments without build_condition_masks.
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

        # Set labels and limits
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
    """
    Real-time flow field visualization with support for different field types.

    Decouples the plotting logic from the solver, allowing flexible visualization.
    """

    def __init__(self, nx, ny, obstacle=None, figsize=(12, 5), dpi=100):
        """
        Initialize the plotter.

        Parameters
        ----------
        nx, ny : int
            Grid dimensions
        obstacle : ndarray, optional
            Boolean mask for obstacle regions
        figsize : tuple, optional
            Figure size in inches
        dpi : int, optional
            Figure DPI
        """
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
        """Initialize figure and axes if not already done."""
        if self.fig is None:
            self.fig, self.ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)

    def enable_interactive_mode(self):
        """Enable interactive plotting (for live animation)."""
        if not self._is_interactive:
            plt.ion()
            self._is_interactive = True
            self.setup_figure()

    def disable_interactive_mode(self):
        """Disable interactive plotting."""
        if self._is_interactive:
            plt.ioff()
            self._is_interactive = False

    def plot_field(self, field, visualizer, step=None):
        """
        Plot a scalar field with given visualizer.

        Parameters
        ----------
        field : ndarray
            2D array of field values
        visualizer : FieldVisualizer
            Visualizer object specifying plot parameters
        step : int, optional
            Timestep number for title

        Returns
        -------
        im : matplotlib image object
        """
        self.setup_figure()
        self.ax.clear()

        # Mask obstacle region
        field_masked = field.copy()
        field_masked[self.obstacle] = np.nan

        # Get visualization parameters
        params = visualizer.get_plot_params()
        vmin = params['vmin']
        vmax = params['vmax']

        # Handle auto-scaling
        if vmin is None or vmax is None:
            vmin = np.nanmin(field_masked)
            vmax = np.nanmax(field_masked)

        # Create image
        self.im = self.ax.imshow(
            field_masked.T, origin='lower',
            cmap=params['cmap'],
            vmin=vmin, vmax=vmax,
            aspect='auto',
            extent=[0, self.nx, 0, self.ny]
        )

        # Colorbar
        if self.cbar is None:
            self.cbar = plt.colorbar(self.im, ax=self.ax, label=params['label'])
        else:
            self.cbar.update_normal(self.im)

        # Labels and title
        self.ax.set_xlabel("x (lattice units)")
        self.ax.set_ylabel("y (lattice units)")
        self.ax.set_title(visualizer.get_title(step))

        self.fig.tight_layout()

        return self.im

    def pause(self, interval=0.001):
        """Pause for interactive animation."""
        if self._is_interactive:
            plt.pause(interval)

    def show(self):
        """Display the current figure."""
        plt.show()

    def save(self, filename):
        """Save current figure to file."""
        if self.fig is not None:
            self.fig.savefig(filename, dpi=self.dpi, bbox_inches='tight')
            print(f"Figure saved to {filename}")

    def close(self):
        """Close the figure."""
        if self.fig is not None:
            plt.close(self.fig)
            self.fig = None
            self.ax = None
            self.im = None
            self.cbar = None


class FlowVideoRecorder:
    """
    Record flow field animations to video file.

    Separates video recording logic from solver and visualization.
    """

    def __init__(self, nx, ny, obstacle=None, fps=30, figsize=(12, 5), dpi=100):
        """
        Initialize video recorder.

        Parameters
        ----------
        nx, ny : int
            Grid dimensions
        obstacle : ndarray, optional
            Boolean mask for obstacle regions
        fps : int, optional
            Frames per second for video
        figsize : tuple, optional
            Figure size in inches
        dpi : int, optional
            Figure DPI
        """
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
        """Initialize figure for recording."""
        if self.fig is None:
            self.fig, self.ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)

    def start_recording(self, filename, visualizer):
        """
        Start recording to video file.

        Parameters
        ----------
        filename : str
            Output video filename (e.g., 'output.mp4')
        visualizer : FieldVisualizer
            Visualizer for field type

        Returns
        -------
        bool
            True if recording started successfully, False if FFMpeg unavailable
        """
        self._setup_figure()

        # Setup FFMpeg writer
        try:
            self.writer = FFMpegWriter(fps=self.fps, bitrate=1800)
            self.writer.setup(self.fig, filename, dpi=self.dpi)
            self.frame_count = 0
            print(f"Started recording to {filename} at {self.fps} fps")
            return True
        except Exception as e:
            warnings.warn(
                f"Could not setup video writer: {e}\n"
                "Make sure FFMpeg is installed. Video recording disabled.\n"
                "Install with: apt-get install ffmpeg (Linux), "
                "brew install ffmpeg (macOS), or download from ffmpeg.org (Windows)"
            )
            self.writer = None
            return False

    def add_frame(self, field, visualizer, step=None):
        """
        Add a frame to the video.

        Parameters
        ----------
        field : ndarray
            2D field to plot
        visualizer : FieldVisualizer
            Visualizer for field
        step : int, optional
            Timestep for display
        """
        if self.writer is None:
            return

        self.ax.clear()

        # Mask obstacle
        field_masked = field.copy()
        field_masked[self.obstacle] = np.nan

        # Get visualization params
        params = visualizer.get_plot_params()
        vmin = params['vmin']
        vmax = params['vmax']

        if vmin is None or vmax is None:
            vmin = np.nanmin(field_masked)
            vmax = np.nanmax(field_masked)

        # Create image
        self.im = self.ax.imshow(
            field_masked.T, origin='lower',
            cmap=params['cmap'],
            vmin=vmin, vmax=vmax,
            aspect='auto',
            extent=[0, self.nx, 0, self.ny]
        )

        # Colorbar
        if self.cbar is None:
            self.cbar = plt.colorbar(self.im, ax=self.ax, label=params['label'])

        # Labels and title
        self.ax.set_xlabel("x (lattice units)")
        self.ax.set_ylabel("y (lattice units)")
        self.ax.set_title(visualizer.get_title(step))

        self.fig.tight_layout()

        # Write frame
        self.writer.grab_frame()
        self.frame_count += 1

    def finish_recording(self):
        """Finalize video recording."""
        if self.writer is not None:
            self.writer.finish()
            print(f"Saved {self.frame_count} frames to video")
            self.writer = None

    def close(self):
        """Close and cleanup."""
        if self.writer is not None:
            self.finish_recording()
        if self.fig is not None:
            plt.close(self.fig)
            self.fig = None
            self.ax = None


class SimulationVisualizer:
    """
    High-level visualizer for monitoring LBM simulation in real-time.

    Integrates flow field plotter with optional video recording.
    """

    def __init__(self, nx, ny, obstacle=None, fps=30, record_video=False,
                 video_filename='simulation.mp4', figsize=(12, 5), dpi=100):
        """
        Initialize simulation visualizer.

        Parameters
        ----------
        nx, ny : int
            Grid dimensions
        obstacle : ndarray, optional
            Obstacle mask
        fps : int, optional
            Video FPS if recording
        record_video : bool, optional
            Whether to record video
        video_filename : str, optional
            Output video filename
        figsize : tuple, optional
            Figure size
        dpi : int, optional
            Figure DPI
        """
        self.plotter = FlowFieldPlotter(nx, ny, obstacle, figsize, dpi)
        self.recorder = None
        self.record_video = record_video
        self.video_filename = video_filename

        if record_video:
            self.recorder = FlowVideoRecorder(nx, ny, obstacle, fps, figsize, dpi)

    def setup(self, visualizer):
        """
        Setup visualization with a specific field visualizer.

        Parameters
        ----------
        visualizer : FieldVisualizer
            The type of field to visualize
        """
        self.plotter.enable_interactive_mode()
        if self.recorder is not None:
            # Try to start recording, but continue if FFMpeg unavailable
            recording_started = self.recorder.start_recording(
                self.video_filename, visualizer
            )
            if not recording_started:
                # FFMpeg not available, disable recording
                self.recorder = None
        self.visualizer = visualizer

    def update(self, field, step):
        """
        Update visualization with new field data.

        Parameters
        ----------
        field : ndarray
            Current field values
        step : int
            Current timestep
        """
        # Plot live
        self.plotter.plot_field(field, self.visualizer, step)
        self.plotter.pause(interval=0.001)

        # Record if enabled
        if self.recorder is not None:
            self.recorder.add_frame(field, self.visualizer, step)

    def finalize(self):
        """Finalize visualization and save video if enabled."""
        self.plotter.disable_interactive_mode()
        if self.recorder is not None:
            self.recorder.finish_recording()


if __name__ == "__main__":
    # Quick test of visualization components
    print("Visualization module loaded successfully")
    print("Available visualizers:")
    print("  - VelocityMagnitudeVisualizer")
    print("  - VorticityVisualizer")
    print("  - PressureVisualizer")
    print("  - EnvironmentVisualizer")
    print("  - FlowFieldPlotter")
    print("  - FlowVideoRecorder")
    print("  - SimulationVisualizer")

