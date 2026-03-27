from abc import ABC, abstractmethod
import numpy as np
from numba import njit


@njit(cache=True)
def _compute_macroscopic(f, c, rho, ux, uy):
    nx, ny, _ = f.shape
    for x in range(nx):
        for y in range(ny):
            rho_xy = 0.0
            ux_xy = 0.0
            uy_xy = 0.0
            for i in range(9):
                fi = f[x, y, i]
                rho_xy += fi
                ux_xy += fi * c[i, 0]
                uy_xy += fi * c[i, 1]
            rho[x, y] = rho_xy
            if rho_xy > 0.0:
                ux[x, y] = ux_xy / rho_xy
                uy[x, y] = uy_xy / rho_xy
            else:
                ux[x, y] = 0.0
                uy[x, y] = 0.0


@njit(cache=True)
def _equilibrium_kernel(rho, ux, uy, c, w, feq):
    nx, ny = rho.shape
    for x in range(nx):
        for y in range(ny):
            rho_xy = rho[x, y]
            ux_xy = ux[x, y]
            uy_xy = uy[x, y]
            usqr = ux_xy * ux_xy + uy_xy * uy_xy
            for i in range(9):
                cu = c[i, 0] * ux_xy + c[i, 1] * uy_xy
                feq[x, y, i] = w[i] * rho_xy * (1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * usqr)


@njit(cache=True)
def _collide_kernel(f, feq, tau, f_out):
    nx, ny, _ = f.shape
    inv_tau = 1.0 / tau
    for x in range(nx):
        for y in range(ny):
            for i in range(9):
                f_out[x, y, i] = f[x, y, i] - (f[x, y, i] - feq[x, y, i]) * inv_tau


@njit(cache=True)
def _bounce_back_kernel(f, f_out, obstacle, opp):
    nx, ny = obstacle.shape
    for x in range(nx):
        for y in range(ny):
            if obstacle[x, y]:
                for i in range(9):
                    f_out[x, y, i] = f[x, y, opp[i]]


@njit(cache=True)
def _stream_kernel(f_out, c, f):
    nx, ny, _ = f.shape
    for x in range(nx):
        for y in range(ny):
            for i in range(9):
                sx = (x - c[i, 0]) % nx
                sy = (y - c[i, 1]) % ny
                f[x, y, i] = f_out[sx, sy, i]


@njit(cache=True)
def _outlet_open_kernel(f):
    nx, ny, _ = f.shape
    x_out = nx - 1
    x_prev = nx - 2
    for y in range(ny):
        for i in range(9):
            f[x_out, y, i] = f[x_prev, y, i]


def _compute_ramped_velocity(step, u_inlet, tau_ramp):
    """
    Compute exponential ramp-up velocity: u(t) = u_inlet * (1 - exp(-t / tau))

    Parameters
    ----------
    step : int
        Current timestep (1-indexed)
    u_inlet : float
        Target inlet velocity
    tau_ramp : float
        Time constant for ramp-up in timesteps

    Returns
    -------
    float
        Ramped inlet velocity at current step
    """
    if tau_ramp <= 0:
        return u_inlet
    return u_inlet * (1.0 - np.exp(-float(step) / tau_ramp))


@njit(cache=True)
def _inlet_zou_he_kernel(f, u_inlet):
    _, ny, _ = f.shape
    for y in range(ny):
        rho_in = (f[0, y, 0] + f[0, y, 2] + f[0, y, 4] + 2.0 * (f[0, y, 3] + f[0, y, 6] + f[0, y, 7])) / (1.0 - u_inlet)
        f[0, y, 1] = f[0, y, 3] + (2.0 / 3.0) * rho_in * u_inlet
        f[0, y, 5] = f[0, y, 7] - 0.5 * (f[0, y, 2] - f[0, y, 4]) + (1.0 / 6.0) * rho_in * u_inlet
        f[0, y, 8] = f[0, y, 6] + 0.5 * (f[0, y, 2] - f[0, y, 4]) + (1.0 / 6.0) * rho_in * u_inlet


@njit(cache=True)
def _inlet_regularized_kernel(f, u_inlet, c, w):
    _, ny, _ = f.shape
    for y in range(ny):
        rho_in = (f[0, y, 0] + f[0, y, 2] + f[0, y, 4] + 2.0 * (f[0, y, 3] + f[0, y, 6] + f[0, y, 7])) / (1.0 - u_inlet)
        ux = u_inlet
        uy = 0.0
        usqr = ux * ux + uy * uy

        feq = np.empty(9, dtype=np.float64)
        for i in range(9):
            cu = c[i, 0] * ux + c[i, 1] * uy
            feq[i] = w[i] * rho_in * (1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * usqr)

        # Reconstruct incoming populations using equilibrium + reflected non-equilibrium.
        f[0, y, 1] = feq[1] + (f[0, y, 3] - feq[3])
        f[0, y, 5] = feq[5] + (f[0, y, 7] - feq[7])
        f[0, y, 8] = feq[8] + (f[0, y, 6] - feq[6])


class Solver(ABC):
    """Abstract base class for PDE solvers"""

    def __init__(self, environment):
        self.environment = environment

    @abstractmethod
    def solve(self):
        pass

class FEMSolver(Solver):
    """Finite Element Method Solver"""

    def __init__(self, environment, mesh, material):
        super().__init__(environment)
        self.mesh = mesh
        self.material = material

    def solve(self):
        print("Solving using FEM Solver...")
        return "FEM Solution"


class FDMSolver(Solver):
    """Finite Difference Method Solver"""

    def __init__(self, environment, grid, time_step):
        super().__init__(environment)
        self.grid = grid
        self.time_step = time_step

    def solve(self):
        print("Solving using FDM Solver...")
        return "FDM Solution"


class LBMSolver(Solver):
    """
    Lattice Boltzmann Method Solver for Navier-Stokes equations.

    Implements the D2Q9 lattice with BGK collision operator for 2D flow
    simulation. Follows the algorithm:
      1. Compute macroscopic quantities (density, velocity) from distributions
      2. Collision step — relax f toward local equilibrium (BGK)
      3. Bounce-back — reflect populations at obstacle nodes (no-slip wall)
      4. Streaming step — propagate f_i along lattice velocities
      5. Boundary conditions — Zou-He inlet, open outlet
    """

    def __init__(self, environment, nx, ny, u_inlet, reynolds_number=150,
                 n_steps=30000, vis_interval=100, velocity_ramp_tau=None,
                 inlet_bc="zou_he"):
        """
        Initialize the LBM solver.

        Parameters
        ----------
        environment : Environment
            Physical environment with initial and boundary conditions
        nx : int
            Domain length in lattice units
        ny : int
            Domain height in lattice units
        u_inlet : float
            Inlet velocity in lattice units (keep << 1 for low Mach)
        reynolds_number : float, optional
            Target Reynolds number (default: 150)
        n_steps : int, optional
            Total number of timesteps (default: 30000)
        vis_interval : int, optional
            Visualization update interval in timesteps (default: 100)
        velocity_ramp_tau : float, optional
            Time constant (in timesteps) for exponential velocity ramp-up.
            If None (default), velocity ramps to full speed in about n_steps/10.
            Smaller values = faster ramp, larger values = slower ramp.
        inlet_bc : str, optional
            Inlet boundary condition model. Supported: "zou_he" (default)
            and "regularized".
        """
        super().__init__(environment)
        self.nx = nx
        self.ny = ny
        self.u_inlet = u_inlet
        self.reynolds_number = reynolds_number
        self.n_steps = n_steps
        self.vis_interval = min(vis_interval, n_steps // 10)
        # Default ramp time constant: 10% of total steps if not specified
        self.velocity_ramp_tau = velocity_ramp_tau if velocity_ramp_tau is not None else max(100, n_steps // 10)

        inlet_bc_normalized = inlet_bc.lower().replace("-", "_")
        if inlet_bc_normalized not in ("zou_he", "regularized"):
            raise ValueError("inlet_bc must be 'zou_he' or 'regularized'.")
        self.inlet_bc = inlet_bc_normalized

        # D2Q9 Lattice velocities
        self.c = np.array([[0, 0],    # 0 - rest
                           [1, 0],    # 1 - east
                           [0, 1],    # 2 - north
                           [-1, 0],   # 3 - west
                           [0, -1],   # 4 - south
                           [1, 1],    # 5 - north-east
                           [-1, 1],   # 6 - north-west
                           [-1, -1],  # 7 - south-west
                           [1, -1]], dtype=np.int32)  # 8 - south-east

        # D2Q9 weights
        self.w = np.array([4/9,                        # rest
                           1/9, 1/9, 1/9, 1/9,         # axis-aligned
                           1/36, 1/36, 1/36, 1/36], dtype=np.float64)    # diagonals

        # Opposite direction index for bounce-back
        self.opp = np.array([0, 3, 4, 1, 2, 7, 8, 5, 6], dtype=np.int32)

        # Setup obstacle from environment
        self._setup_obstacle()

        # Calculate relaxation time
        self._calculate_relaxation_time()

    def _setup_obstacle(self):
        """Initialize obstacle mask from environment properties."""
        # Check if environment is KarmannVortex
        try:
            cx = self.environment.circle_center[0]
            cy = self.environment.circle_center[1]
            r = self.environment.circle_radius
            x_range = self.environment.x_range
            y_range = self.environment.y_range

            # Convert physical coordinates to lattice units
            # Assuming domain [0, Nx] x [0, Ny] maps to x_range x y_range
            cx_latt = (cx - x_range[0]) / (x_range[1] - x_range[0]) * self.nx
            cy_latt = (cy - y_range[0]) / (y_range[1] - y_range[0]) * self.ny
            r_latt = r / (x_range[1] - x_range[0]) * self.nx

            x = np.arange(self.nx)
            y = np.arange(self.ny)
            X, Y = np.meshgrid(x, y, indexing='ij')
            self.obstacle = (X - cx_latt)**2 + (Y - cy_latt)**2 <= r_latt**2
        except (AttributeError, TypeError):
            # Fallback: no obstacle
            self.obstacle = np.zeros((self.nx, self.ny), dtype=bool)

    def _calculate_relaxation_time(self):
        """Calculate BGK relaxation time from Reynolds number."""
        # Use environment properties if available, otherwise estimate from obstacle
        try:
            r = self.environment.circle_radius
            x_range = self.environment.x_range
            D_phys = 2 * r  # cylinder diameter in physical units
            D_latt = D_phys / (x_range[1] - x_range[0]) * self.nx  # convert to lattice units
        except (AttributeError, TypeError):
            # Fallback: estimate from obstacle
            D_latt = 2 * np.sqrt(np.count_nonzero(self.obstacle) / np.pi)

        # Kinematic viscosity from Reynolds number: Re = U*D/nu
        nu = self.u_inlet * D_latt / self.reynolds_number
        # BGK relaxation time: nu = cs^2 * (tau - 0.5), where cs^2 = 1/3
        self.tau = 3.0 * nu + 0.5

    def _equilibrium(self, rho, ux, uy):
        """
        Compute equilibrium distribution function for D2Q9 lattice.

        f_i^eq = w_i * rho * (1 + 3*(c_i·u)/cs² + 4.5*(c_i·u)²/cs⁴ - 1.5*u²/cs²)
        where cs² = 1/3
        """
        feq = np.zeros((self.nx, self.ny, 9), dtype=np.float64)
        _equilibrium_kernel(rho, ux, uy, self.c, self.w, feq)
        return feq

    def _inlet_zou_he(self, f, rho):
        """
        Apply Zou-He inlet boundary condition at x=0 with fixed velocity.

        Determines unknown populations (1, 5, 8) from known populations
        and prescribed inlet velocity.
        """
        rho_in = ((f[0, :, 0] + f[0, :, 2] + f[0, :, 4])
                  + 2.0 * (f[0, :, 3] + f[0, :, 6] + f[0, :, 7])
                 ) / (1.0 - self.u_inlet)

        f[0, :, 1] = f[0, :, 3] + (2.0/3.0) * rho_in * self.u_inlet
        f[0, :, 5] = (f[0, :, 7]
                      - 0.5 * (f[0, :, 2] - f[0, :, 4])
                      + (1.0/6.0) * rho_in * self.u_inlet)
        f[0, :, 8] = (f[0, :, 6]
                      + 0.5 * (f[0, :, 2] - f[0, :, 4])
                      + (1.0/6.0) * rho_in * self.u_inlet)

    def _outlet_open(self, f):
        """Apply open outlet boundary condition at x=Nx-1 (zero-gradient)."""
        _outlet_open_kernel(f)

    def _apply_inlet_boundary(self, f, u_inlet):
        """Apply selected inlet BC model."""
        if self.inlet_bc == "regularized":
            _inlet_regularized_kernel(f, u_inlet, self.c, self.w)
        else:
            _inlet_zou_he_kernel(f, u_inlet)

    def solve(self, verbose=True, visualizer=None, record_video=False, video_filename='lbm_simulation.mp4'):
        """
        Run the LBM simulation for Navier-Stokes equations.

        Parameters
        ----------
        verbose : bool, optional
            Print progress information (default: True)
        visualizer : FieldVisualizer, optional
            Visualizer for field display (from visualization module).
            If provided, enables live plotting. If None, no visualization.
        record_video : bool, optional
            Whether to record simulation to video file (default: False).
            Requires visualizer to be set.
        video_filename : str, optional
            Output video filename (default: 'lbm_simulation.mp4')

        Returns
        -------
        dict
            Dictionary containing final velocity fields (ux, uy), density (rho),
            and simulation metadata
        """
        # ===== Initialization =====
        if verbose:
            print(f"Initializing LBM solver...")
            print(f"  Grid: {self.nx} x {self.ny}")
            print(f"  Inlet velocity: {self.u_inlet}")
            print(f"  Inlet BC: {self.inlet_bc}")
            print(f"  Velocity ramp time constant: {self.velocity_ramp_tau:.1f} timesteps")
            print(f"  Reynolds number: {self.reynolds_number}")
            print(f"  Relaxation time (tau): {self.tau:.4f}")
            print(f"  Obstacle cells: {np.count_nonzero(self.obstacle)}")

        # Initialize macroscopic fields
        rho_init = np.ones((self.nx, self.ny), dtype=np.float64)
        y = np.arange(self.ny)
        profile = 4 * self.u_inlet * y / self.ny * (1 - y / self.ny)
        ux_init = np.tile(profile, (self.nx, 1)) #smooth initial parabolic profile
        uy_init = np.zeros((self.nx, self.ny), dtype=np.float64)

        # Add small transverse perturbation to break symmetry
        y = np.arange(self.ny)
        uy_init += 0.001 * self.u_inlet * np.sin(2.0 * np.pi * y / self.ny)

        # Zero velocity in obstacle
        ux_init[self.obstacle] = 0.0
        uy_init[self.obstacle] = 0.0

        # Initialize distributions to equilibrium
        f = self._equilibrium(rho_init, ux_init, uy_init)

        # Preallocate arrays reused in each timestep to reduce memory churn.
        rho = np.empty((self.nx, self.ny), dtype=np.float64)
        ux = np.empty((self.nx, self.ny), dtype=np.float64)
        uy = np.empty((self.nx, self.ny), dtype=np.float64)
        feq = np.empty((self.nx, self.ny, 9), dtype=np.float64)
        f_out = np.empty((self.nx, self.ny, 9), dtype=np.float64)

        # Setup visualization if requested
        sim_visualizer = None
        if visualizer is not None:
            from visualization import SimulationVisualizer
            sim_visualizer = SimulationVisualizer(
                self.nx, self.ny, self.obstacle,
                fps=30, record_video=record_video,
                video_filename=video_filename
            )
            sim_visualizer.setup(visualizer)

        # ===== Main Simulation Loop =====
        if verbose:
            print(f"\nRunning {self.n_steps} timesteps...\n")

        for step in range(1, self.n_steps + 1):
            # Compute ramped inlet velocity for smooth initialization
            u_inlet_ramped = _compute_ramped_velocity(step, self.u_inlet, self.velocity_ramp_tau)

            # 1. Compute macroscopic quantities
            _compute_macroscopic(f, self.c, rho, ux, uy)

            # 2. Collision step (BGK)
            _equilibrium_kernel(rho, ux, uy, self.c, self.w, feq)
            _collide_kernel(f, feq, self.tau, f_out)

            # 3. Bounce-back on obstacle (no-slip wall)
            _bounce_back_kernel(f, f_out, self.obstacle, self.opp)

            # 4. Streaming step
            _stream_kernel(f_out, self.c, f)

            # 5. Boundary conditions
            _outlet_open_kernel(f)
            self._apply_inlet_boundary(f, u_inlet_ramped)

            # Progress and visualization
            if verbose and step % 1000 == 0:
                avg_rho = np.mean(rho[~self.obstacle])
                print(f"  Step {step:>6d}/{self.n_steps}  |  avg density = {avg_rho:.6f}  |  u_inlet_ramped = {u_inlet_ramped:.6f}")

            if sim_visualizer is not None and step % self.vis_interval == 0:
                field = visualizer.compute_field(ux, uy, rho)
                sim_visualizer.update(field, step)

        # Finalize visualization
        if sim_visualizer is not None:
            sim_visualizer.finalize()

        if verbose:
            print("\nSimulation complete.")

        # ===== Return Results =====
        return {
            'ux': ux,
            'uy': uy,
            'rho': rho,
            'obstacle': self.obstacle,
            'metadata': {
                'nx': self.nx,
                'ny': self.ny,
                'u_inlet': self.u_inlet,
                'inlet_bc': self.inlet_bc,
                'reynolds_number': self.reynolds_number,
                'tau': self.tau,
                'n_steps': self.n_steps
            }
        }
