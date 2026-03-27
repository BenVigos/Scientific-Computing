from abc import ABC, abstractmethod
import numpy as np

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
                 n_steps=30000):
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
        """
        super().__init__(environment)
        self.nx = nx
        self.ny = ny
        self.u_inlet = u_inlet
        self.reynolds_number = reynolds_number
        self.n_steps = n_steps

        # D2Q9 Lattice velocities
        self.c = np.array([[0, 0],    # 0 - rest
                           [1, 0],    # 1 - east
                           [0, 1],    # 2 - north
                           [-1, 0],   # 3 - west
                           [0, -1],   # 4 - south
                           [1, 1],    # 5 - north-east
                           [-1, 1],   # 6 - north-west
                           [-1, -1],  # 7 - south-west
                           [1, -1]])  # 8 - south-east

        # D2Q9 weights
        self.w = np.array([4/9,                        # rest
                           1/9, 1/9, 1/9, 1/9,         # axis-aligned
                           1/36, 1/36, 1/36, 1/36])    # diagonals

        # Opposite direction index for bounce-back
        self.opp = np.array([0, 3, 4, 1, 2, 7, 8, 5, 6])

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
        feq = np.zeros((self.nx, self.ny, 9))
        usqr = ux**2 + uy**2

        for i in range(9):
            cu = self.c[i, 0] * ux + self.c[i, 1] * uy
            feq[:, :, i] = (self.w[i] * rho *
                           (1.0 + 3.0 * cu + 4.5 * cu**2 - 1.5 * usqr))
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
        f[-1, :, :] = f[-2, :, :]


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
            print(f"  Reynolds number: {self.reynolds_number}")
            print(f"  Relaxation time (tau): {self.tau:.4f}")
            print(f"  Obstacle cells: {np.count_nonzero(self.obstacle)}")

        # Initialize macroscopic fields
        rho_init = np.ones((self.nx, self.ny))
        ux_init = np.full((self.nx, self.ny), self.u_inlet)
        uy_init = np.zeros((self.nx, self.ny))

        # Add small transverse perturbation to break symmetry
        y = np.arange(self.ny)
        uy_init += 0.001 * self.u_inlet * np.sin(2.0 * np.pi * y / self.ny)

        # Zero velocity in obstacle
        ux_init[self.obstacle] = 0.0
        uy_init[self.obstacle] = 0.0

        # Initialize distributions to equilibrium
        f = self._equilibrium(rho_init, ux_init, uy_init)

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
            # 1. Compute macroscopic quantities
            rho = np.sum(f, axis=2)
            ux = np.sum(f * self.c[:, 0], axis=2) / rho
            uy = np.sum(f * self.c[:, 1], axis=2) / rho

            # 2. Collision step (BGK)
            feq = self._equilibrium(rho, ux, uy)
            f_out = f - (f - feq) / self.tau

            # 3. Bounce-back on obstacle (no-slip wall)
            for i in range(9):
                f_out[self.obstacle, i] = f[self.obstacle, self.opp[i]]

            # 4. Streaming step
            for i in range(9):
                f[:, :, i] = np.roll(f_out[:, :, i], shift=self.c[i, 0], axis=0)
                f[:, :, i] = np.roll(f[:, :, i], shift=self.c[i, 1], axis=1)

            # 5. Boundary conditions
            self._outlet_open(f)
            self._inlet_zou_he(f, rho)

            # Progress and visualization
            if verbose and step % 1000 == 0:
                avg_rho = np.mean(rho[~self.obstacle])
                print(f"  Step {step:>6d}/{self.n_steps}  |  avg density = {avg_rho:.6f}")

            if sim_visualizer is not None and step % 100 == 0:
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
                'reynolds_number': self.reynolds_number,
                'tau': self.tau,
                'n_steps': self.n_steps
            }
        }
