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
def _collide_kernel(f, feq, tau, f_out, alpha):
    nx, ny, _ = f.shape
    inv_tau = 1.0 / tau

    for x in range(nx):
        for y in range(ny):
            for i in range(9):
                fi = f[x, y, i]
                feq_i = feq[x, y, i]

                f_post = fi - (fi - feq_i) * inv_tau
                f_out[x, y, i] = feq_i + alpha * (f_post - feq_i)


@njit(cache=True)
def _collide_trt_kernel(f, feq, tau_plus, tau_minus, opp, f_out):
    nx, ny, _ = f.shape
    omega_plus = 1.0 / tau_plus
    omega_minus = 1.0 / tau_minus

    for x in range(nx):
        for y in range(ny):

            # Rest population
            fi = f[x, y, 0]
            feq_i = feq[x, y, 0]
            f_out[x, y, 0] = fi - omega_plus * (fi - feq_i)

            for i in range(1, 9):
                io = opp[i]
                if i < io:
                    fi = f[x, y, i]
                    fio = f[x, y, io]

                    feq_i = feq[x, y, i]
                    feq_io = feq[x, y, io]

                    f_plus = 0.5 * (fi + fio)
                    f_minus = 0.5 * (fi - fio)

                    feq_plus = 0.5 * (feq_i + feq_io)
                    feq_minus = 0.5 * (feq_i - feq_io)

                    f_plus_new = f_plus - omega_plus * (f_plus - feq_plus)
                    f_minus_new = f_minus - omega_minus * (f_minus - feq_minus)

                    f_out[x, y, i]  = f_plus_new + f_minus_new
                    f_out[x, y, io] = f_plus_new - f_minus_new


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

    # Start with a copy
    for x in range(nx):
        for y in range(ny):
            for i in range(9):
                f[x, y, i] = f_out[x, y, i]

    # Streaming
    for x in range(nx):
        for y in range(ny):
            for i in range(9):
                sx = x - c[i, 0]
                sy = y - c[i, 1]
                if 0 <= sx < nx and 0 <= sy < ny:
                    f[x, y, i] = f_out[sx, sy, i]


@njit(cache=True)
def _outlet_open_kernel(f):
    """Zero-gradient outlet (copy from previous layer)."""
    nx, ny, _ = f.shape
    x_out = nx - 1
    x_prev = nx - 2
    for y in range(ny):
        for i in range(9):
            f[x_out, y, i] = f[x_prev, y, i]


@njit(cache=True)
def _outlet_hybrid_kernel(f, c, w, rho_target):
    nx, ny, _ = f.shape
    x = nx - 1
    x_prev = nx - 2

    for y in range(ny):
        # --- extrapolate velocity ---
        rho_prev = 0.0
        ux_prev = 0.0
        uy_prev = 0.0

        for i in range(9):
            fi = f[x_prev, y, i]
            rho_prev += fi
            ux_prev += fi * c[i, 0]
            uy_prev += fi * c[i, 1]

        ux = ux_prev / rho_prev
        uy = uy_prev / rho_prev

        # --- enforce pressure ---
        rho = rho_target
        usqr = ux * ux + uy * uy

        # equilibrium
        feq = np.empty(9, dtype=np.float64)
        for i in range(9):
            cu = c[i, 0] * ux + c[i, 1] * uy
            feq[i] = w[i] * rho * (1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * usqr)

        # reconstruct incoming populations
        f[x, y, 3] = feq[3] + (f[x, y, 1] - feq[1])
        f[x, y, 6] = feq[6] + (f[x, y, 8] - feq[8])
        f[x, y, 7] = feq[7] + (f[x, y, 5] - feq[5])


@njit(cache=True)
def _apply_outlet_sponge_kernel(f_out, feq, sponge_start_x, sigma_max):
    """Damp non-equilibrium content near the outlet to reduce reflections."""
    nx, ny, _ = f_out.shape
    width = nx - sponge_start_x
    if width <= 0:
        return

    for x in range(sponge_start_x, nx):
        # Smooth ramp from weak damping at sponge start to sigma_max at outlet.
        xi = (x - sponge_start_x + 1) / width
        sigma = sigma_max * xi * xi
        if sigma > 1.0:
            sigma = 1.0
        inv = 1.0 - sigma

        for y in range(ny):
            for i in range(9):
                f_out[x, y, i] = inv * f_out[x, y, i] + sigma * feq[x, y, i]


def _compute_ramped_velocity(step, u_inlet, tau_ramp):
    if tau_ramp <= 0:
        return u_inlet
    return u_inlet * (1.0 - np.exp(-float(step) / tau_ramp))


def _compute_ramped_value(step, target, tau_ramp):
    if tau_ramp is None or tau_ramp <= 0:
        return target
    return target * (1.0 - np.exp(-float(step) / tau_ramp))


@njit(cache=True)
def _inlet_zou_he_kernel(f, u_inlet, c, w):
    _, ny, _ = f.shape
    for y in range(ny):
        rho_in = ((f[0, y, 0] + f[0, y, 2] + f[0, y, 4])
                  + 2.0 * (f[0, y, 3] + f[0, y, 6] + f[0, y, 7])) / (1.0 - u_inlet)
        ux = u_inlet
        uy = 0.0
        usqr = ux * ux + uy * uy

        feq = np.empty(9, dtype=np.float64)
        for i in range(9):
            cu = c[i, 0] * ux + c[i, 1] * uy
            feq[i] = w[i] * rho_in * (1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * usqr)

        f[0, y, 1] = feq[1] + (f[0, y, 3] - feq[3])
        f[0, y, 5] = feq[5] + (f[0, y, 7] - feq[7])
        f[0, y, 8] = feq[8] + (f[0, y, 6] - feq[6])


@njit(cache=True)
def _inlet_regularized_kernel(f, u_inlet, c, w):
    _, ny, _ = f.shape
    for y in range(ny):
        rho_in = ((f[0, y, 0] + f[0, y, 2] + f[0, y, 4])
                  + 2.0 * (f[0, y, 3] + f[0, y, 6] + f[0, y, 7])) / (1.0 - u_inlet)

        feq = np.empty(9, dtype=np.float64)
        for i in range(9):
            cu = c[i, 0] * u_inlet
            feq[i] = w[i] * rho_in * (1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * u_inlet * u_inlet)

        f[0, y, 1] = feq[1] + (f[0, y, 3] - feq[3])
        f[0, y, 5] = feq[5] + (f[0, y, 7] - feq[7])
        f[0, y, 8] = feq[8] + (f[0, y, 6] - feq[6])


@njit(cache=True)
def _bottom_bounce_back_wall_kernel(f):
    nx, _, _ = f.shape
    y = 0
    for x in range(nx):
        f[x, y, 2] = f[x, y, 4]
        f[x, y, 5] = f[x, y, 7]
        f[x, y, 6] = f[x, y, 8]


@njit(cache=True)
def _top_bounce_back_wall_kernel(f):
    nx, ny, _ = f.shape
    y = ny - 1
    for x in range(nx):
        f[x, y, 4] = f[x, y, 2]
        f[x, y, 7] = f[x, y, 5]
        f[x, y, 8] = f[x, y, 6]


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
                 inlet_bc="zou_he", outlet_bc="open",
                 top_bc="bounce_back", bottom_bc="bounce_back", alpha=1.0,
                 collision_model="bgk", tau_minus=None, trt_lambda=0.25,
                 bc_ramp_tau=None, use_outlet_sponge=False,
                 outlet_sponge_width=0, outlet_sponge_sigma_max=0.15):
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
        outlet_bc : str, optional
            Outlet boundary condition. Supported: "open" (zero-gradient, default)
            and "zou_he_pressure" (Zou-He pressure outlet).
        top_bc : str, optional
            Top boundary condition. Supported: "bounce_back" (default)
            or "periodic".
        bottom_bc : str, optional
            Bottom boundary condition. Supported: "bounce_back" (default)
            or "periodic".
        collision_model : str, optional
            Collision operator model: "bgk" (default) or "trt".
        tau_minus : float, optional
            Anti-symmetric relaxation time for TRT. If None, it is computed
            from trt_lambda and tau_plus (= tau).
        trt_lambda : float, optional
            TRT magic parameter such that
            (tau_plus - 0.5) * (tau_minus - 0.5) = trt_lambda.
        use_outlet_sponge : bool, optional
            Enable sponge damping layer near the outlet boundary.
        outlet_sponge_width : int, optional
            Width of sponge layer in lattice cells from the outlet.
            Set to 0 to disable unless use_outlet_sponge=True.
        outlet_sponge_sigma_max : float, optional
            Maximum sponge blending factor at outlet edge (0 to 1).
        """
        super().__init__(environment)
        self.nx = nx
        self.ny = ny
        self.u_inlet = u_inlet
        self.reynolds_number = reynolds_number
        self.n_steps = n_steps
        self.vis_interval = min(vis_interval, n_steps // 10)
        self.alpha = alpha
        # Default ramp time constant: 10% of total steps if not specified
        self.velocity_ramp_tau = velocity_ramp_tau if velocity_ramp_tau is not None else max(100, n_steps // 10)
        self.collision_model = collision_model.lower().replace("-", "_")
        self.tau_minus = tau_minus
        self.trt_lambda = trt_lambda
        self.neq_reflection_scale = 1.0
        # Optional separate ramp for boundary non-equilibrium scaling; defaults to velocity ramp
        self.bc_ramp_tau = bc_ramp_tau if bc_ramp_tau is not None else self.velocity_ramp_tau

        self.use_outlet_sponge = bool(use_outlet_sponge)
        self.outlet_sponge_width = int(outlet_sponge_width)
        self.outlet_sponge_sigma_max = float(outlet_sponge_sigma_max)

        inlet_bc_normalized = inlet_bc.lower().replace("-", "_")
        if inlet_bc_normalized not in ("zou_he", "regularized"):
            raise ValueError("inlet_bc must be 'zou_he' or 'regularized'.")
        self.inlet_bc = inlet_bc_normalized

        outlet_bc_normalized = outlet_bc.lower().replace("-", "_")
        if outlet_bc_normalized not in ("open", "zou_he_pressure"):
            raise ValueError("outlet_bc must be 'open' or 'zou_he_pressure'.")
        self.outlet_bc = outlet_bc_normalized

        # Keep backward compatibility: old wall modes map to bounce-back walls.
        # Periodic wraparound is disabled with bounded streaming.
        wall_bc_aliases = {
            "zou_he": "bounce_back",
            "regularized": "bounce_back",
            "periodic": "bounce_back",
        }
        self.top_bc = wall_bc_aliases.get(top_bc.lower().replace("-", "_"), top_bc.lower().replace("-", "_"))
        self.bottom_bc = wall_bc_aliases.get(bottom_bc.lower().replace("-", "_"), bottom_bc.lower().replace("-", "_"))

        allowed_wall_bcs = ("bounce_back",)
        if self.top_bc not in allowed_wall_bcs:
            raise ValueError("top_bc must be 'bounce_back'.")
        if self.bottom_bc not in allowed_wall_bcs:
            raise ValueError("bottom_bc must be 'bounce_back'.")

        if self.outlet_sponge_width < 0:
            raise ValueError("outlet_sponge_width must be >= 0.")
        if not (0.0 <= self.outlet_sponge_sigma_max <= 1.0):
            raise ValueError("outlet_sponge_sigma_max must be in [0, 1].")

        # Auto-enable sponge if width is specified.
        if self.outlet_sponge_width > 0:
            self.use_outlet_sponge = True

        if self.use_outlet_sponge and self.outlet_sponge_width == 0:
            raise ValueError("Set outlet_sponge_width > 0 when use_outlet_sponge=True.")
        if self.outlet_sponge_width >= self.nx:
            raise ValueError("outlet_sponge_width must be smaller than nx.")

        self.outlet_sponge_start_x = self.nx - self.outlet_sponge_width if self.use_outlet_sponge else self.nx

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
        self._configure_collision_model()

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

    def _configure_collision_model(self):
        """Validate and derive collision-model parameters."""
        if self.collision_model not in ("bgk", "trt"):
            raise ValueError("collision_model must be 'bgk' or 'trt'.")

        if self.collision_model == "trt":
            tau_plus = self.tau
            if tau_plus <= 0.5:
                raise ValueError("TRT requires tau > 0.5.")

            if self.tau_minus is None:
                self.tau_minus = 0.5 + self.trt_lambda / (tau_plus - 0.5)

            if self.tau_minus <= 0.5:
                raise ValueError("TRT requires tau_minus > 0.5.")

            # TRT-consistent boundary reconstruction scaling for non-equilibrium parts.
            self.neq_reflection_scale = (self.tau_minus - 0.5) / (self.tau - 0.5)
        else:
            self.tau_minus = None
            self.neq_reflection_scale = 1.0

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
            _inlet_zou_he_kernel(f, u_inlet, self.c, self.w)

    def _apply_outlet_boundary(self, f, rho_target=1.0):
        """Apply selected outlet BC model."""
        if self.outlet_bc == "zou_he_pressure":
            _outlet_hybrid_kernel(f, self.c, self.w, rho_target)
        else:
            _outlet_open_kernel(f)

    def _apply_top_bottom_boundaries(self, f):
        """Apply selected top and bottom wall BC models."""
        if self.top_bc == "bounce_back":
            _top_bounce_back_wall_kernel(f)

        if self.bottom_bc == "bounce_back":
            _bottom_bounce_back_wall_kernel(f)

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
            print(f"  Outlet BC: {self.outlet_bc}")
            print(f"  Top BC: {self.top_bc}")
            print(f"  Bottom BC: {self.bottom_bc}")
            print(f"  Collision model: {self.collision_model}")
            if self.collision_model == "trt":
                print(f"  TRT tau+: {self.tau:.4f}, tau-: {self.tau_minus:.4f}, lambda: {self.trt_lambda}")
                print(f"  TRT BC non-eq scale target: {self.neq_reflection_scale:.4f}")
                print(f"  BC ramp time constant: {self.bc_ramp_tau:.1f} timesteps")
            print(f"  Velocity ramp time constant: {self.velocity_ramp_tau:.1f} timesteps")
            print(f"  Reynolds number: {self.reynolds_number}")
            print(f"  Relaxation time (tau): {self.tau:.4f}")
            print(f"  Obstacle cells: {np.count_nonzero(self.obstacle)}")
            print(f"  Outlet sponge enabled: {self.use_outlet_sponge}")
            if self.use_outlet_sponge:
                print(f"  Outlet sponge width: {self.outlet_sponge_width}")
                print(f"  Outlet sponge sigma_max: {self.outlet_sponge_sigma_max:.3f}")

        # Initialize macroscopic fields
        rho_init = np.ones((self.nx, self.ny), dtype=np.float64)
        y = np.arange(self.ny)
        profile = 4 * self.u_inlet * y / self.ny * (1 - y / self.ny)

        # Only apply non-zero velocity in the first 5% of x grid points (inlet region)
        x_inlet_end = max(1, int(0.091 * self.nx))
        ux_init = np.zeros((self.nx, self.ny), dtype=np.float64)
        ux_init[:x_inlet_end, :] = np.tile(profile, (x_inlet_end, 1))
        uy_init = np.zeros((self.nx, self.ny), dtype=np.float64)

        # Add small transverse perturbation to break symmetry (only in inlet region)
        y = np.arange(self.ny)
        uy_init[:x_inlet_end, :] += 0.001 * self.u_inlet * np.sin(2.0 * np.pi * y / self.ny)

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
            # Ramp boundary non-equilibrium scale similarly
            if self.collision_model == "trt":
                bc_neq_scale = _compute_ramped_value(step, self.neq_reflection_scale, self.bc_ramp_tau)
            else:
                bc_neq_scale = 1.0

            # 1. Compute macroscopic quantities
            _compute_macroscopic(f, self.c, rho, ux, uy)

            # 2. Collision step
            _equilibrium_kernel(rho, ux, uy, self.c, self.w, feq)
            if self.collision_model == "trt":
                _collide_trt_kernel(f, feq, self.tau, self.tau_minus, self.opp, f_out)
            else:
                _collide_kernel(f, feq, self.tau, f_out, self.alpha)

            # 3. Bounce-back on obstacle (no-slip wall)
            _bounce_back_kernel(f, f_out, self.obstacle, self.opp)

            # Optional sponge layer near outlet to absorb outgoing disturbances.
            if self.use_outlet_sponge:
                _apply_outlet_sponge_kernel(
                    f_out,
                    feq,
                    self.outlet_sponge_start_x,
                    self.outlet_sponge_sigma_max,
                )

            # 4. Streaming step
            _stream_kernel(f_out, self.c, f)

            # 5. Boundary conditions
            # 5. Boundary conditions
            self._apply_outlet_boundary(f)

            if self.inlet_bc == "regularized":
                _inlet_regularized_kernel(f, u_inlet_ramped, self.c, self.w)
            else:
                _inlet_zou_he_kernel(f, u_inlet_ramped, self.c, self.w)

            self._apply_top_bottom_boundaries(f)

            # Progress and visualization
            if verbose and step % 1000 == 0:
                avg_rho = np.mean(rho[~self.obstacle])
                print(f"  Step {step:>6d}/{self.n_steps}  |  avg density = {avg_rho:.6f}  |  u_inlet_ramped = {u_inlet_ramped:.6f}  |  bc_neq_scale = {bc_neq_scale:.4f}")

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
                'outlet_bc': self.outlet_bc,
                'top_bc': self.top_bc,
                'bottom_bc': self.bottom_bc,
                'collision_model': self.collision_model,
                'tau_minus': self.tau_minus,
                'trt_lambda': self.trt_lambda,
                'neq_reflection_scale': self.neq_reflection_scale,
                'bc_ramp_tau': self.bc_ramp_tau,
                'use_outlet_sponge': self.use_outlet_sponge,
                'outlet_sponge_width': self.outlet_sponge_width,
                'outlet_sponge_sigma_max': self.outlet_sponge_sigma_max,
                'reynolds_number': self.reynolds_number,
                'tau': self.tau,
                'n_steps': self.n_steps
            }
        }
