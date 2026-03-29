from abc import ABC, abstractmethod
import numpy as np
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import bicgstab, cg, spilu, LinearOperator
from numba import njit, prange

try:
    import pyamg
    HAVE_PYAMG = True
except Exception:
    HAVE_PYAMG = False

try:
    import ngsolve as ngs
    from ngsolve import *
    from netgen.geom2d import SplineGeometry
    _NGSOLVE_AVAILABLE = True
except ImportError:
    _NGSOLVE_AVAILABLE = False


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

def _build_karman_geometry(env, global_maxh=0.08, cyl_maxh=0.02):
    """
    Build a fitted 2D channel-with-cylinder geometry for Netgen/NGSolve.

    Boundary names:
        inlet    : left boundary
        outlet   : right boundary
        wall     : top and bottom walls
        cylinder : cylinder surface
    """
    geo = SplineGeometry()

    x0, x1 = env.x_range
    y0, y1 = env.y_range
    cx, cy = env.circle_center
    r = env.circle_radius

    p0 = geo.AppendPoint(x0, y0)
    p1 = geo.AppendPoint(x1, y0)
    p2 = geo.AppendPoint(x1, y1)
    p3 = geo.AppendPoint(x0, y1)

    geo.Append(["line", p0, p1], bc="wall",   leftdomain=1, rightdomain=0, maxh=global_maxh)
    geo.Append(["line", p1, p2], bc="outlet", leftdomain=1, rightdomain=0, maxh=global_maxh)
    geo.Append(["line", p2, p3], bc="wall",   leftdomain=1, rightdomain=0, maxh=global_maxh)
    geo.Append(["line", p3, p0], bc="inlet",  leftdomain=1, rightdomain=0, maxh=global_maxh)

    geo.AddCircle(
        c=(cx, cy),
        r=r,
        bc="cylinder",
        leftdomain=0,
        rightdomain=1,
        maxh=cyl_maxh,
    )

    return geo

class Solver(ABC):
    """Abstract base class for PDE solvers"""

    def __init__(self, environment):
        self.environment = environment

    @abstractmethod
    def solve(self):
        pass


class FEMSolver(Solver):
    """
    Finite-Element incompressible Navier–Stokes solver using NGSolve.

    - Taylor–Hood Pk/P(k-1)
    - IMEX Euler:
          implicit: mass + diffusion + incompressibility (+ optional grad-div)
          explicit: convection
    - Constant implicit matrix assembled once
    - One inverse reused across all time steps
    - Curved mesh for the cylinder
    - Optional Stokes initialization
    """

    def __init__(
        self,
        environment,
        dt=1e-3,
        n_steps=8000,
        global_maxh=0.025,
        cyl_maxh=0.003,
        order=2,
        reynolds_number=None,
        viscosity=None,
        rho=1.0,
        graddiv_gamma=1e-3,
        inlet_profile="parabolic",
        inlet_perturbation=1e-3,
        ramp_time=0.0,
        stokes_start=True,
        curved_order=None,
        probe_point=(0.6, 0.21),
        num_threads=None,
        inverse_name="cg",
        preconditioner_name="amg",
        target_physical_time=None,
        verbose=True,
    ):
        if not _NGSOLVE_AVAILABLE:
            raise ImportError("NGSolve is not installed. Install it before using FEMSolver.")

        super().__init__(environment)

        self.environment = environment
        self.dt = float(dt)
        self.n_steps = int(n_steps)
        self.global_maxh = float(global_maxh)
        self.cyl_maxh = float(cyl_maxh)
        self.order = int(order)
        self.rho = float(rho)
        self.graddiv_gamma = float(graddiv_gamma)
        self.inlet_profile = str(inlet_profile).lower()
        self.inlet_perturbation = float(inlet_perturbation)
        self.ramp_time = float(ramp_time)
        self.stokes_start = bool(stokes_start)
        self.curved_order = int(curved_order if curved_order is not None else max(2, order))
        self.probe_point = tuple(probe_point)
        self.num_threads = num_threads
        self.inverse_name = str(inverse_name)
        self.preconditioner_name = str(preconditioner_name)
        self.verbose = bool(verbose)
        self.target_physical_time = (
            None if target_physical_time is None else float(target_physical_time)
        )
        if self.target_physical_time is not None and self.target_physical_time <= 0.0:
            raise ValueError("target_physical_time must be positive when provided.")

        self.v0 = float(environment.v0)

        diameter = 2.0 * environment.circle_radius
        if reynolds_number is not None:
            if reynolds_number <= 0:
                raise ValueError("reynolds_number must be positive.")
            self.reynolds_number = float(reynolds_number)
            self.nu = self.v0 * diameter / self.reynolds_number
            print(self.nu)
        elif viscosity is not None:
            if viscosity <= 0:
                raise ValueError("viscosity must be positive.")
            self.nu = float(viscosity)
            self.reynolds_number = self.v0 * diameter / self.nu
        else:
            self.nu = float(environment.viscosity)
            self.reynolds_number = self.v0 * diameter / max(self.nu, 1e-12)

        self.time = 0.0

        if self.num_threads is not None:
            ngs.SetNumThreads(int(self.num_threads))

        self._build_mesh()
        self._build_spaces()
        self._setup_gridfunctions()
        self._build_implicit_operator()
        self._setup_initial_condition()


    # Mesh 

    def _build_mesh(self):
        if self.verbose:
            print("  Building FEM mesh...")

        geo = _build_karman_geometry(
            self.environment,
            global_maxh=self.global_maxh,
            cyl_maxh=self.cyl_maxh,
        )

        with ngs.TaskManager():
            ngmesh = geo.GenerateMesh(maxh=self.global_maxh)

        self.mesh = Mesh(ngmesh)
        self.mesh.Curve(self.curved_order)

        if self.verbose:
            print(f"    Elements: {self.mesh.ne}, Vertices: {self.mesh.nv}, Curve order: {self.curved_order}")

    def _build_spaces(self):
        if self.order < 2:
            raise ValueError("Taylor–Hood needs order >= 2")

        dirichlet_velocity = "inlet|wall|cylinder"

        self.V = VectorH1(self.mesh, order=self.order, dirichlet=dirichlet_velocity)
        self.Q = H1(self.mesh, order=self.order - 1)
        self.X = FESpace([self.V, self.Q])

        (self.u, self.p), (self.v, self.q) = self.X.TnT()

        self.freedofs = self.X.FreeDofs()

        # pin one pressure dof
        prange = self.X.Range(1)
        self.pressure_pin_dof = None
        for dof in range(prange.start, prange.stop):
            if self.freedofs[dof]:
                self.freedofs[dof] = False
                self.pressure_pin_dof = dof
                break

        if self.pressure_pin_dof is None:
            raise RuntimeError("Could not find a free pressure dof to pin.")

        if self.verbose:
            print(f"  Velocity DOFs : {self.V.ndof}")
            print(f"  Pressure DOFs : {self.Q.ndof}")
            print(f"  Mixed DOFs    : {self.X.ndof}")
            print(f"  Pinned pressure dof: {self.pressure_pin_dof}")

    def _setup_gridfunctions(self):
        # current mixed solution
        self.gfu = GridFunction(self.X)
        self.gfu_u, self.gfu_p = self.gfu.components

        # previous velocity only
        self.gfu_prev = GridFunction(self.V)

        # mixed lifting for inhomogeneous Dirichlet data
        self.gfd = GridFunction(self.X)
        self.gfd_u, self.gfd_p = self.gfd.components

        # temporary vectors/forms
        self.rhs = LinearForm(self.X)
        self.conv = LinearForm(self.X)

    
    # Inlet 


    def _ramp_factor(self, time=None):
        if time is None:
            time = self.time
        if self.ramp_time <= 0.0:
            return 1.0
        if time <= 0.0:
            return 0.0
        if time >= self.ramp_time:
            return 1.0
        return 1.0 - np.exp(-time / self.ramp_time)

    def _inlet_cf(self, time=None, full_strength=False):
        """
        Inlet profile on the left boundary.
        """
        if time is None:
            time = self.time

        ramp = 1.0 if full_strength else self._ramp_factor(time)

        y0, y1 = self.environment.y_range
        H = y1 - y0
        yy = ngs.y - y0

        if self.inlet_profile == "parabolic":
            ux_base = 4.0 * self.v0 * yy * (H - yy) / (H * H)
        elif self.inlet_profile == "plug":
            ux_base = self.v0
        else:
            raise ValueError("inlet_profile must be 'parabolic' or 'plug'")

        ux = ramp * ux_base

        if self.inlet_perturbation != 0.0:
            uy = ramp * self.inlet_perturbation * self.v0 * ngs.sin(2.0 * ngs.pi * yy / H)
        else:
            uy = 0.0

        return CoefficientFunction((ux, uy))

    def _assemble_lifting(self, time_for_bc):
        """
        Fill self.gfd with the current inhomogeneous velocity Dirichlet data.
        """
        self.gfd.vec[:] = 0.0
        self.gfd_u.Set(CoefficientFunction((0.0, 0.0)), definedon=self.mesh.Boundaries("wall|cylinder"))
        self.gfd_u.Set(self._inlet_cf(time=time_for_bc), definedon=self.mesh.Boundaries("inlet"))
        self.gfd_p.vec[:] = 0.0

   
    # Implicit operator

    def _build_implicit_operator(self):
        """
        Build A* for IMEX:
            (1/dt) M + nu K + saddle + grad-div
        This matrix is constant and reused every step.
        """
        self.astar = BilinearForm(self.X, symmetric=False)

        self.astar += (self.rho / self.dt) * InnerProduct(self.u, self.v) * dx
        self.astar += self.nu * InnerProduct(grad(self.u), grad(self.v)) * dx
        self.astar += -self.p * div(self.v) * dx
        self.astar += div(self.u) * self.q * dx

        if self.graddiv_gamma > 0.0:
            self.astar += self.graddiv_gamma * div(self.u) * div(self.v) * dx

        with ngs.TaskManager():
            self.astar.Assemble()

        if self.verbose:
            print("  Building constant implicit inverse...")

        # Use CG solver with AMG preconditioner
        self.inv_astar = self.astar.mat.Inverse(
            freedofs=self.freedofs,
            inverse="pardiso",
        )

  
    # Initialization
    def _solve_stokes_initial_state(self):
        """
        Solve a steady Stokes problem with full inlet profile.
        Used only if stokes_start=True and ramp_time == 0.
        """
        a0 = BilinearForm(self.X, symmetric=False)
        a0 += self.nu * InnerProduct(grad(self.u), grad(self.v)) * dx
        a0 += -self.p * div(self.v) * dx
        a0 += div(self.u) * self.q * dx

        if self.graddiv_gamma > 0.0:
            a0 += self.graddiv_gamma * div(self.u) * div(self.v) * dx

        f0 = LinearForm(self.X)

        self.gfd.vec[:] = 0.0
        self.gfd_u.Set(CoefficientFunction((0.0, 0.0)), definedon=self.mesh.Boundaries("wall|cylinder"))
        self.gfd_u.Set(self._inlet_cf(full_strength=True), definedon=self.mesh.Boundaries("inlet"))
        self.gfd_p.vec[:] = 0.0

        with ngs.TaskManager():
            a0.Assemble()
            f0.Assemble()

        f0.vec.data -= a0.mat * self.gfd.vec

        sol = GridFunction(self.X)
        sol.vec.data = a0.mat.Inverse(
            freedofs=self.freedofs,
            inverse="pardiso",
        ) * f0.vec
        sol.vec.data += self.gfd.vec
        sol.vec[self.pressure_pin_dof] = 0.0

        self.gfu.vec.data = sol.vec
        self.gfu_prev.vec.data = self.gfu_u.vec

    def _setup_initial_condition(self):
        """
        Recommended:
        - if ramp_time == 0 and stokes_start=True: steady Stokes start
        - otherwise: zero start and let ramp introduce flow smoothly
        """
        self.gfu.vec[:] = 0.0
        self.gfu_prev.vec[:] = 0.0

        if self.stokes_start and self.ramp_time <= 0.0:
            if self.verbose:
                print("  Computing Stokes initial condition...")
            self._solve_stokes_initial_state()
        else:
            # zero start is consistent with ramped inflow
            self.gfu.vec[:] = 0.0
            self.gfu_prev.vec[:] = 0.0
            self.gfu_p.vec[:] = 0.0

    # Time step
    def _assemble_rhs(self):
        """
        IMEX RHS:
            (rho/dt) (u^n, v) - rho * ((u^n · ∇)u^n, v)
        Then subtract A* g_D^{n+1} for inhomogeneous Dirichlet lifting.
        """
        self.rhs = LinearForm(self.X)
        self.rhs += (self.rho / self.dt) * InnerProduct(self.gfu_prev, self.v) * dx

        # explicit convection
        self.rhs += -self.rho * InnerProduct(grad(self.gfu_prev) * self.gfu_prev, self.v) * dx

        with ngs.TaskManager():
            self.rhs.Assemble()

        # next-step boundary data
        self._assemble_lifting(time_for_bc=self.time + self.dt)

        self.rhs.vec.data -= self.astar.mat * self.gfd.vec

    def _single_step(self):
        self._assemble_rhs()

        sol = GridFunction(self.X)
        sol.vec.data = self.inv_astar * self.rhs.vec
        sol.vec.data += self.gfd.vec
        sol.vec[self.pressure_pin_dof] = 0.0

        self.gfu.vec.data = sol.vec
        self.gfu_prev.vec.data = self.gfu_u.vec
        self.time += self.dt

    def _target_time_reached(self):
        if self.target_physical_time is None:
            return False
        return self.time >= self.target_physical_time - 1e-14

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def _sample_point_velocity(self, x, y):
        try:
            mip = self.mesh(x, y)
            val = self.gfu_u(mip)
            return float(val[0]), float(val[1])
        except Exception:
            return np.nan, np.nan

    def _cheap_diagnostics(self):
        """
        Cheap FE-native diagnostics.
        Avoid dense Cartesian export except when needed for visualization/final output.
        """
        div_l2 = float(np.sqrt(ngs.Integrate(div(self.gfu_u) * div(self.gfu_u) * dx, self.mesh)))
        probe_ux, probe_uy = self._sample_point_velocity(*self.probe_point)

        return {
            "time": self.time,
            "div_l2": div_l2,
            "probe_ux": probe_ux,
            "probe_uy": probe_uy,
        }


    def _to_structured_grid(self, nx, ny):
        x_lin = np.linspace(*self.environment.x_range, nx)
        y_lin = np.linspace(*self.environment.y_range, ny)
        X, Y = np.meshgrid(x_lin, y_lin, indexing="ij")

        ux = np.full((nx, ny), np.nan)
        uy = np.full((nx, ny), np.nan)
        p = np.full((nx, ny), np.nan)

        cx, cy = self.environment.circle_center
        r = self.environment.circle_radius
        obstacle = (X - cx) ** 2 + (Y - cy) ** 2 <= r ** 2

        ptsx = X[~obstacle].ravel()
        ptsy = Y[~obstacle].ravel()

        if ptsx.size > 0:
            mip = self.mesh(ptsx, ptsy)
            vel = self.gfu_u(mip)
            ux_vals = np.array(vel[:, 0]).ravel()
            uy_vals = np.array(vel[:, 1]).ravel()
            p_vals = np.array(self.gfu_p(mip)).ravel()

            fluid_idx = np.where((~obstacle).ravel())[0]
            ux.ravel()[fluid_idx] = ux_vals
            uy.ravel()[fluid_idx] = uy_vals
            p.ravel()[fluid_idx] = p_vals

        return ux, uy, p, obstacle

    # SOlve

    def solve(
        self,
        verbose=None,
        visualizer=None,
        record_video=False,
        video_filename="fem_simulation.mp4",
        visualize_every=100,
        history_every=100,
        return_history=True,
        export_nx=301,
        export_ny=121,
    ):
        if verbose is not None:
            self.verbose = bool(verbose)

        if self.verbose:
            print("=" * 68)
            print(f"FEM Solver - NGSolve IMEX Taylor–Hood P{self.order}/P{self.order-1}")
            print("=" * 68)
            print(f"  Mesh elements     : {self.mesh.ne}")
            print(f"  Velocity DOFs     : {self.V.ndof}")
            print(f"  Pressure DOFs     : {self.Q.ndof}")
            print(f"  dt                : {self.dt:.4e}")
            print(f"  n_steps           : {self.n_steps}")
            print(f"  nu                : {self.nu:.6e}")
            print(f"  Reynolds number   : {self.reynolds_number:.3f}")
            print(f"  Linear solver     : {self.inverse_name}")
            print(f"  Inlet profile     : {self.inlet_profile}")
            print(f"  Inlet perturb.    : {self.inlet_perturbation}")
            print(f"  Ramp time         : {self.ramp_time}")
            print(f"  Stokes start      : {self.stokes_start}")
            print(f"  Grad-div gamma    : {self.graddiv_gamma}")
            print(f"  Probe point       : {self.probe_point}")
            print(f"  Threads           : {self.num_threads}")
            if self.target_physical_time is not None:
                print(f"  Target physical t : {self.target_physical_time}")
            print()

        sim_visualizer = None
        if visualizer is not None:
            from visualization import SimulationVisualizer

            ux0, uy0, _, obstacle0 = self._to_structured_grid(export_nx, export_ny)
            sim_visualizer = SimulationVisualizer(
                export_nx,
                export_ny,
                obstacle0,
                fps=30,
                record_video=record_video,
                video_filename=video_filename,
            )
            sim_visualizer.setup(visualizer)

        history = {
            "time": [],
            "div_l2": [],
            "probe_ux": [],
            "probe_uy": [],
            "u_max_sampled": [],
        }

        last_u_max_sampled = np.nan
        stop_reason = "n_steps"
        steps_executed = 0

        for step in range(1, self.n_steps + 1):
            if self._target_time_reached():
                stop_reason = "target_physical_time"
                break

            self._single_step()
            steps_executed = step
            hit_target_time = self._target_time_reached()
            is_last_step = (step == self.n_steps) or hit_target_time

            do_history = return_history and (step % history_every == 0 or step == 1 or is_last_step)
            do_visual = sim_visualizer is not None and (step % visualize_every == 0 or is_last_step)
            do_print = self.verbose and (step % history_every == 0 or step == 1 or is_last_step)

            if do_visual:
                ux, uy, p, _ = self._to_structured_grid(export_nx, export_ny)
                field = visualizer.compute_field(ux, uy, p)
                sim_visualizer.update(field, step)

                speed = np.sqrt(ux**2 + uy**2)
                last_u_max_sampled = float(np.nanmax(speed))

            if do_history or do_print:
                diag = self._cheap_diagnostics()

                if np.isnan(last_u_max_sampled):
                    # only sample a smaller grid occasionally if we have not visualized recently
                    ux_s, uy_s, _, _ = self._to_structured_grid(min(export_nx, 151), min(export_ny, 61))
                    speed_s = np.sqrt(ux_s**2 + uy_s**2)
                    last_u_max_sampled = float(np.nanmax(speed_s))

                if do_history:
                    history["time"].append(diag["time"])
                    history["div_l2"].append(diag["div_l2"])
                    history["probe_ux"].append(diag["probe_ux"])
                    history["probe_uy"].append(diag["probe_uy"])
                    history["u_max_sampled"].append(last_u_max_sampled)

                if do_print:
                    print(
                        f"step={step:6d}/{self.n_steps}  "
                        f"t={diag['time']:.5f}  "
                        f"|u|max~={last_u_max_sampled:.4f}  "
                        f"||div u||_L2={diag['div_l2']:.3e}  "
                        f"probe_uy={diag['probe_uy']:.4e}"
                    )

                last_u_max_sampled = np.nan

            if hit_target_time:
                stop_reason = "target_physical_time"
                break

        if sim_visualizer is not None:
            sim_visualizer.finalize()

        ux, uy, p, obstacle = self._to_structured_grid(export_nx, export_ny)

        result = {
            "ux": ux,
            "uy": uy,
            "p": p,
            "obstacle": obstacle,
            "fluid": ~obstacle,
            "metadata": {
                "nx": export_nx,
                "ny": export_ny,
                "mesh_elements": self.mesh.ne,
                "velocity_dofs": self.V.ndof,
                "pressure_dofs": self.Q.ndof,
                "dt": self.dt,
                "n_steps": self.n_steps,
                "n_steps_executed": steps_executed,
                "stop_reason": stop_reason,
                "target_physical_time": self.target_physical_time,
                "time_final": self.time,
                "nu": self.nu,
                "rho": self.rho,
                "u_inlet": self.v0,
                "reynolds_number": self.reynolds_number,
                "element_type": f"Taylor-Hood P{self.order}/P{self.order-1}",
                "time_scheme": "IMEX Euler (explicit convection, implicit Stokes part)",
                "pressure_nullspace_fix": f"pressure dof {self.pressure_pin_dof} pinned",
                "linear_solver": self.inverse_name,
                "global_maxh": self.global_maxh,
                "cyl_maxh": self.cyl_maxh,
                "curved_order": self.curved_order,
                "inlet_profile": self.inlet_profile,
                "inlet_perturbation": self.inlet_perturbation,
                "ramp_time": self.ramp_time,
                "stokes_start": self.stokes_start,
                "graddiv_gamma": self.graddiv_gamma,
                "probe_point": self.probe_point,
                "history_every": history_every,
                "visualize_every": visualize_every,
            },
        }

        if return_history:
            result["history"] = history

        return result


class FDMSolver(Solver):
    """
    Finite-Difference incompressible Navier–Stokes solver on a Cartesian grid.

    This version uses a standard pressure-projection structure:
        1. Build tentative velocity field (u*, v*) without pressure correction
        2. Solve pressure Poisson equation: -Δp = -(rho/dt) div(u*)
        3. Correct velocity with the pressure gradient
        4. Re-apply boundary conditions

    Pressure solve options:
        - "bicgstab" : BiCGSTAB sparse Krylov solve
        - "cg"       : Conjugate Gradient (use only if matrix behaves SPD)
        - "amg"      : Algebraic multigrid via pyamg (if installed)

    The solver can be controlled either by:
        - explicit viscosity viscosity=...
        - target Reynolds number reynolds_number=...
    If both are omitted, it falls back to environment.viscosity.

    """

    def __init__(
        self,
        environment,
        nx=301,
        ny=121,
        dt=1e-4,
        n_steps=5000,
        rho=1.0,
        adaptive_dt=True,
        cfl_safety=0.20,
        diff_safety=0.20,
        reynolds_number=None,
        viscosity=None,
        poisson_method="bicgstab",
        pressure_tol=1e-8,
        pressure_maxiter=400,
        use_preconditioner=True,
        ilu_drop_tol=1e-4,
        ilu_fill_factor=10.0,
        convection_order="second",
        outlet_bc="zero_gradient",
        outlet_convection_speed=None,
        inlet_profile="parabolic",
        inlet_perturbation=1e-3,
        velocity_ramp_tau=None,
        inlet_strip_fraction=0.091,
        target_physical_time=None,
    ):
        """
        Initialize the finite-difference solver.

        Parameters
        ----------
        environment : Environment
            Problem definition, expected to provide build_condition_masks().
        nx, ny : int
            Number of grid points in x and y.
        dt : float
            Maximum time step. If adaptive_dt=True, the actual step is capped by
            CFL and diffusive stability limits.
        n_steps : int
            Number of outer timesteps.
        rho : float
            Fluid density.
        adaptive_dt : bool
            Whether to adapt dt based on current velocity magnitude.
        cfl_safety : float
            Safety factor for convective CFL.
        diff_safety : float
            Safety factor for explicit diffusion stability.
        reynolds_number : float or None
            Target Reynolds number. If provided, overrides viscosity.
        viscosity : float or None
            Explicit viscosity override. Used only if reynolds_number is None.
        poisson_method : {"bicgstab", "cg", "amg"}
            Sparse pressure solver to use.
        pressure_tol : float
            Relative tolerance passed to the sparse pressure solve.
        pressure_maxiter : int
            Maximum iterations for the sparse pressure solve.
        use_preconditioner : bool
            Whether to build/use an ILU preconditioner for bicgstab/cg.
        ilu_drop_tol : float
            ILU drop tolerance.
        ilu_fill_factor : float
            ILU fill factor.
        convection_order : {"first", "second"}
            Upwind order used in convection term.
            "second" uses second-order upwind with first-order fallback.
        outlet_bc : {"zero_gradient", "convective"}
            Velocity outlet boundary condition.
        outlet_convection_speed : float or None
            Convection speed used when outlet_bc="convective".
            If None, uses environment.v0.
        """
        super().__init__(environment)

        self.nx = int(nx)
        self.ny = int(ny)
        self.dt = float(dt)
        self.n_steps = int(n_steps)
        self.rho = float(rho)
        self.adaptive_dt = bool(adaptive_dt)
        self.cfl_safety = float(cfl_safety)
        self.diff_safety = float(diff_safety)
        self.poisson_method = str(poisson_method).lower()
        self.pressure_tol = float(pressure_tol)
        self.pressure_maxiter = int(pressure_maxiter)
        self.use_preconditioner = bool(use_preconditioner)
        self.ilu_drop_tol = float(ilu_drop_tol)
        self.ilu_fill_factor = float(ilu_fill_factor)
        self.inlet_profile = str(inlet_profile).lower()
        self.inlet_perturbation = float(inlet_perturbation)
        self.velocity_ramp_tau = (
            float(velocity_ramp_tau)
            if velocity_ramp_tau is not None
            else max(100.0 * self.dt, 0.1)
        )
        self.inlet_strip_fraction = float(inlet_strip_fraction)
        self.target_physical_time = (
            None if target_physical_time is None else float(target_physical_time)
        )
        if self.target_physical_time is not None and self.target_physical_time <= 0.0:
            raise ValueError("target_physical_time must be positive when provided.")

        self.convection_order = str(convection_order).lower()
        if self.convection_order not in ("first", "second"):
            raise ValueError("convection_order must be 'first' or 'second'.")

        self.outlet_bc = str(outlet_bc).lower()
        if self.outlet_bc not in ("zero_gradient", "convective"):
            raise ValueError("outlet_bc must be 'zero_gradient' or 'convective'.")

        self.outlet_convection_speed = outlet_convection_speed

        # Grid, masks, initial conditions
        data = self.environment.build_condition_masks(nx=self.nx, ny=self.ny, t=0.0)

        self.x = data["x"]
        self.y = data["y"]
        self.X = data["X"]
        self.Y = data["Y"]
        self.dx = float(data["dx"])
        self.dy = float(data["dy"])

        self.bc = data["bc"]
        self.masks = data["masks"]

        self.u = data["u0"].copy()
        self.v = data["v0"].copy()

        # Tentative velocities
        self.ut = np.zeros_like(self.u)
        self.vt = np.zeros_like(self.v)

        # Pressure and RHS
        self.p = np.zeros_like(self.u)
        self.b_full = np.zeros_like(self.u)

        self.time = 0.0

        # Reynolds number / viscosity handling
        diameter = 2.0 * self.environment.circle_radius
        inlet_speed = float(self.environment.v0)

        if reynolds_number is not None:
            if reynolds_number <= 0:
                raise ValueError("reynolds_number must be positive.")
            self.reynolds_number = float(reynolds_number)
            self.nu = inlet_speed * diameter / self.reynolds_number
        elif viscosity is not None:
            if viscosity <= 0:
                raise ValueError("viscosity must be positive.")
            self.nu = float(viscosity)
            self.reynolds_number = inlet_speed * diameter / self.nu
        else:
            self.nu = float(self.environment.viscosity)
            self.reynolds_number = inlet_speed * diameter / max(self.nu, 1e-12)

        # Pressure unknown mapping and sparse operator
        (
            self.active_pressure_mask,
            self.dirichlet_pressure_mask,
            self.pressure_id,
            self.n_pressure_unknowns,
        ) = self._build_pressure_indexing()

        self.A_pressure = self._build_pressure_laplacian()

        # Optional sparse preconditioner
        self.M_pressure = None
        if self.poisson_method in ("bicgstab", "cg") and self.use_preconditioner:
            self.M_pressure = self._build_ilu_preconditioner(self.A_pressure)

        # Optional AMG hierarchy
        self.amg_solver = None
        if self.poisson_method == "amg":
            if not HAVE_PYAMG:
                raise ImportError(
                    "poisson_method='amg' requested, but pyamg is not installed."
                )
            self.amg_solver = pyamg.smoothed_aggregation_solver(self.A_pressure)

        # Force consistent initial state
        self._apply_velocity_bc(self.u, self.v, dt=self.dt)
        self._apply_pressure_bc(self.p)

    def _ramped_inlet_speed(self, time):
        if self.velocity_ramp_tau is None or self.velocity_ramp_tau <= 0.0:
            return self.environment.v0
        return self.environment.v0 * (1.0 - np.exp(-time / self.velocity_ramp_tau))

    def _inlet_profile_values(self, time):
        """
        LBM-style inlet:
        - parabolic or plug ux
        - small sinusoidal uy perturbation
        - exponential ramp in time
        """
        U = self._ramped_inlet_speed(time)

        y0, y1 = self.environment.y_range
        H = y1 - y0
        yy = self.y - y0

        if self.inlet_profile == "parabolic":
            ux = 4.0 * U * yy * (H - yy) / (H * H)
        elif self.inlet_profile == "plug":
            ux = np.full_like(self.y, U, dtype=float)
        else:
            raise ValueError("inlet_profile must be 'parabolic' or 'plug'.")

        uy = self.inlet_perturbation * U * np.sin(2.0 * np.pi * yy / H)
        return ux, uy
    
    
    # INDEXING / MATRIX ASSEMBLY

    def _build_pressure_indexing(self):
        """
        Build pressure unknown mapping.

        Unknowns are pressure values on:
            - fluid cells
            - excluding obstacle cells
            - excluding outlet cells with Dirichlet p = 0
        """
        fluid = self.masks["fluid"]
        obstacle = self.masks["obstacle"]
        right = self.masks["right"]

        # Dirichlet pressure anchor at outlet
        dirichlet_pressure_mask = fluid & right & (~obstacle)

        # Active unknowns: all fluid cells except outlet Dirichlet and obstacle
        active_pressure_mask = fluid & (~obstacle) & (~dirichlet_pressure_mask)

        pressure_id = -np.ones((self.nx, self.ny), dtype=np.int64)
        counter = 0
        for i in range(self.nx):
            for j in range(self.ny):
                if active_pressure_mask[i, j]:
                    pressure_id[i, j] = counter
                    counter += 1

        return active_pressure_mask, dirichlet_pressure_mask, pressure_id, counter

    def _build_pressure_laplacian(self):
        """
        Assemble sparse matrix for:
            -Δ p = rhs

        Boundary treatment:
            - left boundary (inlet):   dp/dx = 0
            - top/bottom walls:        dp/dy = 0
            - obstacle:                dp/dn = 0
            - right boundary:          p = 0
        """
        nx, ny = self.nx, self.ny
        dx2 = self.dx * self.dx
        dy2 = self.dy * self.dy

        A = lil_matrix((self.n_pressure_unknowns, self.n_pressure_unknowns))

        active = self.active_pressure_mask
        dirichlet = self.dirichlet_pressure_mask
        obstacle = self.masks["obstacle"]

        def coeff(di, dj):
            return 1.0 / dx2 if di != 0 else 1.0 / dy2

        for i in range(nx):
            for j in range(ny):
                if not active[i, j]:
                    continue

                row = self.pressure_id[i, j]
                diag = 0.0

                for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    ni = i + di
                    nj = j + dj
                    c = coeff(di, dj)

                    # Outside domain -> Neumann
                    if ni < 0 or ni >= nx or nj < 0 or nj >= ny:
                        continue

                    # Obstacle neighbor -> zero-normal-gradient
                    if obstacle[ni, nj]:
                        continue

                    # Outlet Dirichlet p = 0
                    if dirichlet[ni, nj]:
                        diag += c
                        continue

                    # Normal active neighbor
                    if active[ni, nj]:
                        diag += c
                        col = self.pressure_id[ni, nj]
                        A[row, col] += -c
                        continue

                if diag <= 0.0:
                    diag = 1.0
                A[row, row] += diag

        return csr_matrix(A)

    def _build_ilu_preconditioner(self, A):
        """
        Build ILU preconditioner for Krylov solves.
        """
        try:
            ilu = spilu(
                A.tocsc(),
                drop_tol=self.ilu_drop_tol,
                fill_factor=self.ilu_fill_factor,
            )

            def mv(x):
                return ilu.solve(x)

            return LinearOperator(A.shape, matvec=mv)
        except Exception:
            return None

    # NUMBA KERNELS
    @staticmethod
    @njit(cache=True, parallel=True)
    def _predictor_step_numba(
        ut,
        vt,
        u,
        v,
        dx,
        dy,
        dt,
        nu,
        fluid_mask,
        obstacle_mask,
        fx,
        fy,
        use_second_order,
    ):
        """
        Compute tentative velocity field (u*, v*) without pressure correction.

        Convection:
            - first-order upwind if use_second_order == False
            - second-order upwind with first-order fallback otherwise
        Diffusion:
            - second-order central differences
        """
        nx, ny = u.shape

        # copy old state first
        for i in prange(nx):
            for j in range(ny):
                ut[i, j] = u[i, j]
                vt[i, j] = v[i, j]

        for i in prange(1, nx - 1):
            for j in range(1, ny - 1):
                if (not fluid_mask[i, j]) or obstacle_mask[i, j]:
                    ut[i, j] = 0.0
                    vt[i, j] = 0.0
                    continue

                # Convection derivatives
                # x-derivatives use sign of u[i,j]
                # y-derivatives use sign of v[i,j]

                if use_second_order:
                    # dudx and dvdx: second-order upwind in x
                    if u[i, j] >= 0.0:
                        if i >= 2 and (not obstacle_mask[i - 1, j]) and (not obstacle_mask[i - 2, j]):
                            dudx = (3.0 * u[i, j] - 4.0 * u[i - 1, j] + u[i - 2, j]) / (2.0 * dx)
                            dvdx = (3.0 * v[i, j] - 4.0 * v[i - 1, j] + v[i - 2, j]) / (2.0 * dx)
                        else:
                            dudx = (u[i, j] - u[i - 1, j]) / dx
                            dvdx = (v[i, j] - v[i - 1, j]) / dx
                    else:
                        if i <= nx - 3 and (not obstacle_mask[i + 1, j]) and (not obstacle_mask[i + 2, j]):
                            dudx = (-3.0 * u[i, j] + 4.0 * u[i + 1, j] - u[i + 2, j]) / (2.0 * dx)
                            dvdx = (-3.0 * v[i, j] + 4.0 * v[i + 1, j] - v[i + 2, j]) / (2.0 * dx)
                        else:
                            dudx = (u[i + 1, j] - u[i, j]) / dx
                            dvdx = (v[i + 1, j] - v[i, j]) / dx

                    # dudy and dvdy: second-order upwind in y
                    if v[i, j] >= 0.0:
                        if j >= 2 and (not obstacle_mask[i, j - 1]) and (not obstacle_mask[i, j - 2]):
                            dudy = (3.0 * u[i, j] - 4.0 * u[i, j - 1] + u[i, j - 2]) / (2.0 * dy)
                            dvdy = (3.0 * v[i, j] - 4.0 * v[i, j - 1] + v[i, j - 2]) / (2.0 * dy)
                        else:
                            dudy = (u[i, j] - u[i, j - 1]) / dy
                            dvdy = (v[i, j] - v[i, j - 1]) / dy
                    else:
                        if j <= ny - 3 and (not obstacle_mask[i, j + 1]) and (not obstacle_mask[i, j + 2]):
                            dudy = (-3.0 * u[i, j] + 4.0 * u[i, j + 1] - u[i, j + 2]) / (2.0 * dy)
                            dvdy = (-3.0 * v[i, j] + 4.0 * v[i, j + 1] - v[i, j + 2]) / (2.0 * dy)
                        else:
                            dudy = (u[i, j + 1] - u[i, j]) / dy
                            dvdy = (v[i, j + 1] - v[i, j]) / dy

                else:
                    # first-order upwind everywhere
                    if u[i, j] >= 0.0:
                        dudx = (u[i, j] - u[i - 1, j]) / dx
                        dvdx = (v[i, j] - v[i - 1, j]) / dx
                    else:
                        dudx = (u[i + 1, j] - u[i, j]) / dx
                        dvdx = (v[i + 1, j] - v[i, j]) / dx

                    if v[i, j] >= 0.0:
                        dudy = (u[i, j] - u[i, j - 1]) / dy
                        dvdy = (v[i, j] - v[i, j - 1]) / dy
                    else:
                        dudy = (u[i, j + 1] - u[i, j]) / dy
                        dvdy = (v[i, j + 1] - v[i, j]) / dy

                # Diffusion: second-order central differences
                lap_u = (
                    (u[i + 1, j] - 2.0 * u[i, j] + u[i - 1, j]) / (dx * dx)
                    + (u[i, j + 1] - 2.0 * u[i, j] + u[i, j - 1]) / (dy * dy)
                )
                lap_v = (
                    (v[i + 1, j] - 2.0 * v[i, j] + v[i - 1, j]) / (dx * dx)
                    + (v[i, j + 1] - 2.0 * v[i, j] + v[i, j - 1]) / (dy * dy)
                )

                ut[i, j] = u[i, j] + dt * (
                    -u[i, j] * dudx
                    -v[i, j] * dudy
                    + nu * lap_u
                    + fx[i, j]
                )

                vt[i, j] = v[i, j] + dt * (
                    -u[i, j] * dvdx
                    -v[i, j] * dvdy
                    + nu * lap_v
                    + fy[i, j]
                )

    @staticmethod
    @njit(cache=True, parallel=True)
    def _build_pressure_rhs_numba(
        b_full, ut, vt, dx, dy, rho, dt, fluid_mask, obstacle_mask,
    ):
        nx, ny = ut.shape

        for i in prange(nx):
            for j in range(ny):
                b_full[i, j] = 0.0

        for i in prange(1, nx - 1):
            for j in range(1, ny - 1):
                if (not fluid_mask[i, j]) or obstacle_mask[i, j]:
                    continue
                if obstacle_mask[i + 1, j]:
                    continue
                if obstacle_mask[i - 1, j]:
                    continue
                if obstacle_mask[i, j + 1]:
                    continue
                if obstacle_mask[i, j - 1]:
                    continue

                div_u = (
                    (ut[i + 1, j] - ut[i - 1, j]) / (2.0 * dx)
                    + (vt[i, j + 1] - vt[i, j - 1]) / (2.0 * dy)
                )
                b_full[i, j] = -(rho / dt) * div_u

    @staticmethod
    @njit(cache=True, parallel=True)
    def _correct_velocity_numba(
        u,
        v,
        ut,
        vt,
        p,
        dx,
        dy,
        dt,
        rho,
        fluid_mask,
        obstacle_mask,
    ):
        """
        Correct tentative velocities with pressure gradient:
            u = u* - dt/rho * dp/dx
            v = v* - dt/rho * dp/dy

        Obstacle-adjacent pressure gradients use a simple mirrored-pressure rule:
            if neighbor is obstacle, use p_neighbor = p_center
        """
        nx, ny = u.shape

        for i in prange(nx):
            for j in range(ny):
                u[i, j] = ut[i, j]
                v[i, j] = vt[i, j]

        for i in prange(1, nx - 1):
            for j in range(1, ny - 1):
                if (not fluid_mask[i, j]) or obstacle_mask[i, j]:
                    u[i, j] = 0.0
                    v[i, j] = 0.0
                    continue

                p_ip1j = p[i, j] if obstacle_mask[i + 1, j] else p[i + 1, j]
                p_im1j = p[i, j] if obstacle_mask[i - 1, j] else p[i - 1, j]
                p_ijp1 = p[i, j] if obstacle_mask[i, j + 1] else p[i, j + 1]
                p_ijm1 = p[i, j] if obstacle_mask[i, j - 1] else p[i, j - 1]

                dpdx = (p_ip1j - p_im1j) / (2.0 * dx)
                dpdy = (p_ijp1 - p_ijm1) / (2.0 * dy)

                u[i, j] = ut[i, j] - (dt / rho) * dpdx
                v[i, j] = vt[i, j] - (dt / rho) * dpdy

    @staticmethod
    @njit(cache=True)
    def _divergence_inf_numba(u, v, fluid_mask, obstacle_mask, dx, dy):
        """
        Compute infinity norm of divergence on clean fluid cells only.
        Left serial intentionally; diagnostics are not the main bottleneck.
        """
        nx, ny = u.shape
        max_div = 0.0

        for i in range(1, nx - 1):
            for j in range(1, ny - 1):
                if (not fluid_mask[i, j]) or obstacle_mask[i, j]:
                    continue
                if obstacle_mask[i + 1, j]:
                    continue
                if obstacle_mask[i - 1, j]:
                    continue
                if obstacle_mask[i, j + 1]:
                    continue
                if obstacle_mask[i, j - 1]:
                    continue

                div_u = abs(
                    (u[i + 1, j] - u[i - 1, j]) / (2.0 * dx)
                    + (v[i, j + 1] - v[i, j - 1]) / (2.0 * dy)
                )
                if div_u > max_div:
                    max_div = div_u

        return max_div

    # HELPERS
    def _apply_velocity_bc(self, u, v, dt=None):
        """
        Apply velocity BCs with LBM-like inlet behavior:
        - ramped parabolic or plug inflow
        - small transverse perturbation
        - no-slip walls and obstacle
        - zero-gradient or convective outlet
        """
        if dt is None:
            dt = self.dt

        masks = self.masks
        bc = self.bc

        # First apply environment wall BCs only
        wall_mask = masks["wall"]
        u[wall_mask] = 0.0
        v[wall_mask] = 0.0

        # LBM-like inlet at next time
        t_bc = self.time + dt
        ux_in, uy_in = self._inlet_profile_values(t_bc)

        left_inlet = masks["left"] & ~masks["wall"]
        if np.any(left_inlet):
            inlet_rows = left_inlet[0, :]
            u[0, inlet_rows] = ux_in[inlet_rows]
            v[0, inlet_rows] = uy_in[inlet_rows]

        # Right outlet
        right_outlet = masks["right"] & ~masks["obstacle"]
        if np.any(right_outlet):
            mask = right_outlet[-1, :]

            if self.outlet_bc == "zero_gradient":
                u[-1, mask] = u[-2, mask]
                v[-1, mask] = v[-2, mask]

            elif self.outlet_bc == "convective":
                Uc = (
                    float(self.outlet_convection_speed)
                    if self.outlet_convection_speed is not None
                    else float(self.environment.v0)
                )

                alpha = Uc * dt / max(self.dx, 1e-14)
                alpha = min(max(alpha, 0.0), 1.0)

                u[-1, mask] = u[-1, mask] - alpha * (u[-1, mask] - u[-2, mask])
                v[-1, mask] = v[-1, mask] - alpha * (v[-1, mask] - v[-2, mask])

        # Obstacle no-slip
        u[masks["obstacle"]] = 0.0
        v[masks["obstacle"]] = 0.0

   

    def _apply_pressure_bc(self, p):
        """
        Scatter fixed pressure values into the full pressure field.

        Current choice:
            - outlet pressure anchor p = 0
            - obstacle pressure stored as 0 for plotting only
        """
        p[self.dirichlet_pressure_mask] = 0.0
        p[self.masks["obstacle"]] = 0.0

    def _compute_dt(self):
        """
        Compute stable timestep based on convective CFL and explicit diffusion.
        """
        if not self.adaptive_dt:
            return self.dt

        fluid = self.masks["fluid"]
        speed_u = np.max(np.abs(self.u[fluid])) if np.any(fluid) else 0.0
        speed_v = np.max(np.abs(self.v[fluid])) if np.any(fluid) else 0.0
        speed_u = max(speed_u, 1e-12)
        speed_v = max(speed_v, 1e-12)

        dt_adv = self.cfl_safety * min(self.dx / speed_u, self.dy / speed_v)
        dt_diff = self.diff_safety * min(self.dx * self.dx, self.dy * self.dy) / max(
            self.nu, 1e-12
        )

        return min(self.dt, dt_adv, dt_diff)

    def _remaining_target_time(self):
        if self.target_physical_time is None:
            return None
        return self.target_physical_time - self.time

    def _target_time_reached(self):
        if self.target_physical_time is None:
            return False
        return self.time >= self.target_physical_time - 1e-14

    def _clip_dt_to_target(self, dt_candidate):
        remaining = self._remaining_target_time()
        if remaining is None:
            return dt_candidate
        return min(dt_candidate, max(0.0, remaining))

    def _build_source_arrays(self):
        """
        Evaluate external forcing from the environment.
        """
        fx_raw, fy_raw = self.environment.source_term(self.X, self.Y, self.time)

        fx = np.asarray(fx_raw, dtype=float)
        fy = np.asarray(fy_raw, dtype=float)

        if fx.shape == ():
            fx = np.full_like(self.u, float(fx))
        if fy.shape == ():
            fy = np.full_like(self.v, float(fy))

        return fx, fy

    def _extract_active_rhs(self):
        rhs = np.zeros(self.n_pressure_unknowns, dtype=float)
        mask = self.pressure_id >= 0
        rhs[self.pressure_id[mask]] = self.b_full[mask]
        return rhs

    def _extract_active_pressure_guess(self):
        x0 = np.zeros(self.n_pressure_unknowns, dtype=float)
        mask = self.pressure_id >= 0
        x0[self.pressure_id[mask]] = self.p[mask]
        return x0

    def _scatter_active_pressure(self, p_active):
        self.p[:, :] = 0.0
        mask = self.pressure_id >= 0
        self.p[mask] = p_active[self.pressure_id[mask]]
        self._apply_pressure_bc(self.p)

    def _solve_pressure(self):
        """
        Solve the sparse pressure system using the requested method.
        """
        self._apply_pressure_bc(self.p)
        rhs = self._extract_active_rhs()
        x0 = self._extract_active_pressure_guess()

        if self.poisson_method == "bicgstab":
            p_active, info = bicgstab(
                self.A_pressure,
                rhs,
                x0=x0,
                rtol=self.pressure_tol,
                atol=0.0,
                maxiter=self.pressure_maxiter,
                M=self.M_pressure,
            )
            if info != 0:
                print(f"  Warning: bicgstab returned info={info}")

        elif self.poisson_method == "cg":
            p_active, info = cg(
                self.A_pressure,
                rhs,
                x0=x0,
                rtol=self.pressure_tol,
                atol=0.0,
                maxiter=self.pressure_maxiter,
                M=self.M_pressure,
            )
            if info != 0:
                print(f"  Warning: cg returned info={info}")

        elif self.poisson_method == "amg":
            if self.amg_solver is None:
                raise RuntimeError("AMG solver not initialized.")
            p_active = self.amg_solver.solve(
                rhs,
                x0=x0,
                tol=self.pressure_tol,
                maxiter=self.pressure_maxiter,
            )

        else:
            raise ValueError(
                f"Unknown poisson_method='{self.poisson_method}'. "
                "Use 'bicgstab', 'cg', or 'amg'."
            )

        self._scatter_active_pressure(p_active)

    def _single_step(self, dt):
        """
        Advance the solution by one timestep.
        """
        fx, fy = self._build_source_arrays()

        # 1) Tentative velocity
        self._predictor_step_numba(
            self.ut,
            self.vt,
            self.u,
            self.v,
            self.dx,
            self.dy,
            dt,
            self.nu,
            self.masks["fluid"],
            self.masks["obstacle"],
            fx,
            fy,
            self.convection_order == "second",
        )
        self._apply_velocity_bc(self.ut, self.vt, dt=dt)

        # 2) Pressure RHS
        self._build_pressure_rhs_numba(
            self.b_full,
            self.ut,
            self.vt,
            self.dx,
            self.dy,
            self.rho,
            dt,
            self.masks["fluid"],
            self.masks["obstacle"],
        )

        # 3) Pressure solve
        self._solve_pressure()

        # 4) Velocity correction
        self._correct_velocity_numba(
            self.u,
            self.v,
            self.ut,
            self.vt,
            self.p,
            self.dx,
            self.dy,
            dt,
            self.rho,
            self.masks["fluid"],
            self.masks["obstacle"],
        )
        self._apply_velocity_bc(self.u, self.v, dt=dt)

        self.time += dt

    def _diagnostics(self):
        """
        Return diagnostic quantities for logging and stability monitoring.
        """
        speed = np.sqrt(self.u**2 + self.v**2)
        speed[self.masks["obstacle"]] = np.nan

        div_inf = self._divergence_inf_numba(
            self.u,
            self.v,
            self.masks["fluid"],
            self.masks["obstacle"],
            self.dx,
            self.dy,
        )

        return {
            "time": float(self.time),
            "u_max": float(np.nanmax(np.abs(self.u))),
            "v_max": float(np.nanmax(np.abs(self.v))),
            "speed_max": float(np.nanmax(speed)),
            "div_inf": float(div_inf),
            "p_max": float(np.nanmax(np.abs(self.p))),
        }

    def solve(
        self,
        verbose=True,
        visualizer=None,
        record_video=False,
        video_filename="fdm_simulation.mp4",
        visualize_every=50,
        return_history=True,
    ):
        """Run the finite-difference Navier-Stokes simulation."""
        sim_visualizer = None
        if visualizer is not None:
            from visualization import SimulationVisualizer
            sim_visualizer = SimulationVisualizer(
                self.nx,
                self.ny,
                self.masks["obstacle"],
                fps=30,
                record_video=record_video,
                video_filename=video_filename,
            )
            sim_visualizer.setup(visualizer)

        if verbose:
            print("Initializing FDM solver...")
            print(f"  Grid: {self.nx} x {self.ny}")
            print(f"  dx, dy: {self.dx:.5e}, {self.dy:.5e}")
            print(f"  Base dt: {self.dt:.5e}")
            print(f"  Viscosity used: {self.nu:.5e}")
            print(f"  Reynolds number: {self.reynolds_number:.2f}")
            print(f"  Pressure unknowns: {self.n_pressure_unknowns}")
            print(f"  Poisson method: {self.poisson_method}")
            print(f"  Pressure tol: {self.pressure_tol:.1e}")
            print(f"  Pressure maxiter: {self.pressure_maxiter}")
            print(f"  Convection order: {self.convection_order}")
            print(f"  Outlet BC: {self.outlet_bc}")
            if self.outlet_bc == "convective":
                Uc = (
                    float(self.outlet_convection_speed)
                    if self.outlet_convection_speed is not None
                    else float(self.environment.v0)
                )
                print(f"  Outlet convection speed: {Uc:.5e}")
            if self.poisson_method == "amg":
                print(f"  PyAMG available: {HAVE_PYAMG}")
            if self.poisson_method in ("bicgstab", "cg"):
                print(f"  ILU preconditioner: {'yes' if self.M_pressure is not None else 'no'}")
            if self.target_physical_time is not None:
                print(f"  Target physical time: {self.target_physical_time}")

        history = {
            "time": [],
            "dt": [],
            "u_max": [],
            "v_max": [],
            "speed_max": [],
            "div_inf": [],
            "p_max": [],
        }

        stop_reason = "n_steps"
        steps_executed = 0

        for step in range(1, self.n_steps + 1):
            if self._target_time_reached():
                stop_reason = "target_physical_time"
                break

            dt = self._clip_dt_to_target(self._compute_dt())
            if dt <= 0.0:
                stop_reason = "target_physical_time"
                break

            self._single_step(dt)
            steps_executed = step
            diag = self._diagnostics()
            hit_target_time = self._target_time_reached()
            is_last_step = (step == self.n_steps) or hit_target_time

            if return_history:
                history["time"].append(diag["time"])
                history["dt"].append(dt)
                history["u_max"].append(diag["u_max"])
                history["v_max"].append(diag["v_max"])
                history["speed_max"].append(diag["speed_max"])
                history["div_inf"].append(diag["div_inf"])
                history["p_max"].append(diag["p_max"])

            if verbose and (step == 1 or step % 1000 == 0 or is_last_step):
                print(
                    f"step={step:6d}/{self.n_steps}  "
                    f"t={diag['time']:.5f}  "
                    f"dt={dt:.3e}  "
                    f"|u|max={diag['speed_max']:.4f}  "
                    f"||div u||_inf={diag['div_inf']:.3e}  "
                    f"|p|max={diag['p_max']:.3e}"
                )

            if sim_visualizer is not None and (step % visualize_every == 0 or is_last_step):
                field = visualizer.compute_field(self.u, self.v, self.p)
                sim_visualizer.update(field, step)

            if hit_target_time:
                stop_reason = "target_physical_time"
                break

        if sim_visualizer is not None:
            sim_visualizer.finalize()

        result = {
            "ux": self.u.copy(),
            "uy": self.v.copy(),
            "p": self.p.copy(),
            "obstacle": self.masks["obstacle"].copy(),
            "fluid": self.masks["fluid"].copy(),
            "metadata": {
                "nx": self.nx,
                "ny": self.ny,
                "dx": self.dx,
                "dy": self.dy,
                "dt_base": self.dt,
                "adaptive_dt": self.adaptive_dt,
                "time_final": self.time,
                "n_steps": self.n_steps,
                "n_steps_executed": steps_executed,
                "stop_reason": stop_reason,
                "target_physical_time": self.target_physical_time,
                "rho": self.rho,
                "nu": self.nu,
                "u_inlet": float(self.environment.v0),
                "reynolds_number": self.reynolds_number,
                "poisson_method": self.poisson_method,
                "pressure_tol": self.pressure_tol,
                "pressure_maxiter": self.pressure_maxiter,
                "pressure_unknowns": self.n_pressure_unknowns,
                "pyamg_available": HAVE_PYAMG,
                "convection_order": self.convection_order,
                "inlet_profile": self.inlet_profile,
                "inlet_perturbation": self.inlet_perturbation,
                "velocity_ramp_tau": self.velocity_ramp_tau,
                "inlet_strip_fraction": self.inlet_strip_fraction,
                "outlet_bc": self.outlet_bc,
                "outlet_convection_speed": (
                    float(self.outlet_convection_speed)
                    if self.outlet_convection_speed is not None
                    else None
                ),
            },
        }

        if return_history:
            result["history"] = history

        return result


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
