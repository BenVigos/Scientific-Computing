from scipy import sparse
import numpy as np
import numpy as np
from scipy import sparse

def wave_equation_1d(u0, c, dx, dt, t_max, bcs=(0, 0)):
    """
    Solves the 1D wave equation using an explicit finite difference scheme.
    The wave equation is: u_tt = c^2 * u_xx
    :param u0: initial displacement.
    :param c: wave speed.
    :param dx: spatial step size.
    :param dt: tiem step size.
    :param t_max: maximum simulation time.
    :param bcs: boundary conditions (Dirichlet values at left and right ends).
    :return: matrix of shape (nt, nx) with the solution at each time step.
    """
    nx = len(u0)
    nt = int(t_max / dt)

    # Stability condition (important!)
    # if c * dt / dx > 1:
    #     raise ValueError("CFL condition violated: c*dt/dx must be <= 1")

    A = laplacian_1d_matrix(nx, dx)
    b = boundary_term(nx, dx, bcs)

    u = np.zeros((nt, nx))

    # Initial condition
    u[0, :] = u0

    # Assume zero initial velocity
    u[1, 1:-1] = (
        u[0, 1:-1]
        + 0.5 * (c * dt) ** 2 * (A @ u[0, 1:-1] + b)
    )

    # Enforce boundary conditions
    u[:, 0]  = bcs[0]
    u[:, -1] = bcs[1]

    # Time stepping
    for n in range(1, nt - 1):
        u[n+1, 1:-1] = (
            2 * u[n, 1:-1]
            - u[n-1, 1:-1]
            + (c * dt) ** 2 * (A @ u[n, 1:-1] + b)
        )

    return u


def laplacian_1d_matrix(nx, dx):
    """
    Returns interior Laplacian matrix (Dirichlet formulation).
    Size: (nx-2) x (nx-2)
    """
    N = nx - 2

    A = sparse.diags([1, -2, 1], offsets=[-1, 0, 1], shape=(N, N), dtype=float)
    return A / dx**2


def boundary_term(nx, dx, bcs):
    """
    Boundary correction vector for interior system.
    """
    b = np.zeros(nx - 2)
    b[0]  = bcs[0] / dx**2
    b[-1] = bcs[1] / dx**2
    return b