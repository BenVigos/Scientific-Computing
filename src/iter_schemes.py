from src.grid import make_grid, empty_sink, empty_insulator
import numpy as np


# make prints toggable later ig 

def convergence_check(c_new, c_old):
    return np.max(np.abs(c_new - c_old))


def jacobi(N, tol=1e-5, max_iter=10000):
    c = make_grid(N)
    c_new = c.copy()

    deltas = []

    for k in range(max_iter):

        for i in range(1, N-1):
            for j in range(N):
                
                jp = (j + 1) % N
                jm = (j - 1) % N

                c_new[i, j] = 0.25 * (
                    c[i, jp] + c[i, jm] +
                    c[i+1, j] + c[i-1, j]
                )

        delta = convergence_check(c_new, c)
        deltas.append(delta)

        if delta < tol:
            # print(f"Jacobi scheme converged in {k+1} iterations")
            break

        c[:] = c_new[:]

    return c, deltas


def gauss_seidel(N, tol=1e-5, max_iter=10000):
    c = make_grid(N)

    deltas = []

    for k in range(max_iter):

        c_old = c.copy()

        for i in range(1, N-1):
            for j in range(N):

                jp = (j + 1) % N
                jm = (j - 1) % N

                c[i, j] = 0.25 * (
                    c[i, jp] + c[i, jm] +
                    c[i+1, j] + c[i-1, j]
                )

        delta = convergence_check(c, c_old)
        deltas.append(delta)

        if delta < tol:
            # print(f"Gauss-Seidel scheme converged in {k+1} iterations")
            break

    return c, deltas


def sor(N, omega = 1, tol=1e-5, max_iter=10000, sink=None, insulator=None, ret_hist=False, init_grid=None):
    """
    Successive Over-Relaxation (SOR) method for solving the steady-state diffusion equation on a 2D grid with specified boundary conditions, sinks, and insulators.

    :param N: Grid size (NxN)
    :param omega: Relaxation factor (ω=1 corresponds to Gauss-Seidel, ω>1 is over-relaxation)
    :param tol: convergence tolerance for the maximum change in concentration between iterations
    :param max_iter: maximum number of iterations to perform
    :param sink: mask (boolean 2D array) indicating locations of sinks where concentration is fixed to zero
    :param insulator: mask (boolean 2D array) indicating locations of insulators where concentration does not change (no flux)
    :param ret_hist: return history of concentration fields at each iteration (for visualization) if True
    :return: concentration field after convergence, list of deltas for each iteration, (optionally) history of concentration fields, number of iterations taken, and whether convergence was achieved
    """

    if init_grid is not None:
        c = init_grid.copy()
    else:
        c = make_grid(N)

    if sink is None:
        sink = empty_sink(N)

    if insulator is None:
        insulator = empty_insulator(N)
    
    c[sink] = 0.0

    deltas = []
    history = []

    converged = False
    for k in range(max_iter):

        c_old = c.copy()

        for i in range(1, N-1):
            for j in range(N):

                if sink[i, j]:
                    c[i, j] = 0.0
                    continue

                if insulator[i, j]:
                    continue

                jp = (j + 1) % N
                jm = (j - 1) % N

                cij = c[i, j]

                right = cij if insulator[i, jp] else c[i, jp]
                left = cij if insulator[i, jm] else c[i, jm]
                top = cij if insulator[i+1, j] else c[i+1, j]
                bottom = cij if insulator[i-1, j] else c[i-1, j]

                gs_value = 0.25 * (right + left + top + bottom)

                c[i, j] = omega * gs_value + (1 - omega) * c[i, j]

        delta = convergence_check(c, c_old)
        deltas.append(delta)

        if ret_hist:
            history.append(c.copy())

        if delta < tol:
            # print(f"SOR (ω={omega}) scheme converged in {k+1} iterations")
            converged = True
            break

    
    if ret_hist:
        return c, deltas, history, k+1, converged

    return c, deltas, k+1, converged
