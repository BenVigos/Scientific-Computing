from src.grid import make_grid, empty_sink, empty_insulator
import numpy as np
from numba import njit, prange


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

def sor(N, omega=1, tol=1e-5, max_iter=10000,
        sink=None, insulator=None,
        ret_hist=False, init_grid=None):
    """
    Successive Over-Relaxation (SOR) method for solving the steady-state diffusion equation on a 2D grid with specified boundary conditions, sinks, and insulators.

    :param N: Grid size (NxN)
    :param omega: Relaxation factor (ω=1 corresponds to Gauss-Seidel, ω>1 is over-relaxation)
    :param tol: convergence tolerance for the maximum change in concentration between iterations
    :param max_iter: maximum number of iterations to perform
    :param sink: mask (boolean 2D array) indicating locations of sinks where concentration is fixed to zero
    :param insulator: mask (boolean 2D array) indicating locations of insulators where concentration does not change (no flux)
    :param ret_hist: return history of concentration fields at each iteration (for visualization) if True
    :param init_grid: optional initial concentration grid (if None, starts with default boundary conditions)
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

    # --- Precompute periodic neighbors (avoid modulo in loop)
    j_plus = [(j + 1) % N for j in range(N)]
    j_minus = [(j - 1) % N for j in range(N)]

    # --- Local variable binding (faster lookups)
    c_local = c
    sink_local = sink
    ins_local = insulator

    for k in range(max_iter):

        max_delta = 0.0  # compute convergence inline

        for i in range(1, N-1):
            for j in range(N):

                if sink_local[i, j]:
                    c_local[i, j] = 0.0
                    continue

                if ins_local[i, j]:
                    continue

                jp = j_plus[j]
                jm = j_minus[j]

                cij = c_local[i, j]

                right = cij if ins_local[i, jp] else c_local[i, jp]
                left  = cij if ins_local[i, jm] else c_local[i, jm]
                top   = cij if ins_local[i+1, j] else c_local[i+1, j]
                bottom= cij if ins_local[i-1, j] else c_local[i-1, j]

                gs_value = 0.25 * (right + left + top + bottom)
                new_val = omega * gs_value + (1 - omega) * cij

                diff = abs(new_val - cij)
                if diff > max_delta:
                    max_delta = diff

                c_local[i, j] = new_val

        deltas.append(max_delta)

        if ret_hist:
            history.append(c_local.copy())

        if max_delta < tol:
            converged = True
            break

    if ret_hist:
        return c_local, deltas, history, k+1, converged
    else:
        return c_local, deltas, k+1, converged



@njit
def sor_numba(N, omega, c, sink,insulator, max_iter = 100000, tol = 1e-5):
    """
        Successive Over-Relaxation (SOR) method for solving the steady-state diffusion equation on a 2D grid with specified boundary conditions, sinks, and insulators. This version is optimized with Numba for performance.
    :param N: Grid size (NxN)
    :param omega: Relaxation factor (ω=1 corresponds to Gauss-Seidel, ω>1 is over-relaxation)
    :param c: initial concentration grid (2D numpy array)
    :param sink: mask (boolean 2D array) indicating locations of sinks where concentration is fixed to zero
    :param insulator: mask (boolean 2D array) indicating locations of insulators where concentration does not change (no flux)
    :param max_iter: maximum number of iterations to perform
    :param tol: convergence tolerance for the maximum change in concentration between iterations
    :return:
    """
    j_plus = np.empty(N, dtype=np.int64)
    j_minus = np.empty(N, dtype=np.int64)
    for j in range(N):
        j_plus[j] = (j + 1) % N
        j_minus[j] = (j - 1) % N



    mask_numeric = 1.0 - sink.astype(np.float64)

    c *= mask_numeric



    for k in range(max_iter):
        max_delta = 0.0
        for i in range(1, N-1):
            for j in range(N):
                if insulator[i, j] or sink[i, j]:
                    continue

                jp = j_plus[j]
                jm = j_minus[j]
                cij = c[i, j]

                right = cij if insulator[i, jp] else c[i, jp]
                left  = cij if insulator[i, jm] else c[i, jm]
                top   = cij if insulator[i+1, j] else c[i+1, j]
                bottom= cij if insulator[i-1, j] else c[i-1, j]

                new_val = omega * 0.25 * (right + left + top + bottom) + (1 - omega) * cij
                diff = abs(new_val - cij)
                if diff > max_delta:
                    max_delta = diff
                c[i, j] = new_val

        if max_delta < tol:
            return c, k+1, True
    return c, max_iter, False


@njit(parallel=True)
def sor_numba_redblack(N, omega, c, sink, insulator, max_iter=100000, tol=1e-5):
    """
    Parallel Red-Black Successive Over-Relaxation (SOR) solver for 2D diffusion.
    """

    j_plus = np.empty(N, dtype=np.int64)
    j_minus = np.empty(N, dtype=np.int64)

    for j in range(N):
        j_plus[j] = (j + 1) % N
        j_minus[j] = (j - 1) % N

    mask_numeric = 1.0 - sink.astype(np.float64)

    c *= mask_numeric

    for k in range(max_iter):

        max_delta = 0.0

        # --------------------
        # RED UPDATE
        # --------------------
        row_deltas = np.zeros(N)
        for i in prange(1, N-1):
            local_max = 0.0
            for j in range(N):

                if (i + j) % 2 != 0:
                    continue

                if insulator[i, j] or sink[i, j]:
                    continue

                jp = j_plus[j]
                jm = j_minus[j]

                cij = c[i, j]

                right  = cij if insulator[i, jp] else c[i, jp]
                left   = cij if insulator[i, jm] else c[i, jm]
                top    = cij if insulator[i+1, j] else c[i+1, j]
                bottom = cij if insulator[i-1, j] else c[i-1, j]

                new_val = omega * 0.25 * (right + left + top + bottom) + (1 - omega) * cij

                diff = abs(new_val - cij)
                if diff > local_max:
                    local_max = diff

                c[i, j] = new_val

            row_deltas[i] = local_max

        red_delta = np.max(row_deltas)


        # --------------------
        # BLACK UPDATE
        # --------------------
        row_deltas[:] = 0.0
        for i in prange(1, N-1):
            local_max = 0.0
            for j in range(N):

                if (i + j) % 2 != 1:
                    continue

                if insulator[i, j] or sink[i, j]:
                    continue

                jp = j_plus[j]
                jm = j_minus[j]

                cij = c[i, j]

                right  = cij if insulator[i, jp] else c[i, jp]
                left   = cij if insulator[i, jm] else c[i, jm]
                top    = cij if insulator[i+1, j] else c[i+1, j]
                bottom = cij if insulator[i-1, j] else c[i-1, j]

                new_val = omega * 0.25 * (right + left + top + bottom) + (1 - omega) * cij

                diff = abs(new_val - cij)
                if diff > local_max:
                    local_max = diff

                c[i, j] = new_val

            row_deltas[i] = local_max

        black_delta = np.max(row_deltas)

        if red_delta > black_delta:
            max_delta = red_delta
        else:
            max_delta = black_delta


        if max_delta < tol:
            return c, k + 1, True

    return c, max_iter, False