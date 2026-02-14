from .grid import make_grid
import numpy as np


def convergence_check(c_new, c_old):
    return np.max(np.abs(c_new - c_old))


def jacobi(N, tol=1e-5, max_iter=10000):
    c = make_grid(N)
    c_new = c.copy()

    deltas = []

    for k in range(max_iter):

        for i in range(N):
            for j in range(1, N-1):

                ip = (i + 1) % N
                im = (i - 1) % N

                c_new[i, j] = 0.25 * (
                    c[ip, j] + c[im, j] +
                    c[i, j+1] + c[i, j-1]
                )

        delta = convergence_check(c_new, c)
        deltas.append(delta)

        if delta < tol:
            print(f"Jacobi scheme converged in {k} iterations")
            break

        c[:] = c_new[:]

    return c, deltas


def gauss_seidel(N, tol=1e-5, max_iter=10000):
    c = make_grid(N)

    deltas = []

    for k in range(max_iter):

        c_old = c.copy()

        for i in range(N):
            for j in range(1, N-1):

                ip = (i + 1) % N
                im = (i - 1) % N

                c[i, j] = 0.25 * (
                    c[ip, j] + c[im, j] +
                    c[i, j+1] + c[i, j-1]
                )

        delta = convergence_check(c, c_old)
        deltas.append(delta)

        if delta < tol:
            print(f"Gauss-Seidel scheme converged in {k} iterations")
            break

    return c, deltas


def sor(N, omega, tol=1e-5, max_iter=10000):
    c = make_grid(N)

    deltas = []

    for k in range(max_iter):

        c_old = c.copy()

        for i in range(N):
            for j in range(1, N-1):

                ip = (i + 1) % N
                im = (i - 1) % N

                gs_value = 0.25 * (
                    c[ip, j] + c[im, j] +
                    c[i, j+1] + c[i, j-1]
                )

                c[i, j] = omega * gs_value + (1 - omega) * c[i, j]

        delta = convergence_check(c, c_old)
        deltas.append(delta)

        if delta < tol:
            print(f"SOR (ω={omega}) scheme converged in {k} iterations")
            break

    return c, deltas
