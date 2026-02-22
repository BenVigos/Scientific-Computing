"""Wave analysis helpers: reusable utilities for analytic solutions, errors,
and grid-convergence helpers.

This module is intended to hold code reused by experiment scripts in
assignment_1/scripts.
"""

from __future__ import annotations

import numpy as np
from typing import Tuple, Dict
from src import fdm_schemes


def analytic_solution(x: np.ndarray, t: float) -> np.ndarray:
    """Analytic solution u(x,t) = sin(2πx) cos(2πt) on [0,1]."""
    return np.sin(2 * np.pi * x) * np.cos(2 * np.pi * t)


def compute_L2(u_num: np.ndarray, u_ref: np.ndarray) -> float:
    """Compute discrete L2 error between two vectors (same grid).

    Uses the root-mean-square (sqrt of mean squared error).
    """
    e = u_num - u_ref
    return float(np.sqrt(np.sum(e ** 2) / e.size))


def fit_loglog_slope(dxs: np.ndarray, errs: np.ndarray) -> Tuple[float, float]:
    """Fit a straight line to log(err) = slope*log(dx) + intercept.

    Returns (slope, intercept). If not enough valid points, returns (nan, nan).
    """
    dxs = np.asarray(dxs, dtype=float)
    errs = np.asarray(errs, dtype=float)
    mask = (dxs > 0) & (errs > 0) & np.isfinite(dxs) & np.isfinite(errs)
    if np.sum(mask) < 2:
        return np.nan, np.nan
    dxs_f = dxs[mask]
    errs_f = errs[mask]
    idx = np.argsort(dxs_f)
    dxs_f = dxs_f[idx]
    errs_f = errs_f[idx]
    coeff = np.polyfit(np.log(dxs_f), np.log(errs_f), 1)
    slope = float(coeff[0])
    intercept = float(coeff[1])
    return slope, intercept


def run_single_simulation(N: int, dt: float, T: float, c: float, *, compare_analytic: bool = False) -> Dict:
    """Run the wave solver on a grid with N points and return results.

    The function returns a dictionary with basic keys always present:
      - 'N', 'x', 'u', 'dx', 'dt', 't'

    Optionally include:
      - 'L2_final' when compare_analytic=True (error vs analytic u=sin(2πx)cos(2πt)).

    Parameters
    - compare_analytic: if True compute L2 error at final time vs analytic solution.
    """
    x = np.linspace(0.0, 1.0, N)
    dx = x[1] - x[0] if N > 1 else 0.0

    # analytic initial condition for the manufactured solution
    u0 = analytic_solution(x, 0.0)

    # call solver
    res = fdm_schemes.wave_equation_1d(u0, c, dx, dt, T, return_dt=True)
    if isinstance(res, tuple):
        u, dt_used, t = res
    else:
        u = res
        dt_used = dt
        t = np.linspace(0.0, T, u.shape[0])

    out: Dict = {
        'N': N,
        'x': x,
        'u': u,
        'dx': dx,
        'dt': dt_used,
        't': t,
    }

    # Analytic comparison (optional)
    if compare_analytic:
        u_anal_final = analytic_solution(x, t[-1])
        L2_final = compute_L2(u[-1], u_anal_final)
        out['L2_final'] = L2_final

    return out

