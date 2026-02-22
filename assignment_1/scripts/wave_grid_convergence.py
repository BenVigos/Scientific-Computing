import sys
from pathlib import Path
# ensure project root is importable
project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import numpy as np
import matplotlib.pyplot as plt
from assignment_1.scripts.wave_analysis import run_single_simulation, fit_loglog_slope


def main():
    L = 1.0
    dt = 0.001
    T = 1.1
    c = 1.0

    # reasonable default grids for quick convergence checks
    Ns = np.logspace(1, 3, num=50, dtype=int)

    results = {}
    for N in Ns:
        print(f"Running N={N} ...")
        res = run_single_simulation(N, dt, T, c, compare_analytic=True)
        results[N] = res
        print(f"N={N}: dx={res['dx']:.3e}, L2_final={res.get('L2_final', float('nan')):.3e}")

    dxs = np.array([results[N]['dx'] for N in Ns])
    errs = np.array([results[N].get('L2_final', np.nan) for N in Ns])

    # fit slope in log-log
    slope, intercept = fit_loglog_slope(dxs, errs)
    print(f"Fitted slope (L2 vs dx): {slope:.4f}")

    # plot
    plt.figure(figsize=(6, 4))
    plt.loglog(dxs, errs, 'o-', label='L2 error')
    if np.isfinite(slope):
        dx_line = np.array([dxs.min(), dxs.max()])
        plt.loglog(dx_line, np.exp(intercept) * dx_line ** slope, '--', label=f'fit slope {slope:.2f}')
    plt.gca().invert_xaxis()
    plt.xlabel('dx')
    plt.ylabel('L2 error')
    plt.grid(True, which='both', ls='--', lw=0.5)
    plt.legend()
    plt.title('Grid convergence')
    plt.tight_layout()
    plt.savefig('grid_convergence_errors.png', dpi=200)
    plt.show()


if __name__ == '__main__':
    main()
