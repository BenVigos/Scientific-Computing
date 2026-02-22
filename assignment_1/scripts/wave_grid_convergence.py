import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from src import fdm_schemes

# Choose a base font size
base_fs = 14

# Global update (applies for the rest of this Python session)
mpl.rcParams.update({
    'font.size': base_fs,                   # default text size
    'axes.titlesize': base_fs * 1.1,       # axes title
    'axes.labelsize': base_fs,              # x/y labels
    'xtick.labelsize': base_fs * 0.9,       # x tick labels
    'ytick.labelsize': base_fs * 0.9,       # y tick labels
    'legend.fontsize': base_fs * 0.9,       # legend text
    'legend.title_fontsize': base_fs * 0.9, # legend title
    'figure.titlesize': base_fs * 1.3,      # figure suptitle
    'figure.figsize': (8, 6),             # optional default figure size
})

L = 1.0
dt = 0.001
T = 1.1

c = 1.0

# Grids to test (increasing refinement) — keep a small set by default
Ns = np.logspace(1, 3, num=50, dtype=int)

# helper to build the initial condition used previously (analytic mode uses sin(2πx))
def make_initial(N):
    x = np.linspace(0, L, N)
    u0 = np.sin(2 * np.pi * x)
    return x, u0

# store results per grid
results = {}

print("Running simulations on grids:", Ns)
for N in Ns:
    dx = L / (N - 1)
    x, u0 = make_initial(N)

    # call the solver and request the actual dt and time array used
    u_res = fdm_schemes.wave_equation_1d(u0, c, dx, dt, T, return_dt=True)
    if isinstance(u_res, tuple):
        u, dt_used, t = u_res
    else:
        # backward compatibility: solver returned only u
        u = u_res
        dt_used = dt
        t = np.linspace(0.0, T, u.shape[0])

    results[N] = {"x": x, "u": u, "dx": dx, "dt": dt_used, "t": t}
    print(f"Finished N={N}: nx={N}, dx={dx:.5e}, nt={u.shape[0]}, dt_used={dt_used:.5e}")

# Use the finest grid as the reference solution (if desired)
N_ref = max(Ns)
ref = results[N_ref]
x_ref = ref["x"]
u_ref_final = ref["u"][-1]

# prepare lists for errors and energies
dxs = []
L2 = []

# analytic energy expression for u(x,t)=sin(2πx)cos(2πt)
# E_analytic(t) = π^2 * (sin^2(2π t) + c^2 cos^2(2π t)) on domain [0,1]
pi = np.pi

for N in Ns:
    x = results[N]["x"]
    u = results[N]["u"]
    dx = results[N]["dx"]
    dt_used = results[N]["dt"]
    t = results[N]["t"]

    nt = u.shape[0]

    # Analytical final-time solution
    u_anal_final = np.sin(2 * np.pi * x) * np.cos(2 * np.pi * t[-1])

    u_final = u[-1]
    e_final = u_final - u_anal_final
    L2_final = np.sqrt(np.sum(e_final ** 2) / len(e_final))

    dxs.append(dx)
    L2.append(L2_final)


    print(f"N={N:4d}, dx={results[N]['dx']:.5e}, L2={L2_final:.5e}")



plt.plot(dxs, L2, 'o-')
plt.xscale('log')
plt.yscale('log')


plt.gca().invert_xaxis()
plt.xlabel(r'Grid spacing $\Delta x$')
plt.ylabel(r'$\mathcal{L}_2$ error')
plt.grid(True, which='both', ls='--', lw=0.5)
plt.legend()
plt.title('Grid convergence (final time)')
plt.tight_layout()
plt.savefig('grid_convergence_errors.png', dpi=200)
plt.show()

# convert to arrays and perform robust log-log fit and plotting
dxs_arr = np.array(dxs, dtype=float)
errs_arr = np.array(L2, dtype=float)
mask = (dxs_arr > 0) & (errs_arr > 0) & np.isfinite(dxs_arr) & np.isfinite(errs_arr)

plt.figure(figsize=(6, 4))
if np.sum(mask) >= 2:
    dxs_fit = dxs_arr[mask]
    errs_fit = errs_arr[mask]
    idx_sort = np.argsort(dxs_fit)
    dxs_fit = dxs_fit[idx_sort]
    errs_fit = errs_fit[idx_sort]

    coeff = np.polyfit(np.log(dxs_fit), np.log(errs_fit), 1)
    slope = coeff[0]
    intercept = coeff[1]
    print(f"Fitted log-log slope (L2 vs dx): {slope:.4f}")

    # plot data and fitted line
    plt.loglog(dxs_arr, errs_arr, 'o', label='L2 errors')
    dxs_line = np.array([dxs_fit.min(), dxs_fit.max()])
    errs_line = np.exp(intercept) * dxs_line ** slope
    plt.loglog(dxs_line, errs_line, '--', color='gray', label=f'fit: error ~ dx^{slope:.2f}')

    # annotate slope at mid-point
    x_annot = np.exp((np.log(dxs_line[0]) + np.log(dxs_line[1])) / 2)
    y_annot = np.exp(intercept) * x_annot ** slope
    plt.annotate(f'slope = {slope:.2f}', xy=(x_annot, y_annot), xytext=(10, -20), textcoords='offset points')
else:
    print('Not enough valid points for log-log fit')
    plt.loglog(dxs_arr, errs_arr, 'o-', label='L2 errors')

plt.gca().invert_xaxis()
plt.xlabel(r'Grid spacing $\Delta x$')
plt.ylabel(r'$\mathcal{L}_2$ error')
plt.grid(True, which='both', ls='--', lw=0.5)
plt.legend()
plt.title('Grid convergence (t=1.1) with log-log fit')
plt.tight_layout()
plt.savefig('grid_convergence_errors.png', dpi=200)
plt.show()
