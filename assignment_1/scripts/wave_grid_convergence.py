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
T = 2.0
c = 1.0

# Grids to test (increasing refinement)
Ns = np.logspace(1, 3, 50, dtype=int)

# helper to build the initial condition used previously
def make_initial(N):
    x = np.linspace(0, L, N)
    u0 = np.array(np.sin(5 * np.pi * x))
    # zero-out outside [1/5, 2/5] as in the original script
    u0[:int(np.ceil(1 / 5 * (N - 1)))] = 0
    u0[int(np.floor(2 / 5 * (N - 1)) + 1):] = 0
    return x, u0

# store results per grid
results = {}

print("Running simulations on grids:", Ns)
for N in Ns:
    dx = L / (N - 1)
    x, u0 = make_initial(N)

    # call the solver (keeps dt constant as in the original script)
    u = fdm_schemes.wave_equation_1d(u0, c, dx, dt, T)

    results[N] = {"x": x, "u": u, "dx": dx}
    print(f"Finished N={N}: nx={N}, dx={dx:.5e}, nt={u.shape[0]}")

# Use the finest grid as the reference solution
N_ref = max(Ns)
ref = results[N_ref]
x_ref = ref["x"]
u_ref_final = ref["u"][-1]

dxs = []
errors_L2 = []

for N in Ns:
    x = results[N]["x"]
    u_final = results[N]["u"][-1]

    # interpolate reference final solution onto current grid
    u_ref_on_x = np.interp(x, x_ref, u_ref_final)

    e = u_final - u_ref_on_x
    L2 = np.sqrt(np.sum(e ** 2) / len(e))

    dxs.append(results[N]["dx"])
    errors_L2.append(L2)


    print(f"N={N:4d}, dx={results[N]['dx']:.5e}, L2={L2:.5e}")



plt.plot(dxs, errors_L2, 'o-')
plt.xscale('log')


plt.gca().invert_xaxis()
plt.xlabel(r'Grid spacing $\Delta x$')
plt.ylabel(r'$\mathcal{L}_2$ error')
plt.grid(True, which='both', ls='--', lw=0.5)
plt.legend()
plt.title('Grid convergence (final time)')
plt.tight_layout()
plt.savefig('grid_convergence_errors.png', dpi=200)
plt.show()

