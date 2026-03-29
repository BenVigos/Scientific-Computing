import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.optimize import differential_evolution
import matplotlib.pyplot as plt

C = 3e8
FREQ = 0.8e9  # Scaled to 0.8 GHz
K_AIR = 2 * np.pi * FREQ / C
A_AMP = 10**4
SIGMA = 0.2

DOMAIN_W = 10.0
DOMAIN_H = 8.0
DX = 0.05
NX = int(DOMAIN_W / DX) + 1
NY = int(DOMAIN_H / DX) + 1

# Measurement Points
MEASUREMENT_POINTS = np.array([
    [1.0, 5.0], [2.0, 1.0], [9.0, 1.0], [9.0, 7.0]
])

history_x = []
history_y = []
history_scores = []
best_score_progression = []
current_best = -1

def build_refractive_index_map():
    n_map = np.ones((NX, NY), dtype=complex)
    wall_val = 2.5 + 0.5j
    
    def add_wall(xmin, xmax, ymin, ymax):
        ix1, ix2 = int(xmin/DX), int(xmax/DX)
        iy1, iy2 = int(ymin/DX), int(ymax/DX)
        n_map[ix1:ix2+1, iy1:iy2+1] = wall_val

    # Outer Walls
    add_wall(0, 0.15, 0, 8.0); add_wall(9.85, 10.0, 0, 8.0)
    add_wall(0, 10.0, 0, 0.15); add_wall(0, 10.0, 7.85, 8.0)
    
    # Inner Walls
    add_wall(0.15, 3.0, 2.925, 3.075)   # Kitchen top
    add_wall(4.0, 6.0, 2.925, 3.075)    # Hall top
    add_wall(6.85, 9.85, 2.925, 3.075)  # Bathroom top
    add_wall(2.425, 2.575, 0.15, 2.0)   # Kitchen right
    add_wall(5.925, 6.075, 3.075, 7.85) # Bedroom 1 left
    add_wall(6.925, 7.075, 0.15, 1.5)   # Bathroom left
    
    return n_map

def build_system_matrix(n_map):
    N = NX * NY
    main_diag = np.zeros(N, dtype=complex)
    upper_x, lower_x = np.zeros(N - NY, dtype=complex), np.zeros(N - NY, dtype=complex)
    upper_y, lower_y = np.zeros(N - 1, dtype=complex), np.zeros(N - 1, dtype=complex)
    
    for i in range(NX):
        for j in range(NY):
            idx = i * NY + j
            k_eff = K_AIR * n_map[i, j]
            
            diag_val = -4/DX**2 + k_eff**2
            ux = 1/DX**2
            lx = 1/DX**2
            uy = 1/DX**2
            ly = 1/DX**2
            
            if i == 0: 
                diag_val = -1/DX + 1j * K_AIR
                ux = 1/DX
                lx = uy = ly = 0
            elif i == NX - 1: 
                diag_val = 1/DX - 1j * K_AIR
                lx = -1/DX
                ux = uy = ly = 0
            elif j == 0: 
                diag_val = -1/DX + 1j * K_AIR
                uy = 1/DX
                lx = ux = ly = 0
            elif j == NY - 1: 
                diag_val = 1/DX - 1j * K_AIR
                ly = -1/DX
                lx = ux = uy = 0
            
            main_diag[idx] = diag_val
            if i < NX - 1:
                upper_x[idx] = ux
            if i > 0:
                lower_x[idx - NY] = lx
            if j < NY - 1:
                upper_y[idx] = uy
            if j > 0: 
                lower_y[idx - 1] = ly
            
    return sp.diags([main_diag, upper_x, lower_x, upper_y, lower_y], 
                    [0, NY, -NY, 1, -1], shape=(N, N), format='csc')

print("Assembling floor plan and PDE matrix...")
n_map = build_refractive_index_map()
wall_mask = np.real(n_map) > 1.0
A = build_system_matrix(n_map)
print("Pre-computing LU factorization for fast optimization...")
lu_solver = spla.splu(A)

def generate_source(xr, yr):
    b = np.zeros(NX * NY, dtype=complex)
    for i in range(NX):
        for j in range(NY):
            x, y = i * DX, j * DX
            dist_sq = (x - xr)**2 + (y - yr)**2
            if dist_sq < (4 * SIGMA)**2: 
                b[i * NY + j] = -A_AMP * np.exp(-dist_sq / (2 * SIGMA**2))
    return b

def objective_function(params):
    global current_best
    xr, yr = params
    
    # Must be > 0.5m from targets
    for pt in MEASUREMENT_POINTS:
        if np.linalg.norm([xr, yr] - pt) <= 0.5:
            return 1e9
            
    # Cannot be in walls
    ix, iy = int(xr/DX), int(yr/DX)
    if ix >= NX or iy >= NY or wall_mask[ix, iy]:
        return 1e9
        
    b = generate_source(xr, yr)
    u_vec = lu_solver.solve(b)
    u_grid = u_vec.reshape((NX, NY))
    
    total_signal = 0
    for pt in MEASUREMENT_POINTS:
        px, py = int(pt[0]/DX), int(pt[1]/DX)
        local_field = u_grid[max(0, px-1):px+2, max(0, py-1):py+2]
        total_signal += np.mean(np.abs(local_field)) 
        
    history_x.append(xr)
    history_y.append(yr)
    history_scores.append(total_signal)
    
    if total_signal > current_best:
        current_best = total_signal
    best_score_progression.append(current_best)
        
    return -total_signal

if __name__ == "__main__":
    bounds = [(0.5, 9.5), (0.5, 7.5)]
    
    print("Running Differential Evolution...")
    result = differential_evolution(
        objective_function, bounds, strategy='best1bin', 
        maxiter=30, popsize=10, disp=True, tol=1e-3
    )
    
    best_x, best_y = result.x
    print(f"\nOptimal Router Position: X = {best_x:.2f} m, Y = {best_y:.2f} m")

    plt.figure(figsize=(10, 8))
    # Create semi-transparent walls for the scatter plot
    wall_img_semi = np.zeros((NY, NX, 4))
    wall_img_semi[wall_mask.T, 0:3] = 0
    wall_img_semi[wall_mask.T, 3] = 0.3  # Alpha 0.3
    plt.imshow(wall_img_semi, origin='lower', extent=[0, DOMAIN_W, 0, DOMAIN_H])
    
    sc = plt.scatter(history_x, history_y, c=history_scores, cmap='viridis', 
                     s=40, edgecolors='k', alpha=0.8)
    plt.scatter(*MEASUREMENT_POINTS.T, c='red', marker='X', s=100, label='Targets')
    plt.scatter(best_x, best_y, c='magenta', marker='*', s=300, edgecolors='k', label='Best Pos')
    
    plt.title("DE Sampling Strategy & Optimization Landscape")
    plt.xlabel("X (m)"); plt.ylabel("Y (m)")
    plt.colorbar(sc, label='Total Signal Strength')
    plt.legend()
    plt.tight_layout()
    plt.show()
    plt.savefig("assignment_3/data/de_sampling_optimisation_landscape.png")

    # Convert the linear sum of amplitudes into a relative decibel scale
    # using 20 * log10(amplitude)
    best_score_db = 20 * np.log10(np.array(best_score_progression) + 1e-12)

    plt.figure(figsize=(8, 5))
    plt.plot(best_score_db, color='blue', linewidth=2)
    plt.title("Algorithm Convergence over Evaluations")
    plt.xlabel("Number of Function Evaluations")
    plt.ylabel("Best Aggregate Signal (dB)")
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.tight_layout()
    plt.show()
    plt.savefig("assignment_3/data/convergence_plot.png")


    b_opt = generate_source(best_x, best_y)
    u_opt = lu_solver.solve(b_opt).reshape((NX, NY))
    
    u_wave = np.real(u_opt) 
    
    u_ref = np.percentile(np.abs(u_opt), 99.5) 
    
    u_db = 20 * np.log10((np.abs(u_wave) + 1e-12) / u_ref)
    u_db = np.clip(u_db, a_min=-40, a_max=0)

    fig, ax = plt.subplots(figsize=(10, 8))
    
    im = ax.imshow(u_db.T, origin='lower', extent=[0, DOMAIN_W, 0, DOMAIN_H], 
                   cmap='jet', vmin=-40, vmax=0, interpolation='bilinear')
    
    wall_img_solid = np.zeros((NY, NX, 4))
    wall_img_solid[wall_mask.T, 0:3] = 0     # Set wall pixels to Black (RGB=0)
    wall_img_solid[wall_mask.T, 3] = 1.0     # Set ONLY wall pixels to fully opaque (Alpha=1.0)
    ax.imshow(wall_img_solid, origin='lower', extent=[0, DOMAIN_W, 0, DOMAIN_H])

    label_props = dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.6, edgecolor='none')
    text_props = dict(color='white', fontweight='bold', fontsize=9, ha='center', va='center', bbox=label_props)
    
    ax.text(2.5, 6.0, "Living Room", **text_props)
    ax.text(7.5, 5.5, "Bedroom 1", **text_props)
    ax.text(1.5, 1.5, "Kitchen", **text_props)
    ax.text(5.0, 1.5, "Hall", **text_props)
    ax.text(8.5, 1.5, "Bathroom", **text_props)

    ax.scatter(best_x, best_y, c='white', marker='*', s=300, 
               edgecolors='black', linewidths=1.5, label='WiFi Router', zorder=5)

    ax.set_title(f"WiFi Signal Coverage at 0.8 GHz (Scaled)\nOptimal Router Position: ({best_x:.2f}, {best_y:.2f}) m")
    ax.set_xlabel("X (meters)")
    ax.set_ylabel("Y (meters)")
    ax.set_xlim(0, 10); ax.set_ylim(0, 8)
    
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Signal Strength (dB)')
    ax.legend(loc='upper right')
    
    plt.tight_layout()
    plt.show()
    plt.savefig("assignment_3/data/wifi_signal_coverage_optimal_router_position.png")
    
    
    u_phase = np.angle(u_opt) 

    fig, ax = plt.subplots(figsize=(10, 8))
    
    im_phase = ax.imshow(u_phase.T, origin='lower', extent=[0, DOMAIN_W, 0, DOMAIN_H], 
                         cmap='twilight_shifted', interpolation='bilinear')
    
    ax.imshow(wall_img_solid, origin='lower', extent=[0, DOMAIN_W, 0, DOMAIN_H])

    ax.scatter(best_x, best_y, c='white', marker='*', s=300, edgecolors='black', zorder=5)
    ax.scatter(*MEASUREMENT_POINTS.T, c='black', marker='X', s=100)

    ax.set_title("Wave Phase Distribution\n(Shows wavefront propagation, refraction, and diffraction)")
    ax.set_xlabel("X (meters)")
    ax.set_ylabel("Y (meters)")
    
    cbar_phase = plt.colorbar(im_phase, ax=ax, fraction=0.046, pad=0.04)
    cbar_phase.set_label('Phase Angle (radians)')
    
    plt.tight_layout()
    plt.show()
    plt.savefig("assignment_3/data/wifi_phase_distribution.png")


    fig, ax = plt.subplots(figsize=(10, 8))
    
    # > -15 is Excellent, -15 to -25 is Fair, -25 to -40 is Poor/Dead Zone
    levels = [-40, -25, -15, 0]
    colors = ['#d7191c', '#fdae61', '#a6d96a']

    X_grid, Y_grid = np.meshgrid(np.linspace(0, DOMAIN_W, NX), np.linspace(0, DOMAIN_H, NY))
    
    cs = ax.contourf(X_grid, Y_grid, u_db.T, levels=levels, colors=colors, extend='min')
    
    ax.imshow(wall_img_solid, origin='lower', extent=[0, DOMAIN_W, 0, DOMAIN_H])
    
    ax.text(2.5, 6.0, "Living Room", **text_props)
    ax.text(7.5, 5.5, "Bedroom 1", **text_props)
    ax.text(1.5, 1.5, "Kitchen", **text_props)
    ax.text(5.0, 1.5, "Hall", **text_props)
    ax.text(8.5, 1.5, "Bathroom", **text_props)

    ax.scatter(best_x, best_y, c='white', marker='*', s=300, edgecolors='black', zorder=5)
    ax.scatter(*MEASUREMENT_POINTS.T, c='black', marker='X', s=100)

    ax.set_title("Categorical WiFi Coverage Zones\n(Green = Excellent, Orange = Fair, Red = Poor)")
    ax.set_xlabel("X (meters)")
    ax.set_ylabel("Y (meters)")
    
    cbar_zones = plt.colorbar(cs, ax=ax, fraction=0.046, pad=0.04)
    cbar_zones.set_label('Relative Signal Zones (dB)')
    
    plt.tight_layout()
    plt.show()
    plt.savefig("assignment_3/data/categorical_wifi_coverage_zones.png")