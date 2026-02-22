import matplotlib.pyplot as plt
import numpy as np
from scipy.special import erfc

D = 1.0
N = 100
L = 1.0
dx = L / N
dy = dx

dt = 0.2 * (dx**2) / (4 * D)

target_times = [0.001, 0.01, 0.1, 1.0]
captured_data = {}


x = np.linspace(0, L, N+1)
y = np.linspace(0, L, N+1)
c = np.zeros((N+1, N+1))


c[:, -1] = 1.0
c[:, 0]  = 0.0

def get_analytical_c(y_arr, t, D_val=1.0):
    """
    Computes the analytical solution using Eq (9).
    Summation of erfc terms.
    
    Parameters:
    - y_arr: Array of y positions.
    - t: Time at which to evaluate the solution.
    Returns:
    - c_anal: Analytical concentration profile at time t.
    """
    c_anal = np.zeros_like(y_arr)
 
    for i in range(20):
        denom = 2 * np.sqrt(D_val * t)
        term1 = erfc((1.0 - y_arr + 2*i) / denom)
        
        term2 = erfc((1.0 + y_arr + 2*i) / denom)
        
        c_anal += (term1 - term2)
    return c_anal

t_current = 0.0

targets = sorted(target_times)
target_idx = 0

print("Starting simulation...")

while target_idx < len(targets):
    c_new = c.copy()
    
    c_up    = c[:, 2:]
    c_down  = c[:, :-2]
    c_left  = np.roll(c, 1, axis=0)
    c_right = np.roll(c, -1, axis=0)

    c_center = c[:, 1:-1]
    c_left_in = c_left[:, 1:-1]
    c_right_in = c_right[:, 1:-1]
    
    factor = (dt * D) / (dx**2)
    
    c_new[:, 1:-1] = c_center + factor * (
        c_right_in + c_left_in + c_up + c_down - 4*c_center
    )
    
    c_new[:, -1] = 1.0
    c_new[:, 0]  = 0.0
    
    c = c_new
    t_current += dt
    
    if t_current >= targets[target_idx]:
        mid_x_idx = N // 2
        captured_data[targets[target_idx]] = c[mid_x_idx, :].copy()
        print(f"Captured data for t = {targets[target_idx]}")
        target_idx += 1

plt.figure(figsize=(10, 7))

colors = ["blue", "green", "orange", "red"]
diff = []
for i, t_val in enumerate(targets):
    sim_y = captured_data[t_val]

    anal_y = get_analytical_c(y, t_val, D)
    diff.append(np.mean(np.abs(sim_y - anal_y)))
    
    plt.plot(y, sim_y, "o", label=f"Sim t={t_val}", markevery=4, color=colors[i], markerfacecolor=colors[i])
    
    plt.plot(y, anal_y, "-", label=f"Analytic t={t_val}", color=colors[i], linewidth=1.5)

print("Average absolute differences between simulation and analytical:")
for t_val, d in zip(targets, diff, strict=False):
    print(f"t={t_val}: {d:.5f}")

plt.title("Comparison: Analytical Eq(9) vs Finite Difference Eq(7)")
plt.xlabel("y (Position)")
plt.ylabel("Concentration c(y,t)")
plt.legend()
plt.grid(True)
# plt.xlim(0, 1.0)
# plt.ylim(0, 1.05)
plt.show()