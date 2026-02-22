import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np

N = 50
L = 1.0
D = 1.0
dx = L / N
dt = 0.2 * (dx**2) / (4 * D)


x = np.linspace(0, L, N+1)
y = np.linspace(0, L, N+1)
X, Y = np.meshgrid(x, y)

c = np.zeros((N+1, N+1))

c[-1, :] = 1.0
c[0, :]  = 0.0

def update_step(c_curr):
    """
    Performs one time-step update using Finite Difference
    
    Parameters:
    - c_curr: Current concentration profile (2D array)
    Returns:
    - c_new: Updated concentration profile after one time step (2D array)
    
    """
    c_new = c_curr.copy()
    
    c_up    = c_curr[2:, :]
    c_down  = c_curr[:-2, :]
    
    c_left  = np.roll(c_curr, 1, axis=1)
    c_right = np.roll(c_curr, -1, axis=1)
    
    c_left_in  = c_left[1:-1, :]
    c_right_in = c_right[1:-1, :]
    c_center   = c_curr[1:-1, :]
    
    factor = (dt * D) / (dx**2)
    c_new[1:-1, :] = c_center + factor * (
        c_right_in + c_left_in + c_up + c_down - 4 * c_center
    )
    
    c_new[-1, :] = 1.0
    c_new[0, :]  = 0.0
    
    return c_new

fig, ax = plt.subplots(figsize=(7, 6))

im = ax.imshow(c, origin="lower", extent=[0, 1, 0, 1], cmap="inferno", vmin=0, vmax=1)
fig.colorbar(im, label="Concentration (c)")
ax.set_title("2D Diffusion Equation Evolution")
ax.set_xlabel("x")
ax.set_ylabel("y")

time_template = "Time = {:.4f} s"
time_text = ax.text(0.05, 0.9, "", transform=ax.transAxes, color="white")

current_c = c.copy()
current_time = 0.0
equilibrium_reached = False

def animate(frame):
    global current_c, current_time, equilibrium_reached
    
    steps_per_frame = 20
    for _ in range(steps_per_frame):
        current_c = update_step(current_c)
        current_time += dt
        
    im.set_data(current_c)
    time_text.set_text(time_template.format(current_time))
    
    error = np.mean(np.abs(current_c[:, N//2] - y))
    if error < 0.01: 
        time_text.set_text(f"Equilibrium Reached (t={current_time:.3f})")
        if not equilibrium_reached:
            equilibrium_reached = True
            fig.savefig("equilibrium_state.png", dpi=150, bbox_inches="tight")
            print(f"Equilibrium state saved as 'equilibrium_state.png' at t={current_time:.3f}s")
            ani.event_source.stop()
        
    return im, time_text

ani = animation.FuncAnimation(fig, animate, frames=200, interval=50, blit=True)

plt.show()