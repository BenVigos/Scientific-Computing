import numpy as np

N = 50
L = 1.0
dx = L / N
dy = dx


D = 1.0
T_max = 0.2
dt = (dx**2) / (4 * D) * 0.9

c = np.zeros((N + 1, N + 1)) 


c[:, -1] = 1.0
c[:, 0]  = 0.0


filename = "diffusion_data.csv"
with open(filename, "w") as f:
    f.write("Time,X_Index,Y_Index,Concentration\n")

print(f"Starting simulation. Data will be saved to {filename}")


t = 0.0
iteration = 0
output_interval = 100

while t < T_max:
    c_new = c.copy()
    
    for j in range(1, N):
        for i in range(0, N):
            i_left = (i - 1) % N
            i_right = (i + 1) % N
            j_up = j + 1
            j_down = j - 1
            
            term = (c[i_right, j] + c[i_left, j] + c[i, j_up] + c[i, j_down] - 4 * c[i, j])
            c_new[i, j] = c[i, j] + (dt * D / (dx**2)) * term

    c = c_new
    
    c[N, :] = c[0, :]
    
    t += dt
    iteration += 1

    if iteration % output_interval == 0:
        print(f"Iteration {iteration}, Time: {t:.5f} - Writing to file...")
        with open(filename, "a") as f:
            mid_x = N // 2
            for j in range(N + 1):
                f.write(f"{t:.5f},{mid_x},{j},{c[mid_x, j]:.5f}\n")

print("Simulation finished.")