def wave_equation_1d(u0, c, dx, dt, t_max):
    """
    Solves the 1D wave equation using finite difference method.

    Parameters:
    u0 (numpy array): Initial displacement of the wave.
    c (float): Speed of the wave.
    dx (float): Spatial step size.
    dt (float): Time step size.
    t_max (float): Maximum time to simulate.

    Returns:
    numpy array: Displacement of the wave at each time step.
    """
    import numpy as np

    # Number of spatial points
    nx = len(u0)

    # Number of time steps
    nt = int(t_max / dt)

    # Initialize arrays to store wave displacement
    u = np.zeros((nt, nx))

    # Set initial condition
    u[0, :] = u0

    # Time-stepping loop
    for n in range(1, nt):
        for i in range(1, nx - 1):
            u[n, i] = 2 * u[n-1, i] - u[n-2, i] + (c * dt / dx) ** 2 * (u[n-1, i+1] - 2 * u[n-1, i] + u[n-1, i-1])

    return u