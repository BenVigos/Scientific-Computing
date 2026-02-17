import numpy as np
import matplotlib.pyplot as plt

# as described in 1.2
def make_grid(N):
    """
    Create NxN grid with boundary conditions:
    top = 1
    bottom = 0
    periodic 
    """
    c = np.zeros((N, N))
    c[:, 0] = 0       
    c[:, -1] = 1      

    return c
