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
    c[0, :] = 1      
    c[-1, :] = 0  

    return c

def empty_sink(N):
    return np.zeros((N, N), dtype=bool)

def rectangle_sink(N, i0, i1, j0, j1):
    s = empty_sink(N)
    s[i0:i1, j0:j1] = True
    return s


def combine_sinks(*sinks): # do i end up using this or no lol
    out = np.zeros_like(sinks[0], dtype=bool)
    for s in sinks:
        out |= s
    return out

def empty_insulator(N):
    return np.zeros((N, N), dtype=bool)

def rectangle_insulator(N, i0, i1, j0, j1):
    m = empty_insulator(N)
    m[i0:i1, j0:j1] = True
    return m