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

def empty_sink(N):
    return np.zeros((N, N), dtype=bool)

def rectangle_sink(N, x0, x1, y0, y1):
    s = empty_sink(N)
    s[x0:x1, y0:y1] = True
    return s


def combine_sinks(*sinks):
    out = np.zeros_like(sinks[0], dtype=bool)
    for s in sinks:
        out |= m
    return out
