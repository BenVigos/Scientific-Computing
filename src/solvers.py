from abc import ABC, abstractmethod

class Solver(ABC):
    """Abstract base class for PDE solvers"""

    def __init__(self, environment):
        self.environment = environment

    @abstractmethod
    def solve(self):
        pass

class FEMSolver(Solver):
    """Finite Element Method Solver"""

    def __init__(self, environment, mesh, material):
        super().__init__(environment)
        self.mesh = mesh
        self.material = material

    def solve(self):
        print("Solving using FEM Solver...")
        return "FEM Solution"


class FDMSolver(Solver):
    """Finite Difference Method Solver"""

    def __init__(self, environment, grid, time_step):
        super().__init__(environment)
        self.grid = grid
        self.time_step = time_step

    def solve(self):
        print("Solving using FDM Solver...")
        return "FDM Solution"


class LBMSolver(Solver):
    """Lattice Boltzmann Method Solver"""

    def __init__(self, environment, lattice, relaxation_time):
        super().__init__(environment)
        self.lattice = lattice
        self.relaxation_time = relaxation_time

    def solve(self):
        print("Solving using LBM Solver...")
        return "LBM Solution"