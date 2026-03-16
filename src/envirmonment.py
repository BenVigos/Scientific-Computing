from abc import ABC, abstractmethod
import numpy as np

class Environment(ABC):
    """Abstract base class for physical environments"""

    @abstractmethod
    def initial_condition(self, x, y):
        pass

    @abstractmethod
    def boundary_condition(self, x, y, t):
        pass

    @abstractmethod
    def source_term(self, x, y, t):
        pass

class KarmannVortex(Environment):
    """Karmann Vortex Street Environment"""

    def initial_condition(self, x, y):
        return (np.sin(np.pi * x) * np.sin(np.pi * y),
                -np.cos(np.pi * x) * np.cos(np.pi * y))

    def boundary_condition(self, x, y, t):
        return (0, 0)

    def source_term(self, x, y, t):
        return (0, 0)

class RoomWifi(Environment):
    """Room Wi-Fi Signal Propagation Environment"""

    def initial_condition(self, x, y):
        return 0

    def boundary_condition(self, x, y, t):
        return 0

    def source_term(self, x, y, t):
        return np.exp(-((x-5)**2 + (y-5)**2) / 2) * np.sin(2 * np.pi * t)