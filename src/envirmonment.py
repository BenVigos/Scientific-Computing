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

    def __init__(self, viscosity=0.01, v0=0.12):
        self.viscosity = viscosity
        self.v0 = v0

        self.x_range = (0, 2.2) # meters
        self.y_range = (0, 0.41) # meters
        self.circle_center = (0.2, 0.2) # meters (cardinal point of the cylinder)
        self.circle_radius = 0.05 # meters

    def initial_condition(self, x, y):
        return (
            self.v0,
            0.001 * self.v0 * np.sin(2.0 * np.pi * y / self.y_range[1]),
        )

    def _region_masks(self, x, y, eps=1e-10):
        """Return vectorized geometric masks for boundaries and obstacle."""
        x_arr = np.asarray(x)
        y_arr = np.asarray(y)

        x_min, x_max = self.x_range
        y_min, y_max = self.y_range

        left = np.isclose(x_arr, x_min, atol=eps)
        right = np.isclose(x_arr, x_max, atol=eps)
        bottom = np.isclose(y_arr, y_min, atol=eps)
        top = np.isclose(y_arr, y_max, atol=eps)

        cx, cy = self.circle_center
        obstacle = (x_arr - cx) ** 2 + (y_arr - cy) ** 2 <= self.circle_radius ** 2

        wall = top | bottom | obstacle
        boundary = left | right | top | bottom
        fluid = ~obstacle
        interior = fluid & ~boundary

        return {
            "left": left,
            "right": right,
            "top": top,
            "bottom": bottom,
            "obstacle": obstacle,
            "wall": wall,
            "fluid": fluid,
            "interior": interior,
            "boundary": boundary,
        }

    def boundary_condition(self, x, y, t, eps=1e-10):
        """
        Mixed BCs:
        - Left inlet: u = v0 (Dirichlet), dv/dn = 0 (Neumann)
        - Top/bottom and cylinder: u = v = 0 (Dirichlet no-slip)
        - Right boundary: du/dn = dv/dn = 0 (Neumann outflow)
        """
        masks = self._region_masks(x, y, eps=eps)
        shape = np.asarray(x).shape

        zeros = np.zeros(shape, dtype=float)

        # Give wall BC priority on corner points (top-left / bottom-left).
        left_inlet = masks["left"] & ~masks["wall"]
        right_outlet = masks["right"] & ~masks["obstacle"]

        # u is x-velocity: prescribe inlet speed on left, no-slip on walls.
        u_dirichlet_mask = masks["wall"] | left_inlet
        u_neumann_mask = right_outlet

        # v is y-velocity: no-slip on walls, zero-gradient at inlet/outlet.
        v_dirichlet_mask = masks["wall"]
        v_neumann_mask = left_inlet | right_outlet

        u_dirichlet_value = np.where(left_inlet, self.v0, 0.0)

        return {
            "u": {
                "dirichlet_mask": u_dirichlet_mask,
                "dirichlet_value": u_dirichlet_value,
                "neumann_mask": u_neumann_mask,
                "neumann_value": zeros,
            },
            "v": {
                "dirichlet_mask": v_dirichlet_mask,
                "dirichlet_value": zeros,
                "neumann_mask": v_neumann_mask,
                "neumann_value": zeros,
            },
            "masks": masks,
        }

    def validate_condition_masks(self, nx=80, ny=40, t=0.0):
        """Return a small diagnostics dictionary to validate BC/mask consistency."""
        data = self.build_condition_masks(nx=nx, ny=ny, t=t)
        bc = data["bc"]
        masks = data["masks"]

        u_overlap = np.count_nonzero(bc["u"]["dirichlet_mask"] & bc["u"]["neumann_mask"])
        v_overlap = np.count_nonzero(bc["v"]["dirichlet_mask"] & bc["v"]["neumann_mask"])

        left_inlet = masks["left"] & ~masks["wall"]
        inlet_u_vals = data["u0"][left_inlet]

        return {
            "grid": (nx, ny),
            "n_fluid": int(np.count_nonzero(masks["fluid"])),
            "n_obstacle": int(np.count_nonzero(masks["obstacle"])),
            "n_left_inlet": int(np.count_nonzero(left_inlet)),
            "n_right_outlet": int(np.count_nonzero(bc["v"]["neumann_mask"])),
            "u_dirichlet_neumann_overlap": int(u_overlap),
            "v_dirichlet_neumann_overlap": int(v_overlap),
            "left_inlet_u_min": float(inlet_u_vals.min()) if inlet_u_vals.size else None,
            "left_inlet_u_max": float(inlet_u_vals.max()) if inlet_u_vals.size else None,
        }

    def build_condition_masks(self, nx, ny, t=0.0):
        """Create grid, initial-condition arrays, and BC masks for a given resolution."""
        x = np.linspace(*self.x_range, nx)
        y = np.linspace(*self.y_range, ny)
        X, Y = np.meshgrid(x, y, indexing="ij")

        dx = (self.x_range[1] - self.x_range[0]) / max(nx - 1, 1)
        dy = (self.y_range[1] - self.y_range[0]) / max(ny - 1, 1)
        eps = 0.5 * min(dx, dy) + 1e-12

        bc = self.boundary_condition(X, Y, t, eps=eps)
        masks = bc["masks"]

        u0, v0 = self.initial_condition(X, Y)
        # Enforce solid and Dirichlet values once so the solver starts from a consistent state.
        u0 = np.where(masks["obstacle"], 0.0, u0)
        v0 = np.where(masks["obstacle"], 0.0, v0)

        u0 = np.where(bc["u"]["dirichlet_mask"], bc["u"]["dirichlet_value"], u0)
        v0 = np.where(bc["v"]["dirichlet_mask"], bc["v"]["dirichlet_value"], v0)

        return {
            "x": x,
            "y": y,
            "X": X,
            "Y": Y,
            "dx": dx,
            "dy": dy,
            "u0": u0,
            "v0": v0,
            "bc": bc,
            "masks": masks,
        }

    def source_term(self, x, y, t):
        return (0, 0)

    def show(self):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(11, 2.8))

        x = np.linspace(*self.x_range, 100)
        y = np.linspace(*self.y_range, 100)
        X, Y = np.meshgrid(x, y, indexing="xy")
        U, V = self.initial_condition(X, Y)

        ax.quiver(X, Y, U, V)
        circle = plt.Circle(self.circle_center, self.circle_radius, color="r", fill=True)
        ax.add_patch(circle)

        ax.set_title("Karmann Vortex Street Initial Condition")
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.set_xlim(self.x_range)
        ax.set_ylim(self.y_range)
        ax.grid(True)

        fig.tight_layout()
        plt.show()


class RoomWifi(Environment):
    """Room Wi-Fi Signal Propagation Environment"""

    def initial_condition(self, x, y):
        return 0

    def boundary_condition(self, x, y, t):
        return 0

    def source_term(self, x, y, t):
        return np.exp(-((x - 5) ** 2 + (y - 5) ** 2) / 2) * np.sin(2 * np.pi * t)


def main():
    env = KarmannVortex(v0=0.12)

    diagnostics = env.validate_condition_masks(nx=120, ny=60)
    print("Boundary/mask diagnostics:")
    for key, value in diagnostics.items():
        print(f"  {key}: {value}")

    env.show()

    small = env.build_condition_masks(nx=60, ny=30)
    print("\nSmall-grid sanity check:")
    print(f"  u0 shape: {small['u0'].shape}")
    print(f"  v0 shape: {small['v0'].shape}")
    print(f"  obstacle cells: {int(np.count_nonzero(small['masks']['obstacle']))}")


if __name__ == "__main__":
    main()