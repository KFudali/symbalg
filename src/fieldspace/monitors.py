from matplotlib import animation
import matplotlib.pyplot as plt
import numpy as np

from algebra.field import Field
from discrete import DiscreteSpace
from discrete.field import DiscreteField

def plot_field_2d(field: Field[DiscreteSpace]):
    if field.space.ndim != 2:
        raise ValueError((
            "plot_field_2d only works with spaces of ndim == 2. ",
            f"Got field with space of shape: {field.space.shape}"
        ))
    X, Y = field.space.points()
    U = field.value().eval()
    comps = field.components 
    fig = plt.figure(figsize=(comps * 5,5))
    for comp in range(comps):
        ax1 = fig.add_subplot(comps, 1, comp + 1, projection='3d')
        surf1 = ax1.plot_surface(
            X, Y, U[comp], cmap='viridis', edgecolor='k', linewidth=0.5
        )
        ax1.set_title("Conjugate Gradient")
        ax1.set_xlabel("x")
        ax1.set_ylabel("y")
        ax1.set_zlabel("u")
        fig.colorbar(surf1, ax=ax1, shrink=0.6, aspect=10, label='u')
        plt.tight_layout()
        plt.show()

class FieldMonitor2D():
    def __init__(self, field: DiscreteField):
        if field.space.ndim != 2:
            raise ValueError((
                "FieldMonitor2D only works with spaces of ndim == 2. ",
                f"Got field with space of shape: {field.space.shape}"
            ))
        self._field = field
        self._field.space.time.advanceables.register(self)
        self._values = list[np.ndarray]()

    def advance(self):
        self._values.append(self._field.value().eval())

    def reset(self):
        self._values.clear()

    def animate(self):
        time = self._field.space.time.discrete_steps
        if len(time) != len(self._values):
            raise ValueError("Time step does not match monitor length")
        comps = self._field.components
        X, Y = self._field.space.points()
        fig = plt.figure(figsize=(comps * 5, 5))
        axes = []
        for comp in range(comps):
            ax = fig.add_subplot(comps, 1, comp + 1, projection="3d")
            axes.append(ax)
        plt.tight_layout()

        def update(step: int):
            for comp in range(comps):
                axes[comp].clear()
                axes[comp].plot_surface(
                    X, Y, self._values[step][comp],
                    cmap="viridis", edgecolor="k", linewidth=0.5
                )
                axes[comp].set_title(f"Field component: {comp} at time: {time[step]}")
                axes[comp].set_xlabel("x")
                axes[comp].set_ylabel("y")
                axes[comp].set_zlabel("u")

        animation.FuncAnimation(fig, update, frames=len(time), interval=200)
        plt.show()

