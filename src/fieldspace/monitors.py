import matplotlib.pyplot as plt
import numpy as np

from algebra.field import Field
from discrete import DiscreteSpace

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
