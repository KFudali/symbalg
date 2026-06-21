import matplotlib.pyplot as plt

from algebra.field import Field
from discrete.core.discretization import Discretization


class MonitorFactory:
    def __init__(self, discrete: Discretization):
        self._discrete = discrete

    def plot_field_2d(self, field: Field, name: str = "F"):
        assert (
            field.space.ndim == 2
        ), f"plot_field_2d requires a 2D space, got ndim={field.space.ndim}"
        x, y = self._discrete.points()
        u = field.value().eval()
        components = field.fieldshape.components

        if len(components) == 0:
            figures_count = 1
            sub_shape: tuple[int, ...] = ()
        else:
            figures_count = components[0]
            sub_shape = components[1:]

        sub_count = 1
        for c in sub_shape:
            sub_count *= c

        for fig_idx in range(figures_count):
            fig = plt.figure(figsize=(sub_count * 5, 5))
            for sub_idx in range(sub_count):
                ax = fig.add_subplot(1, sub_count, sub_idx + 1)

                if len(components) == 0:
                    data = u
                    title = name
                else:
                    sub_indices: tuple[int, ...] = ()
                    remaining = sub_idx
                    for dim in reversed(sub_shape):
                        sub_indices = (remaining % dim,) + sub_indices
                        remaining //= dim
                    full_index = (fig_idx, *sub_indices)
                    data = u[full_index]
                    title = name + "".join(f"[{i}]" for i in full_index)

                contour = ax.contourf(x, y, data, levels=50, cmap="viridis")
                ax.set_aspect("equal")
                ax.set_title(title)
                ax.set_xlabel("x")
                ax.set_ylabel("y")
                fig.colorbar(contour, ax=ax, shrink=0.8)
            plt.tight_layout()

    def show(self):
        plt.show()

    def monitor(self, field: Field):
        pass


# from matplotlib import animation
# from matplotlib.gridspec import GridSpec
# import numpy as np
#
# class FieldMonitor2D:
#     def __init__(self, field: DiscreteField):
#         if field.space.ndim != 2:
#             raise ValueError(
#                 (
#                     "FieldMonitor2D only works with spaces of ndim == 2. ",
#                     f"Got field with space of shape: {field.space.shape}",
#                 )
#             )
#         self._field = field
#         self._field.space.time.advanceables.register(self)
#         self._values = list[np.ndarray]()
#
#     def advance(self):
#         self._values.append(self._field.value().eval())
#
#     def reset(self):
#         self._values.clear()
#
#     def animate(self):
#         time = self._field.space.time.discrete_steps[:-1]
#         if len(time) != len(self._values):
#             raise ValueError("Time step does not match monitor length")
#         comps = self._field.components
#         X, Y = self._field.space.points()
#
#         fig = plt.figure(figsize=(5 * comps, 5))
#         gs = GridSpec(1, comps, figure=fig, wspace=0.3)
#
#         data = np.array(self._values)
#         vmin = data.min()
#         vmax = data.max()
#
#         axes = []
#         for comp in range(comps):
#             ax = fig.add_subplot(gs[comp])
#             axes.append(ax)
#
#         for comp in range(comps):
#             contour = axes[comp].contourf(
#                 X, Y, data[0, comp], levels=50, vmin=vmin, vmax=vmax, cmap="viridis"
#             )
#             axes[comp].set_aspect("equal")
#             axes[comp].set_title(f"Component {comp}")
#             axes[comp].set_xlabel("x")
#             axes[comp].set_ylabel("y")
#             fig.colorbar(contour, ax=axes[comp])
#
#         def update(step: int):
#             for comp in range(comps):
#                 axes[comp].clear()
#                 axes[comp].contourf(
#                     X,
#                     Y,
#                     data[step, comp],
#                     levels=50,
#                     vmin=vmin,
#                     vmax=vmax,
#                     cmap="viridis",
#                 )
#                 axes[comp].set_aspect("equal")
#                 axes[comp].set_title(f"Component {comp} at t = {time[step]:.3f}")
#                 axes[comp].set_xlabel("x")
#                 axes[comp].set_ylabel("y")
#
#         anim = animation.FuncAnimation(fig, update, frames=len(time), interval=50)
#         plt.show()
