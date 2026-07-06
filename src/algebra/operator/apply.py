from abc import ABC, abstractmethod
from typing import Callable, Type
import numpy as np
from algebra.space import Space, ShapeTransform

FieldApply = Callable[[int, np.ndarray, np.ndarray], None]


# class OperatorApplier(ABC):
#     def __init__(self, space: Space, apply: FieldApply):
#         self._space = space
#         self._apply = apply
#
#     @property
#     def space(self) -> Space:
#         return self._space
#
#     @abstractmethod
#     def apply(self, inp: np.ndarray, out: np.ndarray):
#         pass
#
#
# class LapOperatorApplier(OperatorApplier):
#     def apply(self, inp: np.ndarray, out: np.ndarray):
#         field_rank = len(inp.shape[: -self.space.ndim])
#         if field_rank == 0:
#             for ax in range(self.space.ndim):
#                 self._apply(ax, inp, out)
#         else:
#             for comp in range(inp.shape[0]):
#                 self.apply(inp[comp], out[comp])
#
#
# class DivOperatorApplier(OperatorApplier):
#     def apply(self, inp: np.ndarray, out: np.ndarray):
#         rank = len(inp.shape[: -self.space.ndim])
#         assert rank >= 1, "DivLikeOperator requires field rank >= 1"
#         assert (
#             inp.shape[rank - 1] == self.space.ndim
#         ), "DivLikeOperator requires the last rank axis to match space.ndim"
#         assert out.shape == inp.shape[: rank - 1] + inp.shape[rank:]
#         if rank == 1:
#             for ax in range(self.space.ndim):
#                 self._apply(ax, inp[ax], out)
#         else:
#             assert inp.shape[0] == out.shape[0], (
#                 "DivLikeOperator requires field and out to share leading " "rank dims"
#             )
#             for comp in range(inp.shape[0]):
#                 self.apply(inp[comp], out[comp])
#
#
# class GradOperatorApplier(OperatorApplier):
#     def apply(self, inp: np.ndarray, out: np.ndarray):
#         field_rank = len(inp.shape[: self.space.ndim])
#         if field_rank == 0:
#             for ax in range(self.space.ndim):
#                 self._apply(ax, inp, out[ax])
#         else:
#             for comp in range(out.shape[0]):
#                 self.apply(inp[comp], out[comp])
#
#
# APPLIERS: dict[ShapeTransform, Type[OperatorApplier]] = {
#     ShapeTransform.NONE: LapOperatorApplier,
#     ShapeTransform.REDUCE_RANK: DivOperatorApplier,
#     ShapeTransform.INCREASE_RANK: GradOperatorApplier,
# }
#
# def applier(
#     space: Space, transform: ShapeTransform, apply: FieldApply
# ) -> OperatorApplier:
#     return APPLIERS[transform](space, apply)


def lap_apply(space: Space, apply: FieldApply, inp: np.ndarray, out: np.ndarray):
    field_rank = len(inp.shape[: -space.ndim])
    if field_rank == 0:
        for ax in range(space.ndim):
            apply(ax, inp, out)
    else:
        for comp in range(inp.shape[0]):
            lap_apply(space, apply, inp[comp], out[comp])


def div_apply(space: Space, apply: FieldApply, inp: np.ndarray, out: np.ndarray):
    rank = len(inp.shape[: -space.ndim])
    assert rank >= 1, "DivLikeOperator requires field rank >= 1"
    assert (
        inp.shape[rank - 1] == space.ndim
    ), "DivLikeOperator requires the last rank axis to match space.ndim"
    assert out.shape == inp.shape[: rank - 1] + inp.shape[rank:]
    if rank == 1:
        for ax in range(space.ndim):
            apply(ax, inp[ax], out)
    else:
        assert inp.shape[0] == out.shape[0], (
            "DivLikeOperator requires field and out to share leading " "rank dims"
        )
        for comp in range(inp.shape[0]):
            div_apply(space, apply, inp[comp], out[comp])


def grad_apply(space: Space, apply: FieldApply, inp: np.ndarray, out: np.ndarray):
    field_rank = len(inp.shape[: space.ndim])
    if field_rank == 0:
        for ax in range(space.ndim):
            apply(ax, inp, out[ax])
    else:
        for comp in range(out.shape[0]):
            grad_apply(space, apply, inp[comp], out[comp])


APPLY: dict[
    ShapeTransform, Callable[[Space, FieldApply, np.ndarray, np.ndarray], None]
] = {
    ShapeTransform.NONE: lap_apply,
    ShapeTransform.REDUCE_RANK: div_apply,
    ShapeTransform.INCREASE_RANK: grad_apply,
}
