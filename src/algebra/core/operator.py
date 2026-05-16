from __future__ import annotations
from abc import ABC, abstractmethod
from typing import TypeVar
import numpy as np

from algebra.exceptions import ShapeMismatchError
from algebra.fieldshape import FieldShape


class Operator(ABC):
    def __init__(self, input_shape: FieldShape, output_shape: FieldShape):
        self._input_shape = input_shape
        self._output_shape = output_shape

    @property
    def input_shape(self) -> FieldShape:
        return self._input_shape

    @property
    def output_shape(self) -> FieldShape:
        return self._output_shape

    def apply(self, input_field: np.ndarray, output_field: np.ndarray):
        if input_field.shape != self.input_shape:
            raise ShapeMismatchError(
                (
                    f"input_field's shape: {input_field.shape} has to match operator"
                    f" input_shape: {self.input_shape}."
                )
            )
        if output_field.shape != self.output_shape:
            raise ShapeMismatchError(
                (
                    f"output_field's shape: {output_field.shape} has to match operator"
                    f" output_shape: {self.output_shape}."
                )
            )
        self._apply(input_field, output_field)

    @abstractmethod
    def copy(self) -> "Operator":
        pass

    @abstractmethod
    def _apply(self, input_field: np.ndarray, output_field: np.ndarray):
        pass

    def __neg__(self) -> "Operator":
        return NotImplemented

    def add(self, other: "Operator") -> "Operator":
        return NotImplemented

    def mul(self, other: "Operator") -> "Operator":
        return NotImplemented

    def div(self, other: "Operator") -> "Operator":
        return NotImplemented

    def scale(self, other: float) -> "Operator":
        return NotImplemented

    def scale_arr(self, other: np.ndarray) -> "Operator":
        return NotImplemented

    def __add__(self, other: "Operator") -> "Operator":
        if isinstance(other, type(self)):
            if self.input_shape != other.input_shape:
                raise ShapeMismatchError(
                    (
                        "Can only add operators of equal input_shape.",
                        f"self.input_shape: {self.input_shape},",
                        f"other.input_shape: {other.input_shape}.",
                    )
                )
            if self.output_shape != other.output_shape:
                raise ShapeMismatchError(
                    (
                        "Can only add operators of equal output_shape.",
                        f"self.output_shape: {self.output_shape},",
                        f"other.output_shape: {other.output_shape}.",
                    )
                )
            return self.add(other)
        return NotImplemented

    def __sub__(self, other: "Operator") -> "Operator":
        return self + (-other)

    def __mul__(self, other: "Operator" | float | np.ndarray) -> "Operator":
        if isinstance(other, type(self)):
            if self.input_shape != other.input_shape:
                raise ShapeMismatchError(
                    (
                        "Can only mul operators of equal input_shape.",
                        f"self.input_shape: {self.input_shape},",
                        f"other.input_shape: {other.input_shape}.",
                    )
                )
            if self.output_shape != other.output_shape:
                raise ShapeMismatchError(
                    (
                        "Can only mul operators of equal output_shape.",
                        f"self.output_shape: {self.output_shape},",
                        f"other.output_shape: {other.output_shape}.",
                    )
                )
            return self.mul(other)
        if isinstance(other, float):
            return self.scale(other)
        if isinstance(other, np.ndarray):
            if other.shape == ():
                return self.scale(float(other))
            if self.output_shape != other.shape:
                raise ShapeMismatchError(
                    (
                        "Can only scale operator elementwise by array that matches ",
                        "its output_shape. ",
                        f"self.output_shape: {self.output_shape}, ",
                        f"array.shape: {other.shape}.",
                    )
                )
            return self.scale_arr(other)
        return NotImplemented

    def __rmul__(self, other: float | np.ndarray) -> "Operator":
        return self.__mul__(other)

    def __truediv__(self, other: "Operator" | float | np.ndarray) -> "Operator":
        if isinstance(other, (float | np.ndarray)):
            return self * (1.0 / other)
        if isinstance(other, type(self)):
            return self.div(other)
        return NotImplemented


TOperator = TypeVar("TOperator", bound=Operator)
